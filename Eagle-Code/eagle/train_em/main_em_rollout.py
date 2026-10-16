"""
eagle/train/main_em_rollout.py -- EM multi-drafter training over FULL speculative-decoding rollouts.

For every prompt we run the real SD loop with the live teacher until EOS / max_new:
    cycle:  K drafters draft gamma tokens (on-policy, with grad)
            teacher verifies each chain (separate small cached forwards)
            E-step : winner = argmax_k AL_k (+ balancing bias)
            M-step : winner gets  v_w*SmoothL1(feat) + p_w*DraftOPD-CE ; router gets CE(winner)
            advance: context moves forward by (AL_adv + 1) tokens (accepted + teacher bonus token)
    -> next cycle starts from the NEW context, so later blocks see the contexts deployment would produce.

Each cycle = one train step (loss.backward() immediately; optimizer steps every --cycles_per_update cycles).
Single process / single GPU (bs=1 rollout). Data files are the same .ckpt files as before; only
input_ids + loss_mask are used (prompt = tokens before the first loss_mask==1 token).

Assumes cnets1.Model.forward(hidden, input_ids=..., past_key_values=..., use_cache=True) -> (hidden, kv).
Needs transformers with DynamicCache.crop.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from ..model.configs import EConfig
from ..model.cnets1 import Model
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, get_linear_schedule_with_warmup
from safetensors import safe_open
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import optim
import argparse
import json
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True)
parser.add_argument('--ea_path', required=True)
parser.add_argument('--base_config', default="config.json")
parser.add_argument('--ea_config', required=True)
parser.add_argument('--tmpdir', required=True)
parser.add_argument('--cpdir', required=True)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--num_drafters', type=int, default=4)
parser.add_argument('--gamma', type=int, default=5)
parser.add_argument('--max_prompt', type=int, default=512)
parser.add_argument('--max_new', type=int, default=256)
parser.add_argument('--temperature', type=float, default=0.0, help="0 => greedy")
parser.add_argument('--advance', default='winner', choices=['winner', 'router', 'random'])
parser.add_argument('--cycles_per_update', type=int, default=8)
parser.add_argument('--total_updates', type=int, default=100000)
parser.add_argument('--num_warmup_steps', type=int, default=2000)
parser.add_argument('--random_assign_cycles', type=int, default=4000)
parser.add_argument('--bias_lr', type=float, default=1e-3)
parser.add_argument('--ce_w', type=float, default=0.1)
parser.add_argument('--feature_w', type=float, default=1.0)
parser.add_argument('--router_w', type=float, default=1.0)
parser.add_argument('--grad_clip', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--eval_prompts', type=int, default=8)
parser.add_argument('--eval_max_new_tokens', type=int, default=128)
parser.add_argument('--eval_every_rollouts', type=int, default=500)
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()


torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
random.seed(0)
dev = torch.device("cuda")
num_drafters, GAMMA = args.num_drafters, args.gamma
GREEDY = args.temperature <= 0
TAU = 1.0 if GREEDY else args.temperature
def AC(): return torch.autocast("cuda", dtype=torch.bfloat16)


if args.wandb:
    import wandb
    wandb.init(project="em-multidrafter-rollout", config=vars(args))


def log(d): return wandb.log(d) if args.wandb else None


# ------------------------------------------------------------------ frozen head + live teacher
baseconfig = AutoConfig.from_pretrained(args.basepath)              # AutoConfig instead of EConfig since its teacher
head = nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
try:
    idx = json.load(open(os.path.join(args.basepath, "model.safetensors.index.json")))
    with safe_open(os.path.join(args.basepath, idx["weight_map"]["lm_head.weight"]), framework="pt", device="cpu") as f:
        head.weight.data = f.get_tensor("lm_head.weight").float()
except Exception:
    idx = json.load(open(os.path.join(args.basepath, "pytorch_model.bin.index.json")))
    head.weight.data = torch.load(os.path.join(args.basepath, idx["weight_map"]["lm_head.weight"]))["lm_head.weight"].float()
head.eval().requires_grad_(False).to(dev)

teacher = AutoModelForCausalLM.from_pretrained(args.basepath, torch_dtype=torch.bfloat16).to(dev).eval()
teacher.requires_grad_(False)
_eos = teacher.generation_config.eos_token_id
EOS = set(_eos if isinstance(_eos, list) else [_eos])


def t_head(features):               # teacher logits from teacher features
    return head(features.to(head.weight.dtype)).float()


def sample_logits(logits):               # logits [V] float -> scalar long tensor
    return logits.argmax(-1) if GREEDY else torch.multinomial(F.softmax(logits / TAU, -1), 1)[0]


# ------------------------------------------------------------------ drafters + router
class Router(nn.Module):
    # Predict chosen drafter from hidden embeddings of teacher model
    def __init__(self, hidden_dim, num_drafters, mid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(), nn.Linear(mid, num_drafters))

    def forward(self, x):
        return self.net(x)


class MultiDrafter(nn.Module):
    def __init__(self, config, basepath, num_drafters, hidden_dim):
        super().__init__()
        ds = [Model(config, load_emb=(k == 0), path=basepath) for k in range(num_drafters)]
        for d in ds[1:]:
            d.embed_tokens = ds[0].embed_tokens          # shared frozen embedding
        self.drafters = nn.ModuleList(ds)
        self.router = Router(hidden_dim, num_drafters)


def draft_chain(drafter, h_s, root, pkv):
    """h_s [1,1,H] teacher feature of last committed-with-feature token, root [1,1] = pending token.
    Returns tokens [1,g], logits [1,g,V], feats [1,g,H]; feats[:, i] ~ teacher feature at position root+i."""
    out, p = drafter(h_s, input_ids=root, past_key_values=pkv, use_cache=True)
    f = out[:, -1:]
    toks, lgs, fts = [], [], []
    for i in range(GAMMA):
        lg = head(f)
        tok = lg.argmax(-1) if GREEDY else torch.multinomial(F.softmax(lg[:, 0].float() / TAU, -1), 1)
        fts.append(f)
        lgs.append(lg)
        toks.append(tok)
        if i < GAMMA - 1:
            out, p = drafter(f, input_ids=tok, past_key_values=p, use_cache=True)
            f = out[:, -1:]
    return torch.cat(toks, 1), torch.cat(lgs, 1), torch.cat(fts, 1)


model = MultiDrafter(EConfig.from_pretrained(args.base_config), args.basepath, num_drafters, baseconfig.hidden_size).to(dev)
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95))
scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.total_updates)
smooth_l1 = nn.SmoothL1Loss(reduction="none")


# ------------------------------------------------------------------ EM pieces
@torch.no_grad()
def acceptance(t_lg, s_lg, toks):
    """t_lg,s_lg [K,g,V]; toks [K,g] -> AL [K], expected AL [K]"""
    logp_t = F.log_softmax(t_lg / TAU, -1)
    logp_s = F.log_softmax(s_lg.float() / TAU, -1)
    if GREEDY:
        acc = toks == t_lg.argmax(-1)
    else:
        lt, ls = logp_t.gather(-1, toks[..., None])[..., 0], logp_s.gather(-1, toks[..., None])[..., 0]
        acc = torch.rand_like(lt) < (lt - ls).exp().clamp(max=1.0)
    AL = acc.long().cumprod(-1).sum(-1)
    expAL = torch.minimum(logp_t.exp(), logp_s.exp()).sum(-1).cumprod(-1).sum(-1)
    return AL, expAL


def draft_opd_ce(t_lg, s_lg, AL):
    """[1,g,V] x2, AL [1]: forward KL on accepted positions, reverse KL on the first rejected one."""
    pos = torch.arange(t_lg.shape[1], device=t_lg.device)[None]
    fwd_m, rev_m = pos < AL[:, None], pos == AL[:, None]
    logp_t = F.log_softmax(t_lg / TAU, -1)
    logp_s = F.log_softmax(s_lg.float() / TAU, -1)
    fwd = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    rev = (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    return fwd * fwd_m + rev * rev_m, fwd_m | rev_m


class Balancer:
    """aux-loss-free balancing: bias on selection score, driven by an EMA of usage."""

    def __init__(self, num_drafters, lr, decay=0.99):
        self.bias = torch.zeros(num_drafters, device=dev)
        self.ema = torch.full((num_drafters,), 1.0 / num_drafters, device=dev)
        self.lr, self.decay, self.step = lr, decay, 0

    def select(self, score):
        if self.step < args.random_assign_cycles:
            return random.randrange(num_drafters)
        return int((score + self.bias).argmax())

    def update(self, w):
        oh = torch.zeros(num_drafters, device=dev)
        oh[w] = 1
        self.ema = self.decay * self.ema + (1 - self.decay) * oh
        self.bias += self.lr * torch.sign(1.0 / num_drafters - self.ema)
        self.step += 1


balancer = Balancer(num_drafters, args.bias_lr)


# ------------------------------------------------------------------ one full SD rollout
def run_rollout(prompt, train, mode, max_new):
    """mode: which drafter's accepted tokens advance the context:
       train: 'winner' | 'router' | 'random';  eval: 'oracle' | 'router' | 'k0'..'k{K-1}'"""
    global n_cycles
    ids = prompt.to(dev)[None]                               # [1,n]
    n = ids.shape[1]
    s = n - 1
    with torch.no_grad():
        # teacher prefill: features of the prompt, KV cache, first generated token (= first root)
        out = teacher.model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        cache, feats = out.past_key_values, out.last_hidden_state                 # feats [1,n,H]
        root = sample_logits(t_head(feats[0, -1]))[None, None]                    # [1,1]
        # drafter committed KV: pairs (h_t, x_{t+1}) for t = 0..s-1 (no grad)
        pkv = []
        with AC():
            for drafter in model.drafters:
                pkv.append(drafter(feats[:, :s], input_ids=ids[:, 1:s + 1], use_cache=True)[1] if s > 0 else None)
    h_s = feats[:, s:s + 1]
    n_new, cycles, tok_sum = 1, 0, 0
    sumAL = torch.zeros(num_drafters)
    done = int(root) in EOS

    while n_new < max_new and not done:
        # ---------------- draft (with grad iff train)
        with torch.set_grad_enabled(train), AC():
            outs = [draft_chain(drafter, h_s, root, pkv[k]) for k, drafter in enumerate(model.drafters)]
            r_lg = model.router(h_s[:, 0].float())                                # [1,K]
        toks = torch.cat([o[0] for o in outs], 0)                                 # [K,g]
        s_lg = torch.cat([o[1] for o in outs], 0)                                 # [K,g,V]

        # ---------------- teacher verification (one cached forward per chain, cache cropped back)
        with torch.no_grad():
            base = cache.get_seq_length()
            t_ft = []
            for k in range(num_drafters):
                o = teacher.model(input_ids=torch.cat([root, toks[k:k + 1]], 1), past_key_values=cache,
                                  use_cache=True).last_hidden_state
                cache.crop(base)
                t_ft.append(o[0])
            t_ft = torch.stack(t_ft)                                              # [K,g+1,H]
            t_lg = t_head(t_ft)                                                   # [K,g+1,V]
            AL, expAL = acceptance(t_lg[:, :GAMMA], s_lg.detach(), toks)

        # ---------------- E-step + M-step
        score = AL.float() + 1e-2 * expAL
        w = balancer.select(score)
        if train:
            ce, mask = draft_opd_ce(t_lg[w:w + 1, :GAMMA], s_lg[w:w + 1], AL[w:w + 1])
            ft = smooth_l1(outs[w][2].float(), t_ft[w:w + 1, :GAMMA].float()).mean(-1)
            loss_ce = (ce * mask).sum() / mask.sum().clamp(min=1)
            loss_ft = (ft * mask).sum() / mask.sum().clamp(min=1)
            loss_r = F.cross_entropy(r_lg, torch.tensor([w], device=dev))
            loss = (args.feature_w * loss_ft + args.ce_w * loss_ce + args.router_w * loss_r) / args.cycles_per_update
            loss.backward()                                                       # graph freed every cycle
            balancer.update(w)
            n_cycles += 1
            if n_cycles % args.cycles_per_update == 0:
                nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            log({"train/loss_feat": loss_ft.item(), "train/loss_ce": loss_ce.item(), "train/loss_router": loss_r.item(),
                 "train/router_acc": float(int(r_lg.argmax()) == w), "train/winner": w,
                 **{f"train/usage_ema_k{k}": balancer.ema[k].item() for k in range(num_drafters)}})

        # ---------------- who advances the context?
        if mode == 'winner':
            a = w
        elif mode == 'router':
            a = int(r_lg.argmax()) if balancer.step >= args.random_assign_cycles or not train else w
        elif mode == 'random':
            a = random.randrange(num_drafters)
        elif mode == 'oracle':
            a = int(AL.argmax())
        else:
            a = int(mode[1:])
        AL_a = int(AL[a])

        # ---------------- bonus token from the teacher (proper residual sampling when sampling)
        with torch.no_grad():
            row = t_lg[a, AL_a]
            if GREEDY or AL_a == GAMMA:
                bonus = sample_logits(row)
            else:
                p_t = F.softmax(row / TAU, -1)
                p_s = F.softmax(s_lg[a, AL_a].float() / TAU, -1)
                res = (p_t - p_s).clamp(min=0)
                bonus = torch.multinomial(res / res.sum(), 1)[0] if res.sum() > 0 else sample_logits(row)

            # ---------------- commit: teacher cache, drafter KV (all K, no grad), new state
            new_tok = torch.cat([root[0], toks[a, :AL_a]])[None]                  # [1,AL+1]  root, d_1..d_AL
            teacher.model(input_ids=new_tok, past_key_values=cache, use_cache=True)
            new_feat = torch.cat([h_s, t_ft[a:a + 1, :AL_a]], 1)                  # [1,AL+1,H] h_s, h(root), h(d_1..d_{AL-1})
            with AC():
                for k, d in enumerate(model.drafters):
                    pkv[k] = d(new_feat, input_ids=new_tok, past_key_values=pkv[k], use_cache=True)[1]
            h_s = t_ft[a:a + 1, AL_a:AL_a + 1]                                    # feature at last accepted token
            root = bonus.view(1, 1)

        n_new += AL_a + 1
        tok_sum += AL_a + 1
        cycles += 1
        sumAL += AL.float().cpu()
        done = any(int(t) in EOS for t in toks[a, :AL_a]) or int(bonus) in EOS

    return {"tokens": tok_sum, "cycles": max(cycles, 1), "AL": (sumAL / max(cycles, 1)).tolist()}


# ------------------------------------------------------------------ data: prompts = tokens before first response token
class PromptDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(self.paths[i])
        nz = (torch.as_tensor(d['loss_mask']).view(-1) == 1).nonzero()
        if len(nz) == 0:
            return None
        n = int(nz[0])
        return d['input_ids'].view(-1).long()[:n] if 2 <= n <= args.max_prompt else None


files = [os.path.join(r, f) for r, _, fs in os.walk(args.tmpdir) for f in fs]
split = int(len(files) * 0.95)
train_loader = DataLoader(PromptDataset(files[:split]), batch_size=None, shuffle=True, num_workers=2)
test_prompts = [p for p in DataLoader(PromptDataset(files[split:split + 4 * args.eval_prompts]), batch_size=None,
                                      num_workers=2) if p is not None][: args.eval_prompts]


@torch.no_grad()
def evaluate():
    res = {}
    for mode in ['router', 'oracle'] + [f'k{k}' for k in range(num_drafters)]:
        tok = cyc = 0
        for p in test_prompts:
            st = run_rollout(p, False, mode, args.eval_max_new_tokens)
            tok += st["tokens"]
            cyc += st["cycles"]
        res[f"eval/tokens_per_cycle_{mode}"] = tok / max(cyc, 1)
    return res


def save_all(tag):
    out = os.path.join(args.cpdir, tag)
    for k, d in enumerate(model.drafters):
        p = os.path.join(out, f"drafter_{k}")
        os.makedirs(p, exist_ok=True)
        torch.save(d.state_dict(), os.path.join(p, "pytorch_model.bin"))
        shutil.copy(args.base_config, os.path.join(p, "config.json"))
    torch.save({"router": model.router.state_dict(), "bias": balancer.bias.cpu()}, os.path.join(out, "router.pt"))


# ------------------------------------------------------------------ train
n_cycles, n_roll = 0, 0
model.train()
for epoch in range(args.epochs):
    for prompt in tqdm(train_loader):
        if prompt is None:
            continue
        st = run_rollout(prompt, True, args.advance, args.max_new)
        n_roll += 1
        log({"train/tokens_per_cycle": st["tokens"] / st["cycles"],
             **{f"train/AL_k{k}": st["AL"][k] for k in range(num_drafters)}})
        if n_roll % args.eval_every_rollouts == 0:
            model.eval()
            ev = evaluate()
            model.train()
            print(json.dumps(ev))
            log(ev)
            save_all(f"rollout_{n_roll}")
save_all("final")
