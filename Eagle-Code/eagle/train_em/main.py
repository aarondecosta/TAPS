"""
eagle/train/main_em.py  --  EM-style multi-drafter, on-policy (Draft-OPD) training for EAGLE-1 (cnets1.Model).

Per step:
  1. sample one block start `s` for the whole micro-batch (shared prefix length => no padding headaches)
  2. every drafter k rolls out `gamma` tokens on-policy from the cached teacher features of the prefix
  3. the live teacher verifies every chain  -> accepted length AL[k, b]
  4. E-step : winner[b] = argmax_k (AL + eps*expected_AL + balancing_bias)
  5. M-step : winner gets  v_w * SmoothL1(features)  +  p_w * DraftOPD-CE
              (forward KL on accepted positions, reverse KL on first rejected position)
  6. router : CE(router(h_s), winner)

Assumes cnets1.Model.forward(hidden_states, input_ids=..., past_key_values=..., use_cache=True)
returns (out_hidden, past_key_values)  -- exactly how getkacc() in the original main.py calls it.
Data files: same .ckpt files as the original script (dict with input_ids [n], hidden_state [n,H], loss_mask [n]).
"""
import argparse
import json
import os
import random
import shutil

parser = argparse.ArgumentParser(description='em-multidrafter')
parser.add_argument('--basepath', type=str, required=True)
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--tmpdir', type=str, required=True)
parser.add_argument('--cpdir', type=str, required=True)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--num_drafters', type=int, default=4)
parser.add_argument('--gamma', type=int, default=5)
parser.add_argument('--max_len', type=int, default=1024)
parser.add_argument('--min_prefix', type=int, default=16)
parser.add_argument('--temperature', type=float, default=0.0, help="0 => greedy drafting/verification")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--num_warmup_steps', type=int, default=2000)
parser.add_argument('--random_assign_steps', type=int, default=2000, help="uniform-random winners at the start")
parser.add_argument('--bias_lr', type=float, default=1e-3)
parser.add_argument('--p_w', type=float, default=0.1)
parser.add_argument('--v_w', type=float, default=1.0)
parser.add_argument('--router_w', type=float, default=1.0)
parser.add_argument('--router_tau', type=float, default=0.0, help=">0 => soft-posterior router targets")
parser.add_argument('--feat_mask_all', action='store_true', help="feature loss on all gamma positions")
parser.add_argument('--grad_clip', type=float, default=0.5)
parser.add_argument('--eval_batches', type=int, default=20)
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from safetensors import safe_open
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup

from ..model.cnets1 import Model
from ..model.configs import EConfig

torch.backends.cuda.matmul.allow_tf32 = True
set_seed(0)
accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps)
dev = accelerator.device
K, GAMMA = args.num_drafters, args.gamma
GREEDY = args.temperature <= 0
TAU = 1.0 if GREEDY else args.temperature

if args.wandb and accelerator.is_main_process:
    import wandb
    wandb.init(project="em-multidrafter", config=vars(args))


def log(d):
    if args.wandb and accelerator.is_main_process:
        wandb.log(d)


# ----------------------------------------------------------------------------- frozen target LM head
baseconfig = AutoConfig.from_pretrained(args.basepath)
head = nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json")) as f:
        head_path = json.load(f)["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as f:
        head.weight.data = f.get_tensor("lm_head.weight").float()
except Exception:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json")) as f:
        head_path = json.load(f)["weight_map"]["lm_head.weight"]
    head.weight.data = torch.load(os.path.join(args.basepath, head_path))["lm_head.weight"].float()
head.eval().requires_grad_(False).to(dev)

# ----------------------------------------------------------------------------- live teacher (verification only)
teacher = AutoModelForCausalLM.from_pretrained(args.basepath, torch_dtype=torch.bfloat16).to(dev).eval()
teacher.requires_grad_(False)


# ----------------------------------------------------------------------------- data
class RawDataset(Dataset):
    def __init__(self, paths, max_len):
        self.paths, self.max_len = paths, max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(self.paths[i])
        h = d['hidden_state']
        h = h.view(-1, h.shape[-1])[: self.max_len]
        n = h.shape[0]
        return {"input_ids": d['input_ids'].view(-1)[:n].long(),
                "hidden": h,
                "loss_mask": torch.as_tensor(d['loss_mask']).view(-1)[:n].long()}


def collate(items):
    N = max(it["input_ids"].shape[0] for it in items)
    pad1 = lambda x: F.pad(x, (0, N - x.shape[0]))
    return {
        "input_ids": torch.stack([pad1(it["input_ids"]) for it in items]),
        "loss_mask": torch.stack([pad1(it["loss_mask"]) for it in items]),
        "hidden": torch.stack([F.pad(it["hidden"], (0, 0, 0, N - it["hidden"].shape[0])) for it in items]),
        "lengths": torch.tensor([it["input_ids"].shape[0] for it in items]),
    }


files = [os.path.join(r, f) for r, _, fs in os.walk(args.tmpdir) for f in fs]
train_files, test_files = files[: int(len(files) * 0.95)], files[int(len(files) * 0.95):]
train_loader = DataLoader(RawDataset(train_files, args.max_len), batch_size=args.bs, shuffle=True,
                          collate_fn=collate, num_workers=2, pin_memory=True)
test_loader = DataLoader(RawDataset(test_files, args.max_len), batch_size=args.bs, shuffle=False,
                         collate_fn=collate, num_workers=2, pin_memory=True)


# ----------------------------------------------------------------------------- models
class Router(nn.Module):
    def __init__(self, hidden, K, mid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, mid), nn.GELU(), nn.Linear(mid, K))

    def forward(self, x):
        return self.net(x)


class MultiDrafter(nn.Module):
    """forward() does all rollouts + the router so DDP sees a single wrapped forward."""

    def __init__(self, config, basepath, K, hidden):
        super().__init__()
        ds = [Model(config, load_emb=(k == 0), path=basepath) for k in range(K)]
        for d in ds[1:]:
            d.embed_tokens = ds[0].embed_tokens          # share the frozen embedding
        self.drafters = nn.ModuleList(ds)
        self.router = Router(hidden, K)

    @staticmethod
    def rollout(d, head, h_prefix, ids_prefix, gamma, greedy, tau):
        # prefix pass (teacher features h_0..h_s, tokens x_1..x_{s+1}); last output == step-0 prediction of h_{s+1}
        out, pkv = d(h_prefix, input_ids=ids_prefix, use_cache=True)
        f = out[:, -1:]
        toks, lgs, fts = [], [], []
        for i in range(gamma):
            lg = head(f)                                           # [B,1,V]
            if greedy:
                tok = lg.argmax(-1)
            else:
                tok = torch.multinomial(F.softmax(lg[:, 0].float() / tau, -1), 1)
            fts.append(f); lgs.append(lg); toks.append(tok)
            if i < gamma - 1:
                out, pkv = d(f, input_ids=tok, past_key_values=pkv, use_cache=True)
                f = out[:, -1:]
        return torch.cat(toks, 1), torch.cat(lgs, 1), torch.cat(fts, 1)

    def forward(self, h_prefix, ids_prefix, gamma, head, greedy, tau):
        T, L, Fe = [], [], []
        for d in self.drafters:
            t, l, f = self.rollout(d, head, h_prefix, ids_prefix, gamma, greedy, tau)
            T.append(t); L.append(l); Fe.append(f)
        r = self.router(h_prefix[:, -1].float().detach())
        return torch.stack(T), torch.stack(L), torch.stack(Fe), r


config = EConfig.from_pretrained(args.configpath)
model = MultiDrafter(config, args.basepath, K, baseconfig.hidden_size)

# ----------------------------------------------------------------------------- EM pieces
@torch.no_grad()
def teacher_verify(ids_ctx, toks, s):
    """ids_ctx [B,s+2] = x_0..x_{s+1};  toks [K,B,g].  Returns teacher logits/feats at the draft positions."""
    lgs, fts = [], []
    for k in range(toks.shape[0]):
        ids = torch.cat([ids_ctx, toks[k]], 1)
        hs = teacher.model(input_ids=ids).last_hidden_state[:, s + 1: s + 1 + GAMMA]
        fts.append(hs)
        lgs.append(head(hs.to(head.weight.dtype)).float())
    return torch.stack(lgs), torch.stack(fts)


@torch.no_grad()
def acceptance(t_lg, s_lg, toks):
    logp_t = F.log_softmax(t_lg / TAU, -1)
    logp_s = F.log_softmax(s_lg.float() / TAU, -1)
    if GREEDY:
        acc = toks == t_lg.argmax(-1)
    else:
        lt = logp_t.gather(-1, toks[..., None])[..., 0]
        ls = logp_s.gather(-1, toks[..., None])[..., 0]
        acc = torch.rand_like(lt) < (lt - ls).exp().clamp(max=1.0)
    AL = acc.long().cumprod(-1).sum(-1)                                              # [B]
    exp_AL = torch.minimum(logp_t.exp(), logp_s.exp()).sum(-1).cumprod(-1).sum(-1)   # smooth tie-break
    return AL, exp_AL


def draft_opd_ce(t_lg, s_lg, AL):
    """[B,g,V] x2, AL [B] -> per-position loss [B,g], mask [B,g]"""
    pos = torch.arange(t_lg.shape[1], device=t_lg.device)[None]
    fwd_m, rev_m = pos < AL[:, None], pos == AL[:, None]
    logp_t = F.log_softmax(t_lg / TAU, -1)
    logp_s = F.log_softmax(s_lg.float() / TAU, -1)
    p_t, p_s = logp_t.exp(), logp_s.exp()
    fwd = (p_t * (logp_t - logp_s)).sum(-1)      # KL(T||S): mode covering
    rev = (p_s * (logp_s - logp_t)).sum(-1)      # KL(S||T): mode seeking
    return fwd * fwd_m + rev * rev_m, fwd_m | rev_m


class Balancer:
    """aux-loss-free load balancing: per-drafter bias added to the selection score."""

    def __init__(self, K, lr):
        self.bias, self.lr, self.step = torch.zeros(K, device=dev), lr, 0

    def select(self, score, valid):                       # score [K,B]
        if self.step < args.random_assign_steps:
            return torch.randint(0, score.shape[0], (score.shape[1],), device=score.device)
        return (score + self.bias[:, None]).argmax(0)

    def update(self, winner, valid):
        K_ = self.bias.shape[0]
        usage = torch.bincount(winner[valid], minlength=K_).float() / valid.sum().clamp(min=1)
        self.bias += self.lr * torch.sign(1.0 / K_ - usage)
        self.step += 1
        return usage


def sample_block(batch):
    lengths, lm = batch["lengths"], batch["loss_mask"]
    hi = int(lengths.min().item()) - GAMMA - 2
    if hi < args.min_prefix:
        return None
    best = None
    for _ in range(8):
        s = random.randint(args.min_prefix, hi)
        valid = lm[:, s + 2].bool()                       # first drafted token lies in the response region
        if best is None or valid.sum() > best[1].sum():
            best = (s, valid)
        if valid.all():
            break
    return best if best[1].any() else None


def forward_block(batch, s):
    ids, hid = batch["input_ids"], batch["hidden"]
    h_prefix = hid[:, : s + 1]
    ids_prefix = ids[:, 1: s + 2]
    toks, s_lg, s_ft, r_lg = model(h_prefix, ids_prefix, GAMMA, head, GREEDY, TAU)
    t_lg, t_ft = teacher_verify(ids[:, : s + 2], toks, s)
    AL, expAL = [], []
    for k in range(K):
        a, e = acceptance(t_lg[k], s_lg[k].detach(), toks[k])
        AL.append(a); expAL.append(e)
    return toks, s_lg, s_ft, r_lg, t_lg, t_ft, torch.stack(AL), torch.stack(expAL)


# ----------------------------------------------------------------------------- optim
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95))
total_steps = args.epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, total_steps)
model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, test_loader, scheduler)
balancer = Balancer(K, args.bias_lr)
smooth_l1 = nn.SmoothL1Loss(reduction="none")


@torch.no_grad()
def evaluate():
    model.eval()
    per_k, oracle, routed, rand_, n = torch.zeros(K, device=dev), 0., 0., 0., 0
    for bi, batch in enumerate(test_loader):
        if bi >= args.eval_batches:
            break
        blk = sample_block(batch)
        if blk is None:
            continue
        s, valid = blk
        _, _, _, r_lg, _, _, AL, _ = forward_block(batch, s)
        ALf = AL.float()[:, valid]
        pick = r_lg.argmax(-1)[valid]
        per_k += ALf.mean(1); oracle += ALf.max(0).values.mean()
        routed += ALf[pick, torch.arange(ALf.shape[1], device=dev)].mean()
        rand_ += ALf.mean(0).mean(); n += 1
    model.train()
    n = max(n, 1)
    return {"eval/AL_per_drafter": (per_k / n).tolist(), "eval/AL_oracle": oracle / n,
            "eval/AL_routed": routed / n, "eval/AL_random": rand_ / n}


def save_all(tag):
    if not accelerator.is_main_process:
        return
    m = accelerator.unwrap_model(model)
    out = os.path.join(args.cpdir, tag)
    for k, d in enumerate(m.drafters):
        p = os.path.join(out, f"drafter_{k}")
        os.makedirs(p, exist_ok=True)
        torch.save(d.state_dict(), os.path.join(p, "pytorch_model.bin"))   # loadable by EaModel as before
        shutil.copy(args.configpath, os.path.join(p, "config.json"))
    torch.save({"router": m.router.state_dict(), "bias": balancer.bias.cpu()}, os.path.join(out, "router.pt"))


# ----------------------------------------------------------------------------- train
model.train()
for epoch in range(args.epochs):
    for batch in tqdm(train_loader, disable=not accelerator.is_main_process):
        blk = sample_block(batch)
        if blk is None:
            continue
        s, valid = blk
        B = batch["input_ids"].shape[0]
        ar = torch.arange(B, device=dev)

        with accelerator.accumulate(model):
            toks, s_lg, s_ft, r_lg, t_lg, t_ft, AL, expAL = forward_block(batch, s)

            # ---- E-step
            score = AL.float() + 1e-2 * expAL
            winner = balancer.select(score, valid)                       # [B]

            # ---- M-step (winner only)
            w_AL = AL[winner, ar]
            ce, mask = draft_opd_ce(t_lg[winner, ar], s_lg[winner, ar], w_AL)
            mask = mask & valid[:, None]
            ft = smooth_l1(s_ft[winner, ar].float(), t_ft[winner, ar].float()).mean(-1)
            fmask = (valid[:, None] & torch.ones_like(mask)) if args.feat_mask_all else mask
            loss_ce = (ce * mask).sum() / mask.sum().clamp(min=1)
            loss_ft = (ft * fmask).sum() / fmask.sum().clamp(min=1)

            # ---- router
            if args.router_tau > 0:
                q = F.softmax((score + balancer.bias[:, None]).T / args.router_tau, -1)
                loss_r = -(q * F.log_softmax(r_lg, -1)).sum(-1)[valid].mean()
            else:
                loss_r = F.cross_entropy(r_lg[valid], winner[valid])

            loss = args.v_w * loss_ft + args.p_w * loss_ce + args.router_w * loss_r
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        usage = balancer.update(winner, valid)
        ALf = AL.float()[:, valid]
        log({"train/loss": loss.item(), "train/loss_feat": loss_ft.item(), "train/loss_ce": loss_ce.item(),
             "train/loss_router": loss_r.item(), "train/AL_oracle": ALf.max(0).values.mean().item(),
             "train/AL_winner_mean": ALf.max(0).values.mean().item(),
             "train/router_acc": (r_lg.argmax(-1) == winner)[valid].float().mean().item(),
             **{f"train/AL_k{k}": ALf[k].mean().item() for k in range(K)},
             **{f"train/usage_k{k}": usage[k].item() for k in range(K)}})

    ev = evaluate()
    if accelerator.is_main_process:
        print(f"epoch {epoch}: " + json.dumps({k: (v if isinstance(v, list) else float(v)) for k, v in ev.items()}))
    log(ev)
    save_all(f"epoch_{epoch}")