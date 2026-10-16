"""
eagle/train/main_em_rollout.py -- multi-GPU, batched EM multi-drafter training over full SD rollouts.

Launch (4 GPUs):   torchrun --nproc_per_node=4 -m eagle.train.main_em_rollout --basepath ... --tmpdir ... --cpdir ...
Single GPU also works (plain `python -m ...`).

Parallelism = data parallel by hand (every rank: own teacher copy, own K drafters, own prompt shard):
  * Ranks finish rollouts at different times, so DDP's per-backward all-reduce would hang. Instead all ranks run
    the cycle loop in LOCKSTEP: every iteration does one tiny all-reduce of [alive_rows, per-drafter usage].
    A rank whose sequences are all finished keeps iterating ("ghost") and still joins every collective.
  * Every --cycles_per_update iterations (and when the batch finishes) grads are averaged over the ranks that
    contributed, drafters that got no gradient anywhere are skipped (grad=None => AdamW leaves them untouched),
    then one optimizer step.  Optimizer state is sharded with ZeRO-1 (ZeroRedundancyOptimizer).
  * The balancer sees the GLOBAL usage so its bias/EMA stay identical on all ranks.

Per rollout cycle (per sequence): K drafters draft a chain -> teacher verifies -> winner = argmax AL (+bias) ->
winner trains with v_w*SmoothL1(feat) + p_w*DraftOPD-CE (fwd KL accepted / rev KL first rejected), router learns the
winner -> sequence advances by AL_adv+1 tokens.  Ragged batching uses "masked slots": shared write pointer into the
KV caches, fixed (gamma+1)-wide blocks per cycle, per-sequence validity masks and true position_ids.

Assumes cnets1.Model.forward(hidden, input_ids, attention_mask, position_ids, past_key_values, use_cache) -> (hidden, kv),
HF Llama (sdpa/eager) honouring 2D masks with holes + position_ids, DynamicCache.crop, chain drafting (no tree).
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from ..model.configs import EConfig
from ..model.cnets1 import Model
from transformers import AutoModelForCausalLM, AutoConfig, DynamicCache, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.distributed as dist
import argparse
import json
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--basepath', required=True)
parser.add_argument('--eapath', required=True)
parser.add_argument('--configpath', default="config.json")
parser.add_argument('--tmpdir', required=True)
parser.add_argument('--cpdir', required=True)
parser.add_argument('--bs', type=int, default=8, help="prompts per rollout batch PER GPU")
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
parser.add_argument('--random_assign_cycles', type=int, default=2000, help="in lockstep iterations")
parser.add_argument('--bias_lr', type=float, default=1e-3)
parser.add_argument('--p_w', type=float, default=0.1)
parser.add_argument('--v_w', type=float, default=1.0)
parser.add_argument('--router_w', type=float, default=1.0)
parser.add_argument('--grad_clip', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--no_zero', action='store_true', help="disable ZeRO-1 optimizer sharding")
parser.add_argument('--eval_prompts', type=int, default=32, help="total across ranks")
parser.add_argument('--eval_new', type=int, default=128)
parser.add_argument('--eval_every_batches', type=int, default=100)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--init_ckpt', default=None, help="pretrained EAGLE drafter (dir or .bin/.safetensors) loaded into every drafter")
parser.add_argument('--max_prompts', type=int, default=0, help="use only the first N training prompts (0 = all)")
parser.add_argument('--eval_only', action='store_true', help="run evaluate() once and exit")
parser.add_argument('--check_teacher', action='store_true',
                    help="sanity test: greedy rollout tokens must equal teacher.generate (use --temperature 0)")
args = parser.parse_args()


# ------------------------------------------------------------------ distributed setup
WORLD = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(LOCAL)
if WORLD > 1:
    dist.init_process_group("nccl")
dev = torch.device("cuda", LOCAL)
IS_MAIN = RANK == 0
torch.backends.cuda.matmul.allow_tf32 = True

K, GAMMA = args.num_drafters, args.gamma
GREEDY = args.temperature <= 0
TAU = 1.0 if GREEDY else args.temperature
def AC(): return torch.autocast("cuda", dtype=torch.bfloat16)


print(f"DEBUG : WORLD size is {WORLD}")
print(f"DEBUG : RANK is {RANK}")
print(f"DEBUG : LOCAL is {LOCAL}")


def allreduce(t):
    if WORLD > 1:
        dist.all_reduce(t)
    return t


if args.wandb and IS_MAIN:
    import wandb
    wandb.init(project="em-multidrafter-rollout", config=vars(args))


def log(d): return wandb.log(d) if (args.wandb and IS_MAIN) else None


# ------------------------------------------------------------------ live teacher (also provides the frozen LM head)
torch.manual_seed(0)
random.seed(0)                     # identical drafter init on every rank
baseconfig = AutoConfig.from_pretrained(args.basepath)
teacher = AutoModelForCausalLM.from_pretrained(args.basepath, torch_dtype=torch.bfloat16).to(dev).eval()
teacher.requires_grad_(False)
head = teacher.lm_head                                   # shared, bf16, frozen (saves a 2GB copy)
_eos = teacher.generation_config.eos_token_id
EOS = set(_eos if isinstance(_eos, list) else [_eos])
eos_t = torch.tensor(sorted(EOS), device=dev)
CAN_COMPACT = hasattr(DynamicCache, "batch_select_indices")


def t_head(h):                                           # teacher logits from teacher features
    return head(h.to(head.weight.dtype)).float()


def sample_logits(lg):                                   # [N,V] float -> [N]
    return lg.argmax(-1) if GREEDY else torch.multinomial(F.softmax(lg / TAU, -1), 1)[:, 0]


# ------------------------------------------------------------------ drafters + router
class Router(nn.Module):
    def __init__(self, H, K, mid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H, mid), nn.GELU(), nn.Linear(mid, K))

    def forward(self, x):
        return self.net(x)


class MultiDrafter(nn.Module):
    def __init__(self, config, basepath, K, H):
        super().__init__()
        ds = []
        for k in range(K):                                # built one at a time: peak CPU RAM = 2 embeddings
            d = Model(config, load_emb=(k == 0), path=basepath)
            d.gradient_checkpointing = False              # cnets1 checkpoints in train mode and drops the KV cache
            if k == 0:
                d.embed_tokens.to(torch.bfloat16)         # frozen, bf16 saves 1GB
            else:
                d.embed_tokens = ds[0].embed_tokens       # shared frozen embedding
            ds.append(d)
        self.drafters = nn.ModuleList(ds)
        self.router = Router(H, K)


def ones_l(B, n):
    return torch.ones(B, n, dtype=torch.long, device=dev)


def draft_chain(d, h_s, root, pkv, dvalid, pos_s):
    """h_s [B,1,H], root [B,1], pkv committed drafter cache (D slots), dvalid [B,D], pos_s [B].
    Returns tokens [B,g], logits [B,g,V], feats [B,g,H]; feats[:, i] ~ teacher feature at position root+i."""
    B = root.shape[0]
    dv = dvalid.long()
    out, p = d(h_s, input_ids=root, attention_mask=torch.cat([dv, ones_l(B, 1)], 1),
               position_ids=pos_s[:, None], past_key_values=pkv, use_cache=True)
    f = out[:, -1:]
    toks, lgs, fts = [], [], []
    for i in range(GAMMA):
        lg = head(f)                                                              # [B,1,V]
        tok = lg.argmax(-1) if GREEDY else torch.multinomial(F.softmax(lg[:, 0].float() / TAU, -1), 1)
        fts.append(f)
        lgs.append(lg)
        toks.append(tok)
        if i < GAMMA - 1:
            out, p = d(f, input_ids=tok, attention_mask=torch.cat([dv, ones_l(B, i + 2)], 1),
                       position_ids=(pos_s + i + 1)[:, None], past_key_values=p, use_cache=True)
            f = out[:, -1:]
    return torch.cat(toks, 1), torch.cat(lgs, 1), torch.cat(fts, 1)


model = MultiDrafter(EConfig.from_pretrained(args.configpath), args.basepath, K, baseconfig.hidden_size).to(dev)
ea_state_dict = torch.load(os.path.join(args.eapath, "pytorch_model.bin"), map_location=dev, weights_only=True)
for k, d in enumerate(model.drafters):
    d.load_state_dict(ea_state_dict, strict=False)  # Skip incompatible keys like 'fc'
    print(f"DEBUG : loaded state dict for draft {k}")

if WORLD > 1:                                            # make sure all ranks start from rank 0's weights
    for p in model.parameters():
        dist.broadcast(p.data, 0)

groups = [[p for p in d.parameters() if p.requires_grad] for d in model.drafters] + [list(model.router.parameters())]
all_params = [p for g in groups for p in g]
if WORLD > 1 and not args.no_zero:
    from torch.distributed.optim import ZeroRedundancyOptimizer
    optimizer = ZeroRedundancyOptimizer(all_params, optimizer_class=optim.AdamW, lr=args.lr, betas=(0.9, 0.95))
else:
    optimizer = optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.95))
scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.total_updates)
smooth_l1 = nn.SmoothL1Loss(reduction="none")
pending = False                                          # did this rank accumulate gradients since the last step?

torch.manual_seed(1000 + RANK)
random.seed(1000 + RANK)  # different rollouts per rank


def sync_and_step():
    """Called at the same lockstep iteration on every rank."""
    global pending
    flags = torch.zeros(len(groups) + 1, device=dev)
    if pending:
        for gi, g in enumerate(groups):
            flags[gi] = float(any(p.grad is not None for p in g))
        flags[-1] = 1.0
    allreduce(flags)
    n_contrib = flags[-1].item()
    if n_contrib == 0:
        return
    for gi, g in enumerate(groups):
        if flags[gi] == 0:                               # nobody used this drafter => leave it untouched
            for p in g:
                p.grad = None
            continue
        for p in g:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            allreduce(p.grad)
            p.grad /= n_contrib
    nn.utils.clip_grad_value_([p for p in all_params if p.grad is not None], args.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    pending = False


# ------------------------------------------------------------------ EM pieces
@torch.no_grad()
def acceptance(t_lg, s_lg, toks):
    """t_lg,s_lg [N,g,V]; toks [N,g] -> AL [N], expected AL [N]"""
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
    """[n,g,V] x2, AL [n]: forward KL on accepted positions, reverse KL on the first rejected one."""
    pos = torch.arange(t_lg.shape[1], device=t_lg.device)[None]
    fwd_m, rev_m = pos < AL[:, None], pos == AL[:, None]
    logp_t = F.log_softmax(t_lg / TAU, -1)
    logp_s = F.log_softmax(s_lg.float() / TAU, -1)
    fwd = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    rev = (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    return fwd * fwd_m + rev * rev_m, fwd_m | rev_m


class Balancer:
    """aux-loss-free balancing: bias on the selection score, driven by an EMA of GLOBAL usage (same on all ranks)."""

    def __init__(self, K, lr, decay=0.99):
        self.bias = torch.zeros(K, device=dev)
        self.ema = torch.full((K,), 1.0 / K, device=dev)
        self.lr, self.decay, self.step = lr, decay, 0

    def select(self, score):                                  # score [K,B] -> [B]
        if self.step < args.random_assign_cycles:
            return torch.randint(0, K, (score.shape[1],), device=dev)
        return (score + self.bias[:, None]).argmax(0)

    def update(self, counts):                                 # global per-drafter win counts [K]
        total = counts.sum()
        if total <= 0:
            return
        self.ema = self.decay * self.ema + (1 - self.decay) * counts / total
        self.bias += self.lr * torch.sign(1.0 / K - self.ema)
        self.step += 1


balancer = Balancer(K, args.bias_lr)


# ------------------------------------------------------------------ one batched rollout
def run_rollout(batch, train, mode, max_new):
    """batch = (ids, pm) left-padded prompts [B,Lp] or None.  In train mode every rank must call this the same
    number of times (lockstep collectives).  mode = which drafter advances each sequence:
    train: 'winner'|'router'|'random';  eval: 'oracle'|'router'|'k{i}'"""
    global pending
    SYNC = train and WORLD > 1
    print(f"DEBUG : SYNC tag is set to {SYNC}")

    arg = torch.arange(GAMMA + 1, device=dev)
    tok_sum, cyc_sum = 0, 0
    sumAL = torch.zeros(K, device=dev)

    if batch is None:                                    # nothing to roll out on this rank: ghost for the whole batch
        B, alive = 0, torch.zeros(0, dtype=torch.bool, device=dev)
        print(f"DEBUG : batch was None")
    else:
        ids, pm = batch[0].to(dev), batch[1].to(dev)    # input_ids, and position_mask
        print(f"DEBUG : input_ids shape ${ids.shape}, position masks shape ${pm.shape}")
        B, Lp = ids.shape                               # batch_size, prompt length
        with torch.no_grad():                            # ---- prefill (teacher + drafters)
            pos = (pm.cumsum(1) - 1).clamp(min=0)
            out = teacher.model(input_ids=ids, attention_mask=pm, position_ids=pos,
                                past_key_values=DynamicCache(), use_cache=True)
            cache, feats = out.past_key_values, out.last_hidden_state             # feats [B,Lp,H]
            tvalid = pm.bool()                                                    # teacher slot validity
            h_s = feats[:, -1:]                                                   # left pad => last slot = last prompt token
            root = sample_logits(t_head(feats[:, -1]))[:, None]                   # [B,1] first generated token
            pos_s = pm.sum(1) - 1                                                 # true position of token owning h_s
            dvalid = pm[:, :Lp - 1].bool()                                        # pair (h_t,x_{t+1}) valid iff token t valid
            pkv = []
            with AC():
                for drafter in model.drafters:
                    pkv.append(drafter(feats[:, :Lp - 1], input_ids=ids[:, 1:Lp], attention_mask=pm[:, :Lp - 1],
                                       position_ids=pos[:, :Lp - 1], use_cache=True)[1] if Lp > 1 else None)
        n_new = torch.ones(B, dtype=torch.long, device=dev)
        alive = ~torch.isin(root[:, 0], eos_t)      # check if sequence is completed yet
        if IS_MAIN:
            print(f"DEBUG rollout: B={B}, root[0]={root[0].item()}, eos_t={eos_t.tolist()}, alive={alive.tolist()}")

    g_alive = int(allreduce(alive.sum().float().view(1)).item()) if SYNC else int(alive.sum())  # how many sequences are alive?
    if IS_MAIN:
        print(f"DEBUG rollout: g_alive after sync={g_alive}, alive.sum()={int(alive.sum())}")

    for it in range(max_new):
        if g_alive == 0:        # break if all sequences reached eos
            if IS_MAIN:
                print(f"DEBUG rollout: breaking immediately, g_alive=0")
            break
        usage = torch.zeros(K, device=dev)
        if int(alive.sum()) > 0:
            # ---------------- drop finished sequences
            if CAN_COMPACT and not alive.all():     # if there are some finished sequences
                keep = alive.nonzero().squeeze(1)
                root, h_s, pos_s, tvalid, dvalid, n_new = (x[keep] for x in (root, h_s, pos_s, tvalid, dvalid, n_new))
                pkv = [None if p is None else tuple((a_[keep], b_[keep]) for a_, b_ in p) for p in pkv]
                cache.batch_select_indices(keep)
                alive = alive[keep]
                B = keep.numel()
            ar = torch.arange(B, device=dev)

            # ---------------- draft (grad iff train)
            with torch.set_grad_enabled(train), AC():
                outs = [draft_chain(d, h_s, root, pkv[k], dvalid, pos_s) for k, d in enumerate(model.drafters)]
                r_lg = model.router(h_s[:, 0].float())                            # [B,K]
            toks = torch.stack([o[0] for o in outs])                              # [K,B,g]
            s_lg_d = torch.stack([o[1].detach() for o in outs])                   # [K,B,g,V] draft, batch, gamma, vocab

            # ---------------- teacher verification: one cached forward per drafter, cache cropped back
            with torch.no_grad():
                base = cache.get_seq_length()
                pos_r = (pos_s + 1)[:, None] + arg[None]                          # positions of [root, d_1..d_g]
                am = torch.cat([tvalid.long(), ones_l(B, GAMMA + 1)], 1)
                t_ft = []
                for k in range(K):
                    o = teacher.model(input_ids=torch.cat([root, toks[k]], 1), attention_mask=am, position_ids=pos_r,
                                      past_key_values=cache, use_cache=True).last_hidden_state
                    cache.crop(base)
                    t_ft.append(o)
                t_ft = torch.stack(t_ft)                                          # [K,B,g+1,H]
                t_lg = t_head(t_ft)                                               # [K,B,g+1,V]
                AL, expAL = acceptance(t_lg[:, :, :GAMMA].reshape(K * B, GAMMA, -1),
                                       s_lg_d.reshape(K * B, GAMMA, -1), toks.reshape(K * B, GAMMA))
                AL, expAL = AL.view(K, B), expAL.view(K, B)
                w = balancer.select(AL.float() + 1e-2 * expAL)                    # E-step, [B]

            # ---------------- M-step: each drafter trains only on the rows it won
            if train:
                AL_w = AL[w, ar]
                pos_g = torch.arange(GAMMA, device=dev)[None]
                m_all = (pos_g <= AL_w[:, None]) & alive[:, None]                 # accepted + first rejected
                cnt = m_all.sum()
                ce_tot, ft_tot = 0.0, 0.0
                for k in range(K):
                    sel = ((w == k) & alive).nonzero().squeeze(1)
                    if sel.numel() == 0:
                        continue
                    ce, mk = draft_opd_ce(t_lg[k][sel, :GAMMA], outs[k][1][sel], AL[k][sel])
                    ft = smooth_l1(outs[k][2][sel].float(), t_ft[k][sel, :GAMMA].float()).mean(-1)
                    ce_tot = ce_tot + ce.sum()
                    ft_tot = ft_tot + (ft * mk).sum()
                if cnt > 0:
                    loss_ce, loss_ft = ce_tot / cnt, ft_tot / cnt
                    loss_r = F.cross_entropy(r_lg[alive], w[alive])
                    loss = (args.v_w * loss_ft + args.p_w * loss_ce + args.router_w * loss_r) / args.cycles_per_update
                    loss.backward()                                               # graph freed every cycle
                    pending = True
                    usage = torch.bincount(w[alive], minlength=K).float()
                    log({"train/loss_feat": float(loss_ft), "train/loss_ce": float(loss_ce),
                         "train/loss_router": loss_r.item(), "train/alive": int(alive.sum()),
                         "train/router_acc": (r_lg.argmax(-1) == w)[alive].float().mean().item(),
                         **{f"train/usage_ema_k{k}": balancer.ema[k].item() for k in range(K)}})

            # ---------------- which drafter advances each sequence?
            with torch.no_grad():
                if mode == 'winner':
                    a = w
                elif mode == 'router':
                    a = r_lg.argmax(-1) if (not train or balancer.step >= args.random_assign_cycles) else w
                elif mode == 'random':
                    a = torch.randint(0, K, (B,), device=dev)
                elif mode == 'oracle':
                    a = AL.argmax(0)
                else:
                    a = torch.full((B,), int(mode[1:]), device=dev, dtype=torch.long)
                AL_a = AL[a, ar]
                chain_a = toks[a, ar]                                             # [B,g]

                # ---------------- bonus token from the teacher (residual sampling when sampling)
                row = t_lg[a, ar, AL_a]                                           # [B,V]
                if GREEDY:
                    bonus = row.argmax(-1)
                else:
                    p_t = F.softmax(row / TAU, -1)
                    p_s = F.softmax(s_lg_d[a, ar, AL_a.clamp(max=GAMMA - 1)].float() / TAU, -1)
                    res = torch.where((AL_a < GAMMA)[:, None], (p_t - p_s).clamp(min=0), p_t)
                    res = torch.where(res.sum(-1, keepdim=True) > 0, res, p_t)
                    bonus = torch.multinomial(res / res.sum(-1, keepdim=True), 1)[:, 0]

                # ---------------- commit a fixed-width (g+1) block; slots beyond AL are marked invalid
                block_tok = torch.cat([root, chain_a], 1)                         # root, d_1..d_g
                blk_valid = arg[None] <= AL_a[:, None]
                teacher.model(input_ids=block_tok, attention_mask=am, position_ids=pos_r,
                              past_key_values=cache, use_cache=True)
                tvalid = torch.cat([tvalid, blk_valid], 1)
                new_feat = torch.cat([h_s, t_ft[a, ar, :GAMMA]], 1)               # pairs (h_s,root),(h(root),d_1),...
                dpos = pos_s[:, None] + arg[None]
                dam = torch.cat([dvalid.long(), ones_l(B, GAMMA + 1)], 1)
                with AC():                                                        # kv cache of all drafters updated?
                    for k, d in enumerate(model.drafters):
                        pkv[k] = d(new_feat, input_ids=block_tok, attention_mask=dam, position_ids=dpos,
                                   past_key_values=pkv[k], use_cache=True)[1]
                dvalid = torch.cat([dvalid, blk_valid], 1)

                # ---------------- advance state
                h_s = t_ft[a, ar, AL_a][:, None]
                pos_s = pos_s + 1 + AL_a
                root = bonus[:, None]
                n_new = n_new + AL_a + 1
                tok_sum += int((AL_a + 1)[alive].sum())
                cyc_sum += int(alive.sum())
                sumAL += AL.float()[:, alive].sum(1)
                accepted = torch.arange(GAMMA, device=dev)[None] < AL_a[:, None]
                eos_hit = (torch.isin(chain_a, eos_t) & accepted).any(1) | torch.isin(bonus, eos_t)
                alive = alive & ~eos_hit & (n_new < max_new)

        # ---------------- lockstep sync: [alive rows after this iteration, per-drafter win counts]
        s = torch.cat([alive.sum().float().view(1), usage])
        if SYNC:
            allreduce(s)
        g_alive = int(s[0].item())
        if train:
            balancer.update(s[1:])
            if (it + 1) % args.cycles_per_update == 0 or g_alive == 0:
                sync_and_step()

    return {"tokens": tok_sum, "cycles": max(cyc_sum, 1), "AL": (sumAL / max(cyc_sum, 1)).tolist()}


# ------------------------------------------------------------------ data: per-rank shards, left-padded prompt batches
class PromptDataset(Dataset):
    def __init__(self, entries):
        """
        entries: list of (path, is_consolidated, cache, local_idx) tuples
                 for single files: (path, False, None, 0)
                 for consolidated: (path, True, cache_or_None, specific_idx)
        """
        self.entries = entries
        print(f"DEBUG PromptDataset: {len(entries)} entries total")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, is_consolidated, cache, local_idx = self.entries[i]

        if is_consolidated:
            if cache is None:
                cache = torch.load(path, weights_only=True)
            input_ids = cache['input_ids'][local_idx]
            loss_mask = cache['loss_masks'][local_idx]
        else:
            d = torch.load(path, weights_only=True)
            input_ids = d['input_ids']
            loss_mask = d.get('loss_mask', d.get('loss_masks'))

        input_ids = torch.as_tensor(input_ids).view(-1)
        loss_mask = torch.as_tensor(loss_mask).view(-1)

        # Get the length of the prompt (all tokens are valid in this format)
        n = len(input_ids)

        # Verify we have valid tokens and proper length
        if n < 2 or n > args.max_prompt:
            return None

        return input_ids.long()


def collate(items):
    items = [x for x in items if x is not None]
    if not items:
        return None
    Lp = max(len(x) for x in items)
    ids = torch.zeros(len(items), Lp, dtype=torch.long)
    pm = torch.zeros(len(items), Lp, dtype=torch.long)
    for i, x in enumerate(items):
        ids[i, Lp - len(x):] = x
        pm[i, Lp - len(x):] = 1
    return ids, pm


def build_dataset_entries(tmpdir, max_prompts=0):
    """
    Scan tmpdir for .pt files and build dataset entries.
    Returns list of (path, is_consolidated, cache) tuples.
    """
    all_files = sorted(os.path.join(r, f) for r, _, fs in os.walk(tmpdir) for f in fs if f.endswith('.pt'))

    entries = []
    for fpath in all_files:
        if 'consolidated' in fpath:
            # Load consolidated file once and index into it
            print(f"DEBUG: Loading consolidated file: {fpath}")
            cache = torch.load(fpath, weights_only=True)
            print(f"DEBUG: Cache keys: {list(cache.keys())}")
            print(f"DEBUG: 'input_ids' in cache: {'input_ids' in cache}")
            if 'input_ids' in cache:
                print(f"DEBUG: type(cache['input_ids']): {type(cache['input_ids'])}")
                print(f"DEBUG: isinstance list: {isinstance(cache['input_ids'], list)}")
            if 'input_ids' in cache and isinstance(cache['input_ids'], list):
                # This is a consolidated file with list of prompts
                print(f"DEBUG: Detected as CONSOLIDATED with {len(cache['input_ids'])} prompts")
                entries.append((fpath, True, cache))
            else:
                # Might be a regular file mistakenly named consolidated
                print(f"DEBUG: Detected as REGULAR file")
                entries.append((fpath, False, None))
        else:
            entries.append((fpath, False, None))

    if max_prompts > 0:
        # For consolidated files, we need to account for multiple prompts per file
        # Simple approach: limit total entries (each consolidated = 1 entry with many sub-indices)
        # This is approximate; a proper implementation would count total prompts
        pass

    return entries


def count_total_prompts(entries):
    """Count total number of prompts across all entries."""
    total = 0
    for path, is_consolidated, cache in entries:
        if is_consolidated and cache:
            total += len(cache['input_ids'])
        else:
            total += 1
    return total


# Build dataset entries and split into train/eval by PROMPT COUNT (not file count)
all_entries = build_dataset_entries(args.tmpdir, args.max_prompts)
total_prompts = count_total_prompts(all_entries)
print(f"DEBUG: total_prompts={total_prompts}, len(all_entries)={len(all_entries)}")

# Expand entries into individual prompt references
# Each prompt gets a unique (entry_idx, local_idx) tuple
# For non-consolidated: local_idx = 0
# For consolidated: local_idx = 0, 1, 2, ... n_prompts-1
all_prompts = []  # list of (entry_idx, local_idx)
for entry_idx, (path, is_consolidated, cache) in enumerate(all_entries):
    if is_consolidated and cache:
        n_prompts = len(cache['input_ids'])
        for local_idx in range(n_prompts):
            all_prompts.append((entry_idx, local_idx))
    else:
        all_prompts.append((entry_idx, 0))

print(f"DEBUG: expanded to {len(all_prompts)} individual prompt references")

# Split by prompt count
train_cutoff = int(len(all_prompts) * 0.95)
train_prompts = all_prompts[:train_cutoff]
eval_prompts = all_prompts[train_cutoff:]
print(f"DEBUG: train_prompts={len(train_prompts)}, eval_prompts={len(eval_prompts)}")

# Build lookup tables: entry_idx -> (path, is_consolidated, cache)
entry_lookup = {i: (p, c, cache) for i, (p, c, cache) in enumerate(all_entries)}

# Handle sharding across ranks
if WORLD > 1:
    n_train = len(train_prompts)
    n_per_rank = n_train // WORLD
    remainder = n_train % WORLD
    if RANK < remainder:
        rank_start = RANK * (n_per_rank + 1)
        rank_end = rank_start + n_per_rank + 1
    else:
        rank_start = remainder * (n_per_rank + 1) + (RANK - remainder) * n_per_rank
        rank_end = rank_start + n_per_rank
    train_prompts = train_prompts[rank_start:rank_end]
    eval_prompts = eval_prompts[RANK::WORLD]

# Create dataset entries from prompt references


def build_prompt_dataset(prompt_refs, entry_lookup, need_cache=True):
    """Build list of (path, is_consolidated, cache, local_idx) from prompt references."""
    entries = []
    for entry_idx, local_idx in prompt_refs:
        path, is_consolidated, cache = entry_lookup[entry_idx]
        if need_cache and is_consolidated and cache is None:
            cache = torch.load(path, weights_only=True)
        entries.append((path, is_consolidated, cache, local_idx))
    return entries


train_entries = build_prompt_dataset(train_prompts, entry_lookup, need_cache=True)
eval_entries = build_prompt_dataset(eval_prompts, entry_lookup, need_cache=False)

train_dataset = PromptDataset(train_entries)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,
                          collate_fn=collate, num_workers=2)

_eval_dataset = PromptDataset(eval_entries)
_eval = [p for p in DataLoader(_eval_dataset, batch_size=None, num_workers=2, shuffle=False) if p is not None]
_eval = _eval[: max(args.eval_prompts // max(1, WORLD), 1)]
eval_batches = [collate(_eval[i:i + args.bs]) for i in range(0, len(_eval), args.bs)]

if IS_MAIN:
    print(f"Dataset: {len(train_dataset)} train samples, {len(_eval_dataset)} eval samples ({len(eval_batches)} batches)")


@torch.no_grad()
def evaluate():
    res = {}
    for mode in ['router', 'oracle'] + [f'k{k}' for k in range(K)]:
        t = torch.zeros(2, device=dev)
        for b in eval_batches:
            st = run_rollout(b, False, mode, args.eval_new)
            t += torch.tensor([st["tokens"], st["cycles"]], device=dev, dtype=torch.float)
        allreduce(t)
        res[f"eval/tokens_per_cycle_{mode}"] = (t[0] / t[1].clamp(min=1)).item()
    return res


def save_all(tag):
    if IS_MAIN:
        out = os.path.join(args.cpdir, tag)
        for k, d in enumerate(model.drafters):
            p = os.path.join(out, f"drafter_{k}")
            os.makedirs(p, exist_ok=True)
            torch.save(d.state_dict(), os.path.join(p, "pytorch_model.bin"))      # loadable by EaModel as before
            shutil.copy(args.configpath, os.path.join(p, "config.json"))
        torch.save({"router": model.router.state_dict(), "bias": balancer.bias.cpu()}, os.path.join(out, "router.pt"))
    if WORLD > 1:
        dist.barrier()


# ------------------------------------------------------------------ train
n_batches = 0
model.train()
for epoch in range(args.epochs):
    for batch in tqdm(train_loader, disable=not IS_MAIN):
        st = run_rollout(batch, True, args.advance, args.max_new)                  # batch may be None => ghost
        n_batches += 1
        log({"train/tokens_per_cycle": st["tokens"] / st["cycles"], **{f"train/AL_k{k}": st["AL"][k] for k in range(K)}})
        if n_batches % args.eval_every_batches == 0:
            model.eval()
            ev = evaluate()
            model.train()
            if IS_MAIN:
                print(json.dumps(ev))
            log(ev)
            save_all(f"batch_{n_batches}")
save_all("final")
if WORLD > 1:
    dist.destroy_process_group()
