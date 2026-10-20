"""Trainer. Stage-based: 2m -> push -> 5m -> push -> 10m -> push -> 17m -> push.
CPU-optimized: small batches, no AMP, torch threads pinned."""
import os, sys, time, math, json, pickle, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs

# v11 ctx-2048: storing every layer's [B,H,2048,2048] attention for backward
# needs ~1.7GB for a single row — OOM on the 3GB box. Checkpointing each block
# keeps only ONE layer's attention live at a time (+~33% recompute FLOPs).
GRAD_CKPT = os.environ.get("SIGMA_GRAD_CKPT") == "1"

torch.manual_seed(1); random.seed(1); np.random.seed(1)
torch.set_num_threads(2)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")

class FlatLM:
    """v11 memory layout: 290k small numpy arrays ate ~940MB RSS and pushed
    training over the OOM line. One flat token array + offsets = ~250MB."""
    def __init__(self, samples):
        self.lens = np.array([len(s[0]) for s in samples], dtype=np.int64)
        self.flat = np.concatenate([s[0] for s in samples]).astype(np.int32)
        self.mask = np.concatenate([s[1] for s in samples]).astype(np.uint8)
        self.offs = np.concatenate([[0], np.cumsum(self.lens)])
        self.order_len = self.lens
    def __len__(self): return len(self.lens)
    def get(self, i):
        o, n = int(self.offs[i]), int(self.lens[i])
        return self.flat[o:o+n], self.mask[o:o+n]

def load_data():
    lm_path = os.environ.get("SIGMA_LM_DATA", os.path.join(DATA, "lm_train.pkl"))
    ev_path = os.environ.get("SIGMA_LM_EVAL", os.path.join(DATA, "lm_eval.pkl"))
    with open(lm_path, "rb") as f: lm = pickle.load(f)
    flat = FlatLM(lm)
    del lm
    with open(ev_path, "rb") as f: lmev = pickle.load(f)
    lmev = lmev[:256]
    with open(os.path.join(DATA, "intent.pkl"), "rb") as f: it = pickle.load(f)
    return flat, lmev, it[:8000]

BUCKETS = (256, 512, 768, 1024, 1536, 2048)
def pad_batch(samples, pad=0):
    # v11: pad to fixed shape buckets — repeating shapes stop the allocator
    # creep (RSS grew +900MB/25 steps with free-form padding, then OOM-kill)
    mx = max(len(s[0]) for s in samples)
    for b in BUCKETS:
        if mx <= b:
            mx = b
            break
    B = len(samples)
    x = np.full((B, mx), pad, dtype=np.int64)
    y = np.full((B, mx), -100, dtype=np.int64)
    m = np.zeros((B, mx), dtype=np.float32)
    for i, (ids, mask) in enumerate(samples):
        x[i, :len(ids)] = ids
        y[i, :len(ids)] = ids
        m[i, :len(mask)] = mask
    return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m))

def pad_intent(samples, pad=0):
    mx = max(len(s[0]) for s in samples)
    B = len(samples)
    x = np.full((B, mx), pad, dtype=np.int64)
    lab = np.zeros((B,), dtype=np.int64)
    for i, (ids, lab_i) in enumerate(samples):
        x[i, :len(ids)] = ids
        lab[i] = lab_i
    return torch.from_numpy(x), torch.from_numpy(lab)

def train_stage(name, lm, lmev, it, max_steps, max_minutes, lr, model_path):
    import json, string
    with open(os.path.join(DATA, "vocab3.json"), "r", encoding="utf-8") as f:
        vocab_size = len(json.load(f))
    cfg = configs(vocab=vocab_size)[name]
    # v10 context window: positional table 256 -> 512 (set by the launcher env)
    if os.environ.get("SIGMA_BLOCK"):
        cfg.block_size = int(os.environ["SIGMA_BLOCK"])
    model = Transformer(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    class LeanAdamW:
        """manual AdamW: torch's optimizer step transiently allocated ~1.8GB on
        51.5M params and OOM-killed ctx-2048 training. Pre-allocated states +
        in-place per-param math cap the overhead at the states themselves."""
        def __init__(self, params, lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8):
            self.p = [q for q in params if q.requires_grad]
            self.lr, self.b1, self.b2, self.wd, self.eps = lr, *betas, weight_decay, eps
            self.m = [torch.zeros_like(q) for q in self.p]
            self.v = [torch.zeros_like(q) for q in self.p]
            self.t = 0
        def param_groups(self):
            return [{"lr": self.lr}]
        def zero_grad(self, set_to_none=True):
            for q in self.p:
                q.grad = None
        def step(self):
            self.t += 1
            bc1 = 1 - self.b1 ** self.t
            bc2 = 1 - self.b2 ** self.t
            lr_t = self.lr * math.sqrt(bc2) / bc1
            with torch.no_grad():
                for q, mo, vo in zip(self.p, self.m, self.v):
                    g = q.grad
                    if g is None: continue
                    if self.wd: q.mul_(1 - self.lr * self.wd)
                    mo.lerp_(g, 1 - self.b1)
                    vo.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                    q.addcdiv_(mo, vo.sqrt().add_(self.eps), value=-lr_t)
    opt = LeanAdamW(model.parameters(), lr=lr)
    start_step, history = 0, []
    init_from = os.environ.get("SIGMA_INIT_FROM")
    src_path = model_path if os.path.exists(model_path) else init_from
    if src_path and os.path.exists(src_path):
        ck = torch.load(src_path, map_location="cpu", weights_only=False)
        sd = dict(ck["model"])
        # vocab growth/shrink between regens: keep the checkpoint table aligned
        # to the data vocab. Growth appends fresh rows (new letter tokens etc.);
        # shrink truncates tail rows (rare words that dropped out of the cut —
        # the tokenizer already falls back to char-spelling for them).
        old_v = sd["tok_emb.weight"].shape[0]
        if old_v != cfg.vocab_size:
            w = sd["tok_emb.weight"]
            if old_v < cfg.vocab_size:
                g = torch.Generator().manual_seed(77)
                new = torch.empty(cfg.vocab_size, w.shape[1])
                torch.nn.init.normal_(new, mean=0.0, std=0.02, generator=g)
                new[:old_v] = w
            else:
                new = w[:cfg.vocab_size].clone()
            sd["tok_emb.weight"] = new
            sd["lm_head.weight"] = new
            print(f"resized embedding: {old_v} -> {cfg.vocab_size} rows")
        # positional growth (50m ctx 256 -> v10 ctx 512): interpolate the trained
        # rows so mid-sequence positions keep roughly their learned meaning
        old_b = sd["pos_emb.weight"].shape[0]
        if old_b != cfg.block_size:
            w = sd["pos_emb.weight"].T.unsqueeze(0)                     # [1, D, 256]
            w = torch.nn.functional.interpolate(w, size=cfg.block_size, mode="linear", align_corners=True)
            sd["pos_emb.weight"] = w.squeeze(0).T                       # [512, D]
            print(f"interpolated positional table: {old_b} -> {cfg.block_size} rows")
        sd.pop("mask", None)          # rebuilt from cfg (shape differs when ctx grows)
        model.load_state_dict(sd)
        if src_path == model_path:
            start_step = ck.get("step", 0)
            history = ck.get("history", [])
            print(f"resumed from step {start_step}")
        else:
            print(f"warm-started from {os.path.basename(src_path)} (fresh schedule)")
    print(f"== STAGE {name}: {n_par:,} params (d={cfg.d_model} L={cfg.n_layers} ctx={cfg.block_size})", flush=True)
    # v7: batch scales with model size — 3GB RAM box. 50M at bs20 thrashed into
    # swap (activations ~700MB on top of 600MB optimizer states) and steps ballooned.
    if n_par > 35_000_000:
        bs_lm, bs_it = 10, 4
    elif n_par > 15_000_000:
        bs_lm, bs_it = 16, 6
    else:
        bs_lm, bs_it = 20, 8
    if os.environ.get("SIGMA_BS_LM"):
        bs_lm = int(os.environ["SIGMA_BS_LM"])
    print(f"batch: lm {bs_lm} intent {bs_it}", flush=True)
    warm = 100
    def lr_at(s): return lr * min(1.0, (s+1)/warm, 0.1 + 0.45*(1+math.cos(math.pi*min(s,max_steps)/max_steps)))
    t0 = time.time()
    rng = random.Random(5 + start_step)
    os.makedirs(OUT, exist_ok=True)
    if start_step >= max_steps:
        print("ALREADY_DONE")
        return n_par
    last_step = start_step
    # length-bucketed batches: random windows over a length-sorted index
    # (near-uniform batch lengths -> ~2x less padding compute on CPU)
    order = sorted(range(len(lm)), key=lambda i: lm.order_len[i])
    for step in range(start_step, max_steps):
        if (time.time()-t0)/60 > max_minutes:
            print("CHUNK_TIME_REACHED", flush=True); break
        opt.lr = lr_at(step)
        # token-budget batch (v7): fill greedily from a random point of the
        # length-sorted index up to a fixed TOKEN budget — step compute stays
        # bounded no matter where the window lands (a bs10 window of 256-token
        # samples on this 3GB box used to swap-thrash to 100s+/step).
        budget = 1400 if n_par > 35_000_000 else 2600 if n_par > 15_000_000 else 3600
        if os.environ.get("SIGMA_BUDGET"):
            budget = int(os.environ["SIGMA_BUDGET"])
        s0 = rng.randrange(len(order))
        idxs, tot, j = [], 0, s0
        while len(idxs) < bs_lm and j < len(order):
            L = int(lm.order_len[order[j]])
            if idxs and tot + L > budget: break
            idxs.append(order[j]); tot += L; j += 1
        itb = rng.sample(it, bs_it)
        x, y, m = pad_batch([lm.get(i) for i in idxs])
        xi, li = pad_intent(itb)
        def run_seg(xs, ys, ms, pk=None, pv=None):
            """one segment forward+loss, checkpointed blocks, optional cached past"""
            B, Ts = xs.shape
            past_len = 0 if pk is None else pk.shape[3]
            h = model.tok_emb(xs) + model.pos_emb(torch.arange(past_len, past_len + Ts))[None, :, :]
            h = model.drop(h)
            pks, pvs = [], []
            for i, blk in enumerate(model.blocks):
                pki = None if pk is None else pk[i]
                pvi = None if pv is None else pv[i]
                if True:  # checkpoint every block: graph = inputs only
                    h, k, v = checkpoint(lambda e, b=blk, pk_=pki, pv_=pvi: b(e, pk_, pv_), h,
                                         use_reentrant=False)
                else:
                    h, k, v = blk(h, pki, pvi)
                pks.append(k); pvs.append(v)
            h = model.ln_f(h)
            logits = model.lm_head(h)
            return model.loss_from_logits(logits, ys, ms), torch.stack(pks), torch.stack(pvs)

        SEG = int(os.environ.get("SIGMA_SEG", 512))
        def lm_fwd(xx, yy, mm):
            T = xx.shape[1]
            if GRAD_CKPT and T > 1024:
                # v11 SEGMENTED LONG TRAINING: long rows train as ~704-token
                # segments with a detached KV cache carried across boundaries
                # (same approximation as windowed attention). Attention memory
                # per backward graph stays tiny — this is what lets ctx-2048
                # train on a 4GB box.
                tot, wsum, pk, pv = 0.0, 0.0, None, None
                s = 0
                while s < T:
                    e = min(T, s + SEG)
                    xs, ys, ms = xx[:, s:e], yy[:, s:e], mm[:, s:e]
                    if ms.sum() > 0:
                        l, pk2, pv2 = run_seg(xs, ys, ms, pk, pv)
                        w = ms.sum()
                        tot = tot + l * w; wsum = wsum + w
                    if e < T:
                        with torch.no_grad():
                            _, pk, pv = run_seg(xs, ys, ms, pk, pv)
                        pk, pv = pk.detach(), pv.detach()
                    s = e
                return {"lm_loss": tot / wsum.clamp(min=1)}
            if GRAD_CKPT:
                l, _, _ = run_seg(xx, yy, mm)
                return {"lm_loss": l}
            return model(xx, targets=yy, loss_mask=mm)
        if step < start_step + 40:
            import resource as _r
            print(f"  [dbg {step}] T={x.shape[1]} B={x.shape[0]} rss={_r.getrusage(_r.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
        import resource as _r
        out = lm_fwd(x, y, m)
        if step < start_step + 3: print(f"  [fwd {step}] rss={_r.getrusage(_r.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
        oi = model(xi, intent_labels=li)
        loss = out["lm_loss"] + 0.6 * oi["intent_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step < start_step + 3: print(f"  [bwd {step}] rss={_r.getrusage(_r.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step < start_step + 3: print(f"  [opt {step}] rss={_r.getrusage(_r.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
        if step % 25 == 0 or step == max_steps-1:
            import resource
            el = time.time()-t0
            print(f"step {step:5d} lm {out['lm_loss']:.3f} intent {oi['intent_loss']:.3f} "
                  f"ppl {math.exp(min(out['lm_loss'].item(),10)):.2f} lr {lr_at(step):.2e} "
                  f"{el:.0f}s elapsed rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)
        if step % 25 == 0 or step == max_steps-1:
            history.append({"step": step, "lm": out["lm_loss"].item(), "intent": oi["intent_loss"].item()})
        if (step + 1) % 60 == 0:
            import gc; gc.collect()
        save_every = int(os.environ.get("SIGMA_SAVE_EVERY", 400))
        if (step + 1) % save_every == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "params": n_par,
                        "name": name, "step": step + 1, "history": history}, model_path)
        last_step = step + 1
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "params": n_par,
                "name": name, "step": last_step, "history": history}, model_path)
    done = last_step >= max_steps
    print("STAGE_COMPLETE" if done else "CHUNK_SAVED", f"at step {last_step}/{max_steps}", flush=True)
    # quick eval (skip on intermediate chunks: 50M eval burn ~50s per chunk)
    if os.environ.get("SKIP_EVAL") and not done:
        return n_par
    with torch.no_grad():
        lmb = lmev[:8]
        x, y, m = pad_batch(lmb)
        out = model(x, targets=y, loss_mask=m)
        ev_lm = out["lm_loss"].item()
        itb = it[:32]
        xi, li = pad_intent(itb)
        oi = model(xi, intent_labels=None)
        acc = (oi["intent_logits"].argmax(-1) == li).float().mean().item()
    print(f"== DONE {name}: {n_par:,} params | eval lm {ev_lm:.3f} ppl {math.exp(min(ev_lm,10)):.2f} | intent acc {acc:.2%}")
    meta = {"name": name, "params": n_par, "eval_lm": ev_lm, "intent_acc": acc, "history": history}
    with open(model_path.replace(".pt", "_meta.json"), "w") as f: json.dump(meta, f, indent=2)
    return n_par

if __name__ == "__main__":
    stage = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 45
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-4
    lm, lmev, it = load_data()
    n = train_stage(stage, lm, lmev, it, steps, minutes, lr, os.path.join(OUT, f"{stage}.pt"))
    print(json.dumps({"stage": stage, "params": n}))
