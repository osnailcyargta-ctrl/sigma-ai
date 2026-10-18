"""Trainer. Stage-based: 2m -> push -> 5m -> push -> 10m -> push -> 17m -> push.
CPU-optimized: small batches, no AMP, torch threads pinned."""
import os, sys, time, math, json, pickle, random
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs

torch.manual_seed(1); random.seed(1); np.random.seed(1)
torch.set_num_threads(2)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")

def load_data():
    with open(os.path.join(DATA, "lm_train.pkl"), "rb") as f: lm = pickle.load(f)
    with open(os.path.join(DATA, "lm_eval.pkl"), "rb") as f: lmev = pickle.load(f)
    with open(os.path.join(DATA, "intent.pkl"), "rb") as f: it = pickle.load(f)
    return lm, lmev, it

def pad_batch(samples, pad=0):
    mx = max(len(s[0]) for s in samples)
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
    model = Transformer(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    start_step, history = 0, []
    if os.path.exists(model_path):
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
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
        model.load_state_dict(sd)
        start_step = ck.get("step", 0)
        history = ck.get("history", [])
        print(f"resumed from step {start_step}")
    print(f"== STAGE {name}: {n_par:,} params (d={cfg.d_model} L={cfg.n_layers} ctx={cfg.block_size})", flush=True)
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
    order = sorted(range(len(lm)), key=lambda i: len(lm[i][0]))
    for step in range(start_step, max_steps):
        if (time.time()-t0)/60 > max_minutes:
            print("CHUNK_TIME_REACHED"); break
        for g in opt.param_groups: g["lr"] = lr_at(step)
        # batch: ~7/8 LM, ~1/8 intent
        bs_lm, bs_it = 20, 8
        s0 = rng.randrange(max(1, len(order) - bs_lm))
        lmb = [lm[i] for i in order[s0:s0+bs_lm]]
        itb = rng.sample(it, bs_it)
        x, y, m = pad_batch(lmb)
        xi, li = pad_intent(itb)
        out = model(x, targets=y, loss_mask=m)
        oi = model(xi, intent_labels=li)
        loss = out["lm_loss"] + 0.6 * oi["intent_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == max_steps-1:
            el = time.time()-t0
            print(f"step {step:5d} lm {out['lm_loss']:.3f} intent {oi['intent_loss']:.3f} "
                  f"ppl {math.exp(min(out['lm_loss'].item(),10)):.2f} lr {lr_at(step):.2e} "
                  f"{el:.0f}s elapsed", flush=True)
            history.append({"step": step, "lm": out["lm_loss"].item(), "intent": oi["intent_loss"].item()})
        if (step + 1) % 400 == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "params": n_par,
                        "name": name, "step": step + 1, "history": history}, model_path)
        last_step = step + 1
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "params": n_par,
                "name": name, "step": last_step, "history": history}, model_path)
    done = last_step >= max_steps
    print("STAGE_COMPLETE" if done else "CHUNK_SAVED", f"at step {last_step}/{max_steps}")
    # quick eval
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
