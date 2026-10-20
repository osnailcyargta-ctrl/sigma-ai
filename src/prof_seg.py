#!/usr/bin/env python3
"""Reproduce one trainer long-batch step with timing + RSS, outside train.py."""
import os, sys, time, pickle, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from torch.utils.checkpoint import checkpoint

torch.set_num_threads(2)
t0 = time.time()
with open("../data/lm_train.pkl", "rb") as f: lm = pickle.load(f)
print(f"data loaded {time.time()-t0:.1f}s, rss {resource_rss():.0f}MB" if False else f"data loaded {time.time()-t0:.1f}s")

import resource
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

t0 = time.time()
ck = torch.load("../out/v11.pt", map_location="cpu", weights_only=False)
cfg = configs(vocab=1589)["v11"]; cfg.block_size = 2048
m = Transformer(cfg); m.load_state_dict(ck["model"]); m.train()
print(f"model {time.time()-t0:.1f}s rss {rss():.0f}MB")
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

# find a long sample
idx = max(range(len(lm)), key=lambda i: len(lm[i][0]))
ids, mask = lm[idx]
print("longest sample:", len(ids), "tokens, rss", f"{rss():.0f}MB")

x = torch.from_numpy(ids.astype(np.int64))[None]
y = x.clone(); mk = torch.from_numpy(mask.astype(np.float32))[None]

def run_seg(xs, ys, ms, pk=None, pv=None):
    B, Ts = xs.shape
    past_len = 0 if pk is None else pk.shape[3]
    h = m.tok_emb(xs) + m.pos_emb(torch.arange(past_len, past_len + Ts))[None, :, :]
    h = m.drop(h)
    for blk in m.blocks:
        h, k, v = checkpoint(lambda e, b=blk, pk_=pk, pv_=pv: b(e, pk_, pv_), h, use_reentrant=False)
    h = m.ln_f(h)
    logits = m.lm_head(h)
    return m.loss_from_logits(logits, ys, ms)

H = len(ids) - len(ids)//2
for trial in range(2):
    t0 = time.time()
    l1 = run_seg(x[:, :H], y[:, :H], mk[:, :H])
    t1 = time.time()
    with torch.no_grad():
        run_seg(x[:, :H], y[:, :H], mk[:, :H])
    t2 = time.time()
    pk1 = None  # skip cache for pure mem test
    l2 = run_seg(x[:, H:], y[:, H:], mk[:, H:])
    loss = l1 + l2
    t3 = time.time()
    opt.zero_grad(); loss.backward()
    t4 = time.time()
    opt.step()
    print(f"trial {trial}: f1 {t1-t0:.1f}s recompute {t2-t1:.1f}s f2 {t3-t2:.1f}s bwd {t4-t3:.1f}s step {time.time()-t0:.1f}s rss {rss():.0f}MB")
