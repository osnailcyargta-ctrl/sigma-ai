#!/usr/bin/env python3
"""Profile one v11 training step at T=2048 vs T=256, with/without checkpointing."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs

torch.set_num_threads(2)
ck = torch.load("../out/50m.pt", map_location="cpu", weights_only=False)
cfg = configs(vocab=1589)["v11"]
cfg.block_size = 2048
m = Transformer(cfg)
sd = dict(ck["model"])
sd["tok_emb.weight"] = torch.cat([sd["tok_emb.weight"], torch.zeros(1, 640)], 0)
sd["lm_head.weight"] = sd["tok_emb.weight"]
import torch.nn.functional as F
w = sd["pos_emb.weight"].T.unsqueeze(0)
w = F.interpolate(w, size=2048, mode="linear", align_corners=True)
sd["pos_emb.weight"] = w.squeeze(0).T
m.load_state_dict(sd)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

def step(T, ckpt):
    x = torch.randint(0, 1589, (1, T))
    y = x.clone()
    msk = torch.ones(1, T)
    t0 = time.time()
    if ckpt and T > 640:
        h = m.tok_emb(x) + m.pos_emb(torch.arange(T))[None]
        h = m.drop(h)
        for blk in m.blocks:
            h = torch.utils.checkpoint.checkpoint(lambda e: blk(e)[0], h, use_reentrant=False)
        h = m.ln_f(h)
        logits = m.lm_head(h)
        loss = m.loss_from_logits(logits, y, msk)
    else:
        loss = m(x, targets=y, loss_mask=msk)["lm_loss"]
    t1 = time.time()
    opt.zero_grad(); loss.backward()
    t2 = time.time()
    opt.step()
    t3 = time.time()
    print(f"T={T} ckpt={ckpt}: fwd {t1-t0:.1f}s bwd {t2-t1:.1f}s opt {t3-t2:.1f}s total {t3-t0:.1f}s")

for T, c in [(256, False), (512, False), (1024, True), (2048, True), (2048, True)]:
    step(T, c)
