"""Decisive probe: is the t-conditioning pathway the failure?
D1: full data, random t, correct t given to model
D2: full data, random t, model given WRONG t (permuted) — upper bound if cond broken
D3: full data, fixed t=250, fresh batches — generalization at fixed t
Every 100 steps print loss; also per-t bucket loss for D1 at the end.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from pixel_train import PixelDiffusion, AB, T

x = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "pixel_train.npy")).astype(np.float32) / 127.5 - 1
xt = torch.tensor(x.reshape(len(x), -1))
n = len(xt)
cl = torch.arange(n) % 6

def run(name, steps=800, lr=4e-4, tmode="rand", cond="true"):
    torch.manual_seed(0)
    m = PixelDiffusion()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    rng = np.random.default_rng(5)
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, 96))
        xb, cb = xt[idx], cl[idx]
        if tmode == "rand":
            t = torch.from_numpy(rng.integers(0, T, len(idx)))
        else:
            t = torch.full((len(idx,),), 250) if False else torch.full((len(idx),), 250)
        noise = torch.randn(xb.shape, generator=g)
        ab = AB[t][:, None]
        xnoisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        tc = t
        if cond == "wrong":
            tc = t[torch.randperm(len(t))]
        eps = m(xnoisy, tc, cb)
        loss = torch.mean((eps - noise) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0 or s == steps - 1: print(f"  {name} step {s}: {loss.item():.4f}", flush=True)
    if name == "D1":
        m.eval()
        with torch.no_grad():
            for lo, hi in ((0, 50), (50, 150), (150, 250), (250, 300)):
                t = torch.arange(lo, hi)
                ab = AB[t][:, None]
                xb = xt[:hi-lo]
                noise = torch.randn(xb.shape, generator=g)
                xnoisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
                eps = m(xnoisy, t, cl[:hi-lo])
                print(f"  D1 loss t[{lo},{hi}): {torch.mean((eps-noise)**2).item():.4f}")

run("D1 rand-t true-cond")
run("D2 rand-t WRONG-cond", cond="wrong")
run("D3 fixed-250 fresh-batch", tmode="fixed")
