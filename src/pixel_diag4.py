"""LR sweep for the diffusion MLP: fresh batches, random t, 900 steps.
Winner criterion: loss < 0.45 by step 900 -> full train viable.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from pixel_train import PixelDiffusion, AB, T

x = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "pixel_train.npy")).astype(np.float32) / 127.5 - 1
xt = torch.tensor(x.reshape(len(x), -1))
n = len(xt)
cl = torch.arange(n) % 6

def run(lr, betas, steps=900):
    torch.manual_seed(0)
    m = PixelDiffusion()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=betas, weight_decay=1e-5)
    g = torch.Generator().manual_seed(1)
    rng = np.random.default_rng(5)
    hist = []
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, 96))
        xb, cb = xt[idx], cl[idx]
        t = torch.from_numpy(rng.integers(0, T, len(idx)))
        noise = torch.randn(xb.shape, generator=g)
        ab = AB[t][:, None]
        xnoisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        eps = m(xnoisy, t, cb)
        loss = torch.mean((eps - noise) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
    marks = [0, 200, 400, 600, 899]
    print(f"lr {lr} betas {betas}: " + "  ".join(f"s{hist[i]:.3f}" if False else f"{hist[i]:.3f}" for i in marks), flush=True)

for lr, betas in [(1.5e-3, (0.9, 0.99)), (3e-3, (0.9, 0.99)), (1e-3, (0.9, 0.999))]:
    run(lr, betas)
