"""Micro-probes to find why the MLP barely learns:
1) identity: same trunk, y = x  (no diffusion)
2) diffusion w/o conditioning layers (pure MLP)
3) gradient norms per layer group at init
"""
import sys, os, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from pixel_train import PixelDiffusion, AB, T

torch.manual_seed(0)
x = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "pixel_train.npy")).astype(np.float32) / 127.5 - 1
xt = torch.tensor(x.reshape(len(x), -1))[:64]

# 1) plain trunk regression toward x itself
m = PixelDiffusion()
opt = torch.optim.AdamW(m.parameters(), lr=4e-4)
g = torch.Generator().manual_seed(1)
noise = torch.randn(xt.shape, generator=g)
t = torch.full((64,), 250)
ab = AB[t][:, None]
xnoisy = ab.sqrt() * xt + (1 - ab).sqrt() * noise
print("xnoisy std", xnoisy.std().item(), "noise std", noise.std().item())
for s in range(300):
    eps = m(xnoisy, t, torch.zeros(64, dtype=torch.long))
    loss = torch.mean((eps - noise) ** 2)
    opt.zero_grad(); loss.backward()
    if s == 0:
        for nme, p in m.named_parameters():
            if p.grad is not None and ("inp" in nme or "out" in nme or "h1" in nme):
                print(f"  grad {nme}: {p.grad.norm().item():.2e}  wstd {p.std().item():.2e}")
        print("  out act stats: mean", eps.mean().item(), "std", eps.std().item())
    opt.step()
    if s % 100 == 0: print("  fixed-batch diffusion:", loss.item())

# 2) identity probe: does the trunk even learn y=x?
m2 = PixelDiffusion()
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
for s in range(300):
    out = m2(xt, t, torch.zeros(64, dtype=torch.long))
    loss = torch.mean((out - xt) ** 2)
    opt2.zero_grad(); loss.backward(); opt2.step()
    if s % 100 == 0: print("  identity probe:", loss.item(), "target var:", xt.var().item())
