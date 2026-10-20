"""Debug: can the pixel MLP overfit at all? Three probes:
A) overfit 64 images, fixed t — should hit ~0 fast if pipeline is sane
B) overfit 64 images, all t — should drop well below 0.3
C) full data, fixed high t — copy-input baseline check
"""
import sys, os, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pixel_data
from pixel_train import PixelDiffusion, time_embed, AB, T

torch.manual_seed(0)
x = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "pixel_train.npy")).astype(np.float32) / 127.5 - 1
xt = torch.tensor(x.reshape(len(x), -1))

def probe(name, data, tmode, steps=600, lr=4e-4):
    torch.manual_seed(0)
    m = PixelDiffusion()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    n = len(data)
    g = torch.Generator().manual_seed(1)
    for s in range(steps):
        idx = torch.randint(0, n, (64,))
        xb = data[idx]
        if tmode == "fixed":
            t = torch.full((64,), 250)
        else:
            t = torch.randint(0, T, (64,))
        noise = torch.randn(xb.shape, generator=g)
        ab = AB[t][:, None]
        xnoisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        eps = m(xnoisy, t, torch.zeros(64, dtype=torch.long))
        loss = torch.mean((eps - noise) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 150 == 0 or s == steps - 1: print(f"  {name} step {s}: {loss.item():.4f}", flush=True)

print("A) fixed t=250, 64 imgs"); probe("A", xt[:64], "fixed")
print("B) all t, 64 imgs");    probe("B", xt[:64], "all")
