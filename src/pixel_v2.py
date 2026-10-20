"""PixelDiffusion v2 — optimization-fixed architecture:
  - pre-LN residual MLP (linear path from input to output exists from init)
  - FiLM conditioning (timestep+class modulate every block: scale & shift)
  - v-prediction (balanced SNR: the target correlates with x_t at ALL noise
    levels, unlike eps-prediction where low-t targets are invisible)
Probe: loss < 0.35 by step 900 on fresh batches + random t => viable.
"""
import sys, os, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from pixel_train import AB, T, time_embed
import pixel_data

HID = 1024
EMB = 192
N_CLASSES = pixel_data.N_CLASSES

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HID)
        self.fc = nn.Linear(HID, HID)
        self.film = nn.Linear(EMB, 2 * HID)
    def forward(self, h, e):
        sc, sh = self.film(e).chunk(2, dim=-1)
        h = h + self.fc(nn.functional.silu(self.ln(h)) * (1 + sc) + sh)
        return h

class PixelDiffusionV2(nn.Module):
    def __init__(self, target="v"):
        super().__init__()
        self.target = target
        self.temb = nn.Sequential(nn.Linear(64, EMB), nn.SiLU(), nn.Linear(EMB, EMB))
        self.cemb = nn.Embedding(N_CLASSES, EMB)
        self.inp = nn.Linear(3072, HID)
        self.b1, self.b2, self.b3 = Block(), Block(), Block()
        self.outln = nn.LayerNorm(HID)
        self.out = nn.Linear(HID, 3072)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start at 0, grow fast
    def forward(self, x, tidx, cls):
        e = self.temb(time_embed(tidx)) + self.cemb(cls)
        h = self.inp(x)
        h = self.b1(h, e); h = self.b2(h, e); h = self.b3(h, e)
        return self.out(nn.functional.silu(self.outln(h)))

def train_probe(target, steps=900, lr=2e-3):
    torch.manual_seed(0)
    m = PixelDiffusionV2(target)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
    x = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "pixel_train.npy")).astype(np.float32) / 127.5 - 1
    xt = torch.tensor(x.reshape(len(x), -1)); n = len(xt)
    cl = torch.arange(n) % 6
    g = torch.Generator().manual_seed(1); rng = np.random.default_rng(5)
    hist = []
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, 96))
        xb, cb = xt[idx], cl[idx]
        t = torch.from_numpy(rng.integers(0, T, len(idx)))
        noise = torch.randn(xb.shape, generator=g)
        ab = AB[t][:, None]
        xnoisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        if target == "v":
            tgt = ab.sqrt() * noise - (1 - ab).sqrt() * xb
        else:
            tgt = noise
        out = m(xnoisy, t, cb)
        loss = torch.mean((out - tgt) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
    print(f"target={target}: " + " ".join(f"{hist[i]:.3f}" for i in [0, 100, 300, 600, 899]), flush=True)
    return m, hist

if __name__ == "__main__":
    train_probe("v")
    train_probe("eps")
