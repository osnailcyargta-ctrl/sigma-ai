"""PIXEL GEN — real diffusion model trained FROM ZERO on the procedural 32x32
pixel-art dataset. True generative model: at sample time it starts from pure
Gaussian noise and denoises step by step — nothing is picked from a template.

Architecture v2 (optimization-fixed):
  - pre-LN residual MLP with FiLM conditioning (timestep + class modulate
    every block via scale & shift)
  - v-prediction: target v = sqrt(ab)*eps - sqrt(1-ab)*x. Unlike eps-MSE,
    v correlates with the input at ALL noise levels, so a small MLP actually
    optimizes it (plain eps-MPL stalled at loss 1.0 — diagnosed & fixed).
  ~10.4M params, single ONNX file, 25-step DDIM sampler in the browser ~1-2s.

Chunked + resumable like the LLM trainer:
  python src/pixel_train.py <minutes> [max_steps]
Saves out/pixel.pt (model + EMA + optimizer + step) every 400 steps.
Also renders data/pixel_samples.png at every eval so quality is visible.
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
import pixel_data

T = 300                 # diffusion steps
HID = 1280
EMB = 192
N_CLASSES = pixel_data.N_CLASSES
CKPT = os.path.join(OUT, "pixel.pt")

# ---------------- cosine schedule ----------------
def alphas_bar():
    s = 0.008
    ts = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(max=0.999)

AB = alphas_bar().float()               # [T+1] float32

def time_embed(tidx, dim=64):
    """sinusoidal embedding — SAME formula is mirrored in the ONNX graph.
    Standard DDPM scaling: ang = t * freqs (NO extra T multiplier — the ×T
    version explodes channel 0 to 90k radians and makes t uninformative)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    ang = tidx.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

# ---------------- model ----------------
class Block(nn.Module):
    """pre-LN residual block with FiLM conditioning"""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HID)
        self.fc = nn.Linear(HID, HID)
        self.film = nn.Linear(EMB, 2 * HID)
    def forward(self, h, e):
        sc, sh = self.film(e).chunk(2, dim=-1)
        return h + self.fc(nn.functional.silu(self.ln(h)) * (1 + sc) + sh)

class PixelDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(64, EMB), nn.SiLU(), nn.Linear(EMB, EMB))
        self.cemb = nn.Embedding(N_CLASSES, EMB)
        self.inp = nn.Linear(3072, HID)
        self.b1 = Block(); self.b2 = Block(); self.b3 = Block(); self.b4 = Block()
        self.outln = nn.LayerNorm(HID)
        self.out = nn.Linear(HID, 3072)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, tidx, cls):
        e = self.temb(time_embed(tidx)) + self.cemb(cls)
        h = self.inp(x)
        h = self.b1(h, e); h = self.b2(h, e); h = self.b3(h, e); h = self.b4(h, e)
        return self.out(nn.functional.silu(self.outln(h)))   # predicted v

# ---------------- sampling (shared logic with docs/pixel.js) ----------------
@torch.no_grad()
def sample(model, cls, n=1, steps=25):
    model.eval()
    x = torch.randn(n, 3072)
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    for i, t in enumerate(ts):
        ab_t = AB[t]
        ab_prev = AB[ts[i + 1]] if i + 1 < len(ts) else AB[0]
        v = model(x, torch.full((n,), t), torch.full((n,), cls, dtype=torch.long))
        # v-prediction -> x0 and eps
        x0 = ab_t.sqrt() * x - (1 - ab_t).sqrt() * v
        eps = (1 - ab_t).sqrt() * x + ab_t.sqrt() * v
        x0 = x0.clamp(-1.2, 1.2)                    # light dynamic clamping
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x0.clamp(-1, 1)

def to_img(x0):
    """[-1,1] float -> quantized palette RGB (n,32,32,3 uint8)"""
    pal = torch.tensor(pixel_data.PALETTE, dtype=torch.float32)
    v = ((x0.reshape(-1, 32, 32, 3) + 1) / 2).clamp(0, 1)
    d = ((v[:, :, :, None, :] * 255 - pal[None, None, None, :, :]) ** 2).sum(-1)
    idx = d.argmin(-1)
    return pal[idx.reshape(-1, 32, 32)].numpy().astype(np.uint8)

def save_sheet(model, path):
    try:
        from PIL import Image
    except ImportError:
        return
    grid = np.zeros((N_CLASSES * 32, 4 * 32, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        rows_c = to_img(sample(model, c, n=4))
        for k in range(4):
            grid[c*32:(c+1)*32, k*32:(k+1)*32] = rows_c[k]
    Image.fromarray(grid).resize((4 * 160, N_CLASSES * 160), Image.NEAREST).save(path)

# ---------------- train ----------------
def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 9
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    os.makedirs(OUT, exist_ok=True)
    x = np.load(os.path.join(DATA, "pixel_train.npy")).astype(np.float32) / 127.5 - 1
    xt = torch.tensor(x.reshape(len(x), -1))
    cl = torch.arange(len(xt)) % N_CLASSES
    n = len(xt)

    model = PixelDiffusion()
    WARM = 300
    def lr_at(s):
        w = min(1.0, (s + 1) / WARM)
        cos = 0.15 + 0.85 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps)) / 2
        return 2.2e-3 * w * cos
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, betas=(0.9, 0.99), weight_decay=1e-5)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        ema = ck["ema"]; start = ck["step"]
        print(f"resumed pixel diffusion from step {start}", flush=True)
    npar = sum(p.numel() for p in model.parameters())
    print(f"== PIXEL DIFFUSION: {npar:,} params · T={T} · {n} images · target {max_steps} steps", flush=True)

    g = torch.Generator().manual_seed(4242 + start)
    t0 = time.time()
    rng = np.random.default_rng(99 + start)
    model.train()
    for step in range(start, max_steps):
        for gp in opt.param_groups: gp["lr"] = lr_at(step)
        idx = torch.from_numpy(rng.integers(0, n, 128))
        xb, cb = xt[idx], cl[idx]
        t = torch.from_numpy(rng.integers(0, T, len(idx)))
        noise = torch.randn(xb.shape, generator=g)
        ab = AB[t][:, None]
        xt_noisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        v_tgt = ab.sqrt() * noise - (1 - ab).sqrt() * xb        # v-prediction target
        v = model(xt_noisy, t, cb)
        loss = torch.mean((v - v_tgt) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # EMA update
        with torch.no_grad():
            msd = model.state_dict()
            for k in ema: ema[k].mul_(0.999).add_(msd[k], alpha=0.001)
        if (step + 1) % 100 == 0 or step == max_steps - 1:
            el = time.time() - t0
            print(f"step {step+1}/{max_steps} loss {loss.item():.4f} "
                  f"({(step+1-start)/max(el,1e-9):.2f} it/s, {el/60:.1f} min)", flush=True)
        if (step + 1) % 2000 == 0 or step == max_steps - 1:
            model.eval()
            eval_model = PixelDiffusion(); eval_model.load_state_dict(ema)
            save_sheet(eval_model, os.path.join(DATA, "pixel_samples.png"))
            torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(),
                        "step": step + 1, "cfg": {"T": T, "HID": HID, "EMB": EMB}}, CKPT)
            model.train()
        if (time.time() - t0) / 60 >= minutes and step < max_steps - 1:
            print(f"CHUNK time limit at step {step+1}", flush=True)
            break
    done = step + 1 >= max_steps
    print("PIXEL_COMPLETE" if done else "PIXEL_CHUNK_SAVED", f"at step {step+1}/{max_steps}", flush=True)

if __name__ == "__main__":
    main()
