"""PIXEL GEN phase 2: categorical palette head fine-tune.
The continuous v-head leaves per-pixel dithering (multimodal posterior averaged
over 7.7k unique images). Fix: add a 1024x16 cross-entropy head over the FIXED
16-color palette, fine-tune the pretrained trunk at low LR. Sampling argmaxes
the head -> every pixel lands EXACTLY on a palette color: crisp pixel art.
Usage: python src/pixel_ft.py <minutes> [steps]
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import pixel_data
from pixel_train import PixelDiffusion, AB, T, time_embed

HID = 1280
CKPT = os.path.join(os.path.dirname(__file__), "..", "out", "pixel.pt")
FTCKPT = os.path.join(os.path.dirname(__file__), "..", "out", "pixel_ft.pt")
N_CLASSES = pixel_data.N_CLASSES

class PaletteHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base                      # pretrained PixelDiffusion trunk
        self.head = nn.Linear(HID, 1024 * 16)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x, tidx, cls):
        e = self.base.temb(time_embed(tidx)) + self.base.cemb(cls)
        h = self.base.inp(x)
        h = self.base.b1(h, e); h = self.base.b2(h, e); h = self.base.b3(h, e); h = self.base.b4(h, e)
        h = nn.functional.silu(self.base.outln(h))
        return self.base.out(h), self.head(h).reshape(-1, 1024, 16)   # (v, logits)

@torch.no_grad()
def sample_pal(model, cls, n=1, steps=40, temp=1.0):
    """DDIM where x0 comes from the categorical palette head.
    Intermediate steps: MULTINOMIAL sampling at `temp` (exploration — pure
    argmax collapses unsure pixels to the white majority). Final step: argmax
    for crisp pixels."""
    model.eval()
    x = torch.randn(n, 3072)
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    pal = torch.tensor(pixel_data.PALETTE, dtype=torch.float32) / 127.5 - 1
    for i, t in enumerate(ts):
        ab_t = AB[t]
        ab_prev = AB[ts[i + 1]] if i + 1 < len(ts) else AB[0]
        _, logits = model(x, torch.full((n,), t), torch.full((n,), cls, dtype=torch.long))
        lg = logits.reshape(n, 1024, 16)
        if i + 1 < len(ts) and temp > 0:
            probs = torch.softmax(lg / temp, -1)
            idx = torch.multinomial(probs.reshape(-1, 16), 1).reshape(n, 1024)
        else:
            idx = lg.argmax(-1)
        x0 = pal[idx].reshape(n, 3072)                            # exactly on-palette
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt().clamp(min=1e-3)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x0

def to_img_pal(x0):
    """x0 already on-palette; just map to RGB uint8"""
    v = ((x0.reshape(-1, 32, 32, 3) + 1) * 127.5)
    return v.round().clamp(0, 255).numpy().astype(np.uint8)

def save_sheet_pal(model, path):
    try:
        from PIL import Image
    except ImportError:
        return
    grid = np.zeros((N_CLASSES * 32, 4 * 32, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        rows_c = to_img_pal(sample_pal(model, c, n=4))
        for k in range(4):
            grid[c*32:(c+1)*32, k*32:(k+1)*32] = rows_c[k]
    Image.fromarray(grid).resize((4 * 160, N_CLASSES * 160), Image.NEAREST).save(path)

def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    HERE = os.path.dirname(__file__)
    x = np.load(os.path.join(HERE, "..", "data", "pixel_train.npy")).astype(np.float32)
    idx_cache = os.path.join(HERE, "..", "data", "pixel_idx.npy")
    if os.path.exists(idx_cache):
        x_idx = torch.tensor(np.load(idx_cache), dtype=torch.long)
    else:
        pal = np.array(pixel_data.PALETTE, dtype=np.float32)             # [16,3]
        d = ((x[:, :, :, None, :] - pal[None, None, None, :, :]) ** 2).sum(-1)
        x_idx = torch.tensor(d.argmin(-1).reshape(len(x), -1), dtype=torch.long)   # [N,1024]
        np.save(idx_cache, x_idx.numpy())
    xt = torch.tensor(x.reshape(len(x), -1) / 127.5 - 1)
    n = len(xt)
    cl = torch.arange(n) % N_CLASSES

    base = PixelDiffusion()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    base.load_state_dict(ck["ema"])
    model = PaletteHead(base)
    start = 0
    f = None
    if os.path.exists(FTCKPT):
        f = torch.load(FTCKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(f["model"]); start = f["step"]
        print(f"resumed pixel ft from step {start}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=1e-5)
    if start and f is not None:
        if "opt" in f: opt.load_state_dict(f["opt"])
        ema = {k: v.detach().clone() for k, v in f["ema"].items()}
    else:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g = torch.Generator().manual_seed(17 + start)
    rng = np.random.default_rng(31 + start)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    print(f"== PIXEL FT: {sum(p.numel() for p in model.parameters()):,} params, from step {start}", flush=True)
    for step in range(start, max_steps):
        idx = torch.from_numpy(rng.integers(0, n, 128))
        xb, cb, yb = xt[idx], cl[idx], x_idx[idx]
        t = torch.from_numpy(rng.integers(0, T, len(idx)))
        noise = torch.randn(xb.shape, generator=g)
        ab = AB[t][:, None]
        xnoisy = ab.sqrt() * xb + (1 - ab).sqrt() * noise
        v_pred, logits = model(xnoisy, t, cb)
        v_tgt = ab.sqrt() * noise - (1 - ab).sqrt() * xb
        loss = torch.mean((v_pred - v_tgt) ** 2) + ce(logits.reshape(-1, 16), yb.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k in ema: ema[k].mul_(0.999).add_(model.state_dict()[k], alpha=0.001)
        if (step + 1) % 200 == 0 or step == max_steps - 1:
            print(f"step {step+1}/{max_steps} loss {loss.item():.4f} ({(step+1-start)/max(time.time()-t0,1):.2f} it/s)", flush=True)
        if (step + 1) % 3000 == 0:
            em = PaletteHead(base); em.load_state_dict(ema)
            save_sheet_pal(em, os.path.join(HERE, "..", "data", "pixel_ft_samples.png"))
        if (time.time() - t0) / 60 >= minutes and step < max_steps - 1:
            print(f"FT chunk limit at step {step+1}", flush=True)
            torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(),
                        "step": step + 1}, FTCKPT)
            print("FT_CHUNK_SAVED", f"at {step+1}/{max_steps}", flush=True); break
    if step + 1 >= max_steps or True:   # always persist final state
        torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(),
                    "step": step + 1}, FTCKPT)
    done = step + 1 >= max_steps
    print("FT_COMPLETE" if done else "FT_CHUNK_SAVED", f"at {step+1}/{max_steps}", flush=True)

if __name__ == "__main__":
    main()
