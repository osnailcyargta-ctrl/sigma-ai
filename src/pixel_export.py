"""Export the pixel diffusion model (palette-head path) to a single-file ONNX.
Inputs:  x [1,3072] float32 (noisy image in [-1,1])
         t [1]       float32 (timestep index 0..T-1)
         cls [1]     int64   (category id)
Outputs: v     [1,3072]     (unused by the browser sampler, kept for diagnostics)
         logits [1,1024,16] (palette logits — x0 = palette[argmax/softmax])
The sinusoidal time embedding is computed INSIDE the graph (mirrors Python).
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import pixel_data
from pixel_train import T, time_embed

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "docs", "models")

class ExportGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.base = m.base
        self.head = m.head
    def forward(self, x, t, cls):
        b = self.base
        e = b.temb(time_embed(t)) + b.cemb(cls)
        h = b.inp(x)
        h = b.b1(h, e); h = b.b2(h, e); h = b.b3(h, e); h = b.b4(h, e)
        h = nn.functional.silu(b.outln(h))
        logits = self.head(h).reshape(1, 1024, 16)
        return logits

def main():
    from pixel_ft import PaletteHead
    from pixel_train import PixelDiffusion
    f = torch.load(os.path.join(os.path.dirname(__file__), "..", "out", "pixel_ft.pt"),
                   map_location="cpu", weights_only=False)
    base = PixelDiffusion()
    model = PaletteHead(base)
    model.load_state_dict(f["ema"])
    model.eval()
    g = ExportGraph(model).eval()
    x = torch.randn(1, 3072)
    t = torch.tensor([137.0])
    cls = torch.tensor([0])
    with torch.no_grad():
        lg = g(x, t, cls)
    npar = sum(p.numel() for p in g.parameters())
    torch.onnx.export(
        g, (x, t, cls),
        os.path.join(OUTDIR, "pixel.onnx"),
        input_names=["x", "t", "cls"],
        output_names=["logits"],
        dynamic_axes={},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,                      # legacy exporter: inline weights, ORT-safe Split
    )
    # single file > github 100MB limit -> byte-split (browser concats before parse)
    p = os.path.join(OUTDIR, "pixel.onnx")
    size = os.path.getsize(p)
    if size > 95_000_000:
        raw = open(p, "rb").read()
        n = 2
        per = (len(raw) + n - 1) // n
        for i in range(n):
            open(f"{p}.part{i+1}", "wb").write(raw[i*per:(i+1)*per])
        os.remove(p)
        print(f"sharded -> pixel.onnx.part1..{n} x {per/1e6:.1f} MB")
    else:
        print(f"exported pixel.onnx: {npar:,} params, {size/1e6:.1f} MB")
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(OUTDIR, "pixel.onnx"), providers=["CPUExecutionProvider"])
        o = s.run(None, {"x": x.numpy(), "t": t.numpy(), "cls": cls.numpy()})
        print("ort ok, logits shape", o[0].shape)
    except ImportError:
        print("onnxruntime not installed — skipping runtime check")

if __name__ == "__main__":
    main()
