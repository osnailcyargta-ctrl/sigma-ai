"""Diagnose cache decode: torch-cache vs torch-full vs onnx-cache logits."""
import os, sys
import torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer
import onnxruntime as ort

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")

stage = sys.argv[1] if len(sys.argv) > 1 else "2m"
ck = torch.load(os.path.join(OUT, f"{stage}.pt"), map_location="cpu", weights_only=False)
cfg = configs()[stage]
m = Transformer(cfg); m.load_state_dict(ck["model"]); m.eval()
tk = Tokenizer.load(os.path.join(DATA, "vocab.json"))
L, H = cfg.n_layers, cfg.n_heads
Dh = cfg.d_model // cfg.n_heads

prompt = tk.encode("<bos><pers:sigma><user>hi<ai>")

def torch_full_last(ids):
    with torch.no_grad():
        return m(torch.tensor([ids]))["logits"][0, -1]

def torch_cache_last(ids, pk, pv):
    with torch.no_grad():
        o = m.forward_cached(torch.tensor([ids]), pk, pv)
        return o["logits"][0, -1], o["present_k"], o["present_v"]

# step 1: prefill comparison
full = torch_full_last(prompt)
c, pk, pv = torch_cache_last(prompt, None, None)
print("prefill  torch-cache vs torch-full  maxdiff:", (full - c).abs().max().item())

sess = ort.InferenceSession("docs/models/2m.onnx" if stage == "2m" else f"docs/models/{stage}.onnx",
                            providers=["CPUExecutionProvider"])
def onnx_last(ids, pk, pv):
    feeds = {"idx": np.asarray([ids], np.int64),
             "past_k": pk if pk is not None else np.zeros((L, 1, H, 0, Dh), np.float32),
             "past_v": pv if pv is not None else np.zeros((L, 1, H, 0, Dh), np.float32)}
    lg, _, _, pk2, pv2 = sess.run(None, feeds)
    return lg[0, -1], pk2, pv2

o, opk, opv = onnx_last(prompt, None, None)
print("prefill  onnx-cache vs torch-full  maxdiff:", np.abs(full.numpy() - o).max())

# step 2: generate 6 tokens, compare each step
tok = int(full.argmax())
ids = prompt + [tok]
for step in range(6):
    full = torch_full_last(ids)
    c, pk, pv = torch_cache_last([ids[-1]], pk, pv)
    o, opk, opv = onnx_last([ids[-1]], opk, opv)
    print(f"step {step}: cache-vs-full {(full - c).abs().max().item():.6f}  "
          f"onnx-vs-full {np.abs(full.numpy() - o).max():.6f}  "
          f"argmax full={tk.itos[int(full.argmax())]!r} cache={tk.itos[int(c.argmax())]!r} onnx={tk.itos[int(o.argmax())]!r}")
    ids.append(int(full.argmax()))
