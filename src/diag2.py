"""Single-process: where exactly do onnx-cache and torch-full diverge?"""
import os, sys
import torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer
import onnxruntime as ort

stage = "2m"
ck = torch.load(f"out/{stage}.pt", map_location="cpu", weights_only=False)
cfg = configs()[stage]
m = Transformer(cfg); m.load_state_dict(ck["model"]); m.eval()
torch.set_num_threads(1)
tk = Tokenizer.load("data/vocab.json")
L, H = cfg.n_layers, cfg.n_heads
Dh = cfg.d_model // cfg.n_heads
prompt = tk.encode("<bos><pers:sigma><user>hi<ai>")

sess = ort.InferenceSession(f"docs/models/{stage}.onnx", providers=["CPUExecutionProvider"])

def show_topk(tag, row):
    v = torch.tensor(row)
    top = torch.topk(v, 3)
    print(f"{tag}: " + " | ".join(f"{tk.itos[i]!r}={val:.4f}" for val, i in zip(top.values, top.indices)))

with torch.no_grad():
    lg_t = m(torch.tensor([prompt]))["logits"][0, -1].numpy().copy()
feeds = {"idx": np.asarray([prompt], np.int64),
         "past_k": np.zeros((L,1,H,0,Dh), np.float32),
         "past_v": np.zeros((L,1,H,0,Dh), np.float32)}
lg_o, _, _, pk, pv = sess.run(None, feeds)
lg_o = lg_o[0, -1].copy()
print("maxdiff prefill:", np.abs(lg_t - lg_o).max())
show_topk("torch top3", lg_t)
show_topk("onnx  top3", lg_o)
print("argmax torch:", tk.itos[int(lg_t.argmax())], " onnx:", tk.itos[int(lg_o.argmax())])
