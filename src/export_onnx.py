"""Export checkpoint -> ONNX with a KV-cache graph for fast incremental
decoding in the browser (prefill once, then 1 token per forward).
Inputs : idx [1,T] int64, past_k/past_v [L,1,H,Tp,Dh] fp32 (Tp=0 -> prefill)
Outputs: logits [1,T,V], pooled [1,D], intent_logits [1,5],
         present_k/present_v [L,1,H,Tp+T,Dh]
Also verifies the cached ONNX loop against full-forward PyTorch greedy decode.
"""
import os, sys, json
import torch
import numpy as np
torch.set_num_threads(1)  # deterministic reductions
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")
DOCS_MODELS = os.path.join(HERE, "..", "docs", "models")

class CacheWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, idx, past_k, past_v):
        o = self.m.forward_cached(idx, past_k, past_v)
        return o["logits"], o["pooled"], o["intent_logits"], o["present_k"], o["present_v"]

def export(stage, verify=True):
    ck = torch.load(os.path.join(OUT, f"{stage}.pt"), map_location="cpu", weights_only=False)
    cfg = configs(vocab=ck["cfg"]["vocab_size"])[stage]
    cfg.block_size = ck["cfg"].get("block_size", cfg.block_size)   # v10 ships ctx 512
    m = Transformer(cfg)
    m.load_state_dict(ck["model"])
    m.eval()

    w = CacheWrapper(m)
    L, H = cfg.n_layers, cfg.n_heads
    Dh = cfg.d_model // cfg.n_heads
    dummy = torch.randint(0, cfg.vocab_size, (1, 16))
    empty = torch.zeros(L, 1, H, 0, Dh)
    os.makedirs(DOCS_MODELS, exist_ok=True)
    path = os.path.join(DOCS_MODELS, f"{stage}.onnx")
    try:
        torch.onnx.export(
            w, (dummy, empty, empty), path,
            input_names=["idx", "past_k", "past_v"],
            output_names=["logits", "pooled", "intent_logits", "present_k", "present_v"],
            dynamic_axes={"idx": {1: "T"},
                          "past_k": {3: "Tp"}, "past_v": {3: "Tp"},
                          "logits": {1: "T"},
                          "present_k": {3: "TpT"}, "present_v": {3: "TpT"}},
            opset_version=14, do_constant_folding=True, dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            w, (dummy, empty, empty), path,
            input_names=["idx", "past_k", "past_v"],
            output_names=["logits", "pooled", "intent_logits", "present_k", "present_v"],
            dynamic_axes={"idx": {1: "T"},
                          "past_k": {3: "Tp"}, "past_v": {3: "Tp"},
                          "logits": {1: "T"},
                          "present_k": {3: "TpT"}, "present_v": {3: "TpT"}},
            opset_version=14, do_constant_folding=True,
        )
    # inline external weights into a single .onnx file (PWA-friendly)
    import onnx
    m2 = onnx.load(path, load_external_data=True)
    for t in m2.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            t.data_location = onnx.TensorProto.DEFAULT
            t.ClearField("external_data")
    onnx.save(m2, path)
    size = os.path.getsize(path) / 1e6
    print(f"OK {stage}.onnx  {size:.1f} MB", end="  ")

    if not verify:
        print()
        return path

    # ---- numeric verification: ONNX cached decode vs PyTorch full-forward greedy ----
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    vp = os.path.join(DATA, "vocab3.json")
    tk = Tokenizer.load(vp if os.path.exists(vp) else os.path.join(DATA, "vocab.json"))

    def onnx_logits(ids, pk, pv):
        feeds = {"idx": np.asarray([ids], dtype=np.int64),
                 "past_k": pk if pk is not None else np.zeros((L, 1, H, 0, Dh), np.float32),
                 "past_v": pv if pv is not None else np.zeros((L, 1, H, 0, Dh), np.float32)}
        lg, _, _, pk2, pv2 = sess.run(None, feeds)
        return lg[0, -1], pk2, pv2

    def torch_greedy(prompt_ids, n=24):
        """ground truth: full recompute every step (original forward)"""
        idx = torch.tensor([prompt_ids])
        banned = {0, 1}
        out = []
        for _ in range(n):
            lg = m(idx)["logits"][0, -1].clone()
            for b in banned:
                lg[b] = -1e9
            t = int(lg.argmax())
            if t in (tk.stoi["<user>"], tk.stoi["<eos>"]):
                break
            idx = torch.cat([idx, torch.tensor([[t]])], dim=1)
            out.append(t)
        return out

    prompt = tk.encode("<bos><pers:sigma><user>hi<ai>")
    pk = pv = None
    onnx_tok = []
    lg, pk, pv = onnx_logits(prompt, None, None)
    banned = {0, 1}
    for _ in range(24):
        row = lg.astype(np.float64).copy()
        for b in banned:
            row[b] = -1e9
        t = int(row.argmax())
        if t in (tk.stoi["<user>"], tk.stoi["<eos>"]):
            break
        onnx_tok.append(t)
        lg, pk, pv = onnx_logits([t], pk, pv)

    truth = torch_greedy(prompt, 24)
    match = sum(1 for a, b in zip(onnx_tok, truth) if a == b)
    # near-tie aware: a flip counts as OK if torch top-2 gap is tiny
    near = 0
    idx = torch.tensor([prompt])
    with torch.no_grad():
        for i, t in enumerate(truth[:len(onnx_tok)]):
            lg = m(idx)["logits"][0, -1]
            top2 = torch.topk(lg, 2).values
            if (top2[0] - top2[1]).item() < 2e-3:
                near += 1
            idx = torch.cat([idx, torch.tensor([[t]])], dim=1)
    status = "PERFECT" if match == len(truth) == len(onnx_tok) else f"strict {match}/{min(len(truth), len(onnx_tok))} (+{near} near-ties)"
    print(f"cache-decode vs full-forward: {status}")
    print(f"   onnx:  {' '.join(tk.itos[t] for t in onnx_tok[:10])!r}")
    print(f"   torch: {' '.join(tk.itos[t] for t in truth[:10])!r}")
    return path

if __name__ == "__main__":
    import math
    for s in sys.argv[1:]:
        p = export(s)
        # github hard-limits files at 100MB: byte-split big exports into
        # .onnx.partN (browser fetchModelBuf concats before parsing — safe at
        # any split point since parsing happens on the reassembled buffer)
        size = os.path.getsize(p)
        if size > 95_000_000:
            with open(p, "rb") as f:
                data = f.read()
            n = math.ceil(size / 90_000_000)
            per = math.ceil(size / n)
            base = os.path.splitext(p)[0]
            for i in range(n):
                with open(f"{base}.onnx.part{i+1}", "wb") as f:
                    f.write(data[i*per:(i+1)*per])
            os.remove(p)
            print(f"   sharded -> {n} parts x {per/1e6:.1f} MB ({os.path.basename(base)}.onnx.part1..{n})")
