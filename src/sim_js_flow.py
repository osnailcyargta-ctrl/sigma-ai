"""Simulate docs/ai.js generate() flow (old-model path) against the real ONNX.
Validates: empty-past prefill, cache stepping, sampling with rep penalty,
decode with <mem> extraction."""
import os, sys
import torch; torch.set_num_threads(1)
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import Tokenizer
import onnxruntime as ort

STAGE = sys.argv[1] if len(sys.argv) > 1 else "17m"
MDIMS = {"2m":[5,5,170],"5m":[6,8,256],"10m":[6,8,368],"17m":[8,8,416],"20m":[8,8,448],"22m":[8,8,472]}
L, H, D = MDIMS[STAGE]; Dh = D // H

tk = Tokenizer.load("data/vocab.json")
sess = ort.InferenceSession(f"docs/models/{STAGE}.onnx", providers=["CPUExecutionProvider"])
cache = {"k": None, "v": None}

def run_step(ids):
    feeds = {"idx": np.asarray([ids], np.int64)}
    feeds["past_k"] = cache["k"] if cache["k"] is not None else np.zeros((L,1,H,0,Dh), np.float32)
    feeds["past_v"] = cache["v"] if cache["v"] is not None else np.zeros((L,1,H,0,Dh), np.float32)
    out = sess.run(None, feeds)
    cache["k"], cache["v"] = out[3], out[4]
    lg = out[0]
    T, V = lg.shape[1], lg.shape[2]
    return lg[0, T-1]

def sample(row, temp, topK, topP, ban, repPen, repWindow, gen):
    z = row.astype(np.float64).copy()
    for b in ban: z[b] = -1e9
    if gen:
        recent = set(gen[-repWindow:])
        for v in recent:
            z[v] = z[v]/repPen if z[v] > 0 else z[v]*repPen
    z -= z.max(); p = np.exp(z); p /= p.sum()
    idx = np.argsort(-p)[:topK]
    cum = np.cumsum(p[idx])
    keep = idx[:np.searchsorted(cum, topP) + 1]
    keep = keep[:topK]
    p2 = p[keep] / p[keep].sum()
    return int(np.random.choice(keep, p=p2)) if temp > 0.01 else int(keep[0])

def generate(prompt_text, maxNew=72, temp=0.65, topK=20, topP=0.9, repPen=1.28, repWindow=14):
    cache["k"] = cache["v"] = None
    ids = tk.encode(prompt_text)
    stop = {tk.stoi["<user>"], tk.stoi["<eos>"]}
    ban = {0, 1, tk.stoi["<memory>"], tk.stoi["</memory>"]}
    row = run_step(ids)
    out_ids = []
    for _ in range(maxNew):
        t = sample(row, temp, topK, topP, ban, repPen, repWindow, out_ids)
        out_ids.append(t)
        if t in stop: break
        row = run_step([t])
    # mems exactly like the JS decode: collect <mem>..</mem> spans
    mems, cur, inm = [], "", False
    for t in out_ids:
        w = tk.itos[t]
        if w == "<mem>": inm, cur = True, ""
        elif w == "</mem>":
            inm = False
            if cur.strip(): mems.append(cur.strip())
        elif inm: cur += (" " if cur else "") + w
    return tk.decode(out_ids), mems

print("=" * 70)
print(f"JS-FLOW SIMULATION on {STAGE} (old-model path: no think phase)")
print("=" * 70)
hist = []
for u in ["hi", "give me dog names", "stop", "continue", "can we play dice?"]:
    s = "<bos><pers:sigma>" + "".join(f"<user>{hu}<ai>{ha}" for hu, ha in hist[-3:]) + f"<user>{u}<ai>"
    text, mems = generate(s)
    print(f"[U] {u}")
    print(f"[AI] {text}")
    if mems: print(f"     mems -> long-term: {mems}")
    hist.append((u, text))
    print()
