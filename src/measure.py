"""Measure P(<user>) after good sentence ends vs mid-drift. If separable,
adaptive stopping can use the model's own turn-end belief."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")

ck = torch.load(os.path.join(OUT, "17m.pt"), map_location="cpu", weights_only=False)
m = Transformer(configs()["17m"]); m.load_state_dict(ck["model"]); m.eval()
tk = Tokenizer.load(os.path.join(DATA, "vocab.json"))
U = tk.stoi["<user>"]

@torch.no_grad()
def probs_after(text):
    ids = torch.tensor([tk.encode(text)])
    lg = m(ids)["logits"][0, -1]
    p = F.softmax(lg, -1)
    top = torch.topk(p, 5)
    return float(p[U]), [(tk.itos[i], round(float(v),2)) for v,i in zip(top.values, top.indices)]

cases = [
    ("after clean 1-sentence greet", "<bos><pers:fake><user>hi<ai>hey hey!! great to see you!!"),
    ("after name answer", "<bos><pers:fake><user>my name is dino<ai>dino!! love that name!! noted forever!"),
    ("after sigma verdict", "<bos><pers:sigma><user>i think i should text my ex at 3am<ai>no. boredom isn't a good enough reason."),
    ("mid-sentence (should NOT stop)", "<bos><pers:fake><user>hi<ai>hey hey!! great to"),
    ("after list answer", "<bos><pers:sigma><user>give me dog names<ai>here we go: tank, luna, kilo."),
    ("after drift token", "<bos><pers:fake><user>hi<ai>hey hey!! great to see you!! so"),
]
for name, t in cases:
    pu, top = probs_after(t)
    print(f"{name:36s} P(<user>)={pu:.2f}  top={top}")
