"""Quick sampling-parameter sweep on a checkpoint to find clean-generation params."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")

def load(stage):
    ck = torch.load(os.path.join(OUT, f"{stage}.pt"), map_location="cpu", weights_only=False)
    m = Transformer(configs()[stage]); m.load_state_dict(ck["model"]); m.eval()
    return m, Tokenizer.load(os.path.join(DATA, "vocab.json"))

@torch.no_grad()
def chat(model, tk, persona, hist, user, max_new, temp, topk, topp):
    s = f"<bos><pers:{persona}>"
    for u, a in hist: s += f"<user>{u}<ai>{a}"
    s += f"<user>{user}<ai>"
    ids = torch.tensor([tk.encode(s)])
    out = model.generate(ids, max_new=max_new, temperature=temp, top_k=topk, top_p=topp,
                         stop_tokens={tk.stoi["<user>"], tk.stoi["<eos>"]})
    return tk.decode(out[0, ids.shape[1]:].tolist())

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "17m"
    m, tk = load(stage)
    tests = [
        ("fake", [], "hi"),
        ("sigma", [], "i think i should text my ex at 3am"),
        ("sigma", [], "give me dog names"),
        ("fake", [("give me dog names", "here we go: rocky, luna, bella."), ("stop", "stopped. say continue when ready.")], "continue"),
        ("fake", [], "my name is dino"),
    ]
    for temp, topk, topp in [(0.55, 10, 1.0), (0.6, 12, 0.9), (0.7, 8, 1.0), (0.5, 6, 1.0)]:
        print(f"=== temp={temp} topk={topk} topp={topp}")
        torch.manual_seed(7)
        for pers, h, u in tests:
            a = chat(m, tk, pers, h, u, 40, temp, topk, topp)
            print(f"  [{pers:5s}] {u!r} -> {a[:130]}")
        print()

if __name__ == "__main__":
    main()
