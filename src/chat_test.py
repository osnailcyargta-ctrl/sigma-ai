"""REAL chat test against a checkpoint. No mocks: loads weights, generates,
prints transcript + intent analysis. This is the evidence the model works."""
import os, sys, json, pickle
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")

def load(stage):
    ck = torch.load(os.path.join(OUT, f"{stage}.pt"), map_location="cpu", weights_only=False)
    cfg = configs()[stage]
    m = Transformer(cfg)
    m.load_state_dict(ck["model"])
    m.eval()
    tk = Tokenizer.load(os.path.join(DATA, "vocab.json"))
    return m, tk, ck["params"]

@torch.no_grad()
def chat(model, tk, persona, history, user, max_new=44, temp=0.7):
    s = "<bos><pers:%s>" % persona
    for u, a in history:
        s += f"<user>{u}<ai>{a}"
    s += f"<user>{user}<ai>"
    ids = torch.tensor([tk.encode(s)])
    out = model.generate(ids, max_new=max_new, temperature=temp, top_k=25, top_p=0.9,
                         stop_tokens={tk.stoi["<user>"], tk.stoi["<eos>"]})
    gen = out[0, ids.shape[1]:].tolist()
    return tk.decode(gen)

@torch.no_grad()
def intent_of(model, tk, user):
    s = f"<bos><pers:sigma><user>{user}"
    ids = torch.tensor([tk.encode(s)])
    o = model(ids)
    probs = torch.softmax(o["intent_logits"], -1)[0]
    return int(o["intent_logits"][0].argmax()), [round(float(p), 2) for p in probs]

LABELS = ["0:none", "1:tictactoe", "2:rps", "3:dice", "4:ask-list"]

def show(model, tk, title, persona, hist, u):
    a = chat(model, tk, persona, hist, u)
    print(f"  [U] {u}")
    print(f"  [AI/{persona}] {a}")
    print()

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "2m"
    model, tk, npar = load(stage)
    print("=" * 72)
    print(f"REAL CHAT TEST — stage '{stage}' — {npar:,} parameters — NO templates, NO keywords:")
    print("every line below is generated live by the neural network forward pass")
    print("=" * 72)

    print("-- 1) greeting (fake persona)")
    show(model, tk, "greet", "fake", [], "hi")
    print("-- 2) greeting (sigma persona)")
    show(model, tk, "greet", "sigma", [], "hey")

    print("-- 3) STOP/CONTINUE memory test: list -> stop -> continue")
    hist = []
    a = chat(model, tk, "sigma", hist, "give me dog names"); print(f"  [U] give me dog names\n  [AI] {a}\n")
    hist.append(("give me dog names", a))
    a2 = chat(model, tk, "sigma", hist, "stop"); print(f"  [U] stop\n  [AI] {a2}\n")
    hist.append(("stop", a2))
    a3 = chat(model, tk, "sigma", hist, "continue"); print(f"  [U] continue\n  [AI] {a3}   <-- must be NEW names, no repeats\n")

    print("-- 4) name memory test")
    h = []
    a = chat(model, tk, "fake", h, "my name is dino"); print(f"  [U] my name is dino\n  [AI] {a}\n")
    h.append(("my name is dino", a))
    show(model, tk, "name", "fake", h, "what's my name?")

    print("-- 5) sigma honest judgment")
    show(model, tk, "judge", "sigma", [], "i think i should text my ex at 3am")

    print("-- 6) custom persona (unseen prompt, generalization)")
    show(model, tk, "custom", "custom", [], "hello!")

    print("-- 7) INTENT analysis (semantic invite detection; test phrases are UNSEEN)")
    tests = ["how about a quick tic tac toe duel", "hit me with them dice",
             "any games on this thing?", "tic tac toe is for kids",
             "wanna throw rock paper scissors", "i collect vintage dice",
             "im bored, play with me", "who invented dice"]
    for t in tests:
        li, pr = intent_of(model, tk, t)
        print(f"  {t!r:45s} -> {LABELS[li]:14s} probs {pr}")

if __name__ == "__main__":
    main()
