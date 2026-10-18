"""REAL chat test against a checkpoint. No mocks: loads weights, generates,
prints transcript + intent analysis. This is the evidence the model works.
Think-aware: new checkpoints stream their <think> plan first (two-phase),
then the answer — exactly like the browser runtime."""
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
    cfg = configs(vocab=ck["cfg"]["vocab_size"])[stage]
    m = Transformer(cfg)
    m.load_state_dict(ck["model"])
    m.eval()
    vp = os.path.join(DATA, "vocab3.json")
    if not os.path.exists(vp):
        vp = os.path.join(DATA, "vocab2.json")
    tk = Tokenizer.load(vp)
    return m, tk, ck["params"]

@torch.no_grad()
def chat(model, tk, persona, history, user, max_new=70, temp=0.65, verbose_think=True):
    """two-phase generate: <think> plan (models that have the token), then answer."""
    s = "<bos><pers:%s>" % persona
    for u, a in history:
        s += f"<user>{u}<ai>{a}"
    s += f"<user>{user}<ai>"
    ids = tk.encode(s)
    stop = {tk.stoi["<user>"], tk.stoi["<eos>"]}
    think_id = tk.stoi.get("<think>")
    end_think = tk.stoi.get("</think>")
    think_txt = ""

    if think_id is not None:
        cur = ids + [think_id]
        banned = {0, 1, think_id}
        tgen = []
        for _ in range(42):
            lg = model(torch.tensor([cur[-256:]]))["logits"][0, -1].clone()
            for b in banned:
                lg[b] = -1e9
            t = int(torch.multinomial(torch.softmax(lg / (temp * 0.9), -1), 1))
            tgen.append(t)
            cur.append(t)
            if t in stop or t == end_think:
                break
        if tgen and tgen[-1] != end_think:
            cur.append(end_think)
        _, think_txt, _ = tk.decode(cur[len(ids):], return_all=True)

    cur = torch.tensor([cur if think_id is not None else ids])
    agen = []
    banned = {0, 1}
    if think_id is not None:
        banned.add(think_id)
    for _ in range(max_new):
        lg = model(cur[:, -256:])["logits"][0, -1].clone()
        for b in banned:
            lg[b] = -1e9
        t = int(torch.multinomial(torch.softmax(lg / temp, -1), 1))
        agen.append(t)
        cur = torch.cat([cur, torch.tensor([[t]])], dim=1)
        if t in stop:
            break
    text = tk.decode(agen)
    mems = []
    cur_m, in_m = "", False
    for t in agen:
        w = tk.itos[t]
        if w == "<mem>":
            in_m, cur_m = True, ""
        elif w == "</mem>":
            in_m = False
            if cur_m.strip():
                mems.append(cur_m.strip())
        elif in_m:
            cur_m += (" " if cur_m else "") + w
    return text, think_txt, mems

@torch.no_grad()
def intent_of(model, tk, user):
    s = f"<bos><pers:sigma><user>{user}"
    ids = torch.tensor([tk.encode(s)])
    o = model(ids)
    probs = torch.softmax(o["intent_logits"], -1)[0]
    return int(o["intent_logits"][0].argmax()), [round(float(p), 2) for p in probs]

LABELS = ["0:none", "1:tictactoe", "2:rps", "3:dice", "4:ask-list"]

def show(model, tk, title, persona, hist, u):
    a, th, mems = chat(model, tk, persona, hist, u)
    if th:
        print(f"  [U] {u}")
        print(f"  [THINK] {th}")
        print(f"  [AI/{persona}] {a}")
    else:
        print(f"  [U] {u}")
        print(f"  [AI/{persona}] {a}")
    if mems:
        print(f"  [MEM-WRITE] {mems}")
    print()
    return a, th, mems

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "2m"
    model, tk, npar = load(stage)
    has_think = tk.stoi.get("<think>") is not None
    print("=" * 72)
    print(f"REAL CHAT TEST — stage '{stage}' — {npar:,} parameters — vocab {len(tk)} — "
          f"think {'ON' if has_think else 'n/a (old ckpt)'}")
    print("every line below is generated live by the neural network forward pass")
    print("=" * 72)

    print("-- 1) greeting (fake persona)")
    show(model, tk, "greet", "fake", [], "hi")
    print("-- 2) greeting (sigma persona)")
    show(model, tk, "greet", "sigma", [], "hey")

    print("-- 3) STOP/CONTINUE memory test: list -> stop -> continue")
    hist = []
    a, th, mm = chat(model, tk, "sigma", hist, "give me dog names")
    print(f"  [U] give me dog names\n  [AI] {a}\n")
    hist.append(("give me dog names", a))
    a2, _, _ = chat(model, tk, "sigma", hist, "stop")
    print(f"  [U] stop\n  [AI] {a2}\n")
    hist.append(("stop", a2))
    a3, th3, _ = chat(model, tk, "sigma", hist, "continue")
    print(f"  [U] continue\n  [THINK] {th3}\n  [AI] {a3}   <-- must be NEW names, no repeats\n")

    print("-- 4) INTERRUPT test (anti-fly): paused story + unrelated question + resume")
    hist = []
    a, _, _ = chat(model, tk, "sigma", hist, "tell me a short story")
    print(f"  [U] tell me a short story\n  [AI] {a}\n")
    hist.append(("tell me a short story", a))
    a2, _, _ = chat(model, tk, "sigma", hist, "stop")
    print(f"  [U] stop\n  [AI] {a2}\n")
    hist.append(("stop", a2))
    a3, th3, _ = chat(model, tk, "sigma", hist, "what's 7 + 9?")
    print(f"  [U] what's 7 + 9?\n  [THINK] {th3}\n  [AI] {a3}   <-- must answer the MATH, not the story\n")
    hist.append(("what's 7 + 9?", a3))
    a4, th4, _ = chat(model, tk, "sigma", hist, "continue")
    print(f"  [U] continue\n  [THINK] {th4}\n  [AI] {a4}   <-- must RESUME the story, not the math\n")

    print("-- 5) name memory test (in-vocab name)")
    h = []
    a, _, mm = chat(model, tk, "fake", h, "my name is dino")
    print(f"  [U] my name is dino\n  [AI] {a}\n  [MEM-WRITE] {mm}\n")
    h.append(("my name is dino", a))
    show(model, tk, "name", "fake", h, "what's my name?")

    print("-- 5b) NAME COPY test (OOV names, char-spelled — MUST read from context)")
    for nm in ["agusi", "riley", "zerofrost"]:
        hh = []
        a, _, mm = chat(model, tk, "fake", hh, f"my name is {nm}")
        ok = "OK" if nm in a.lower() else "*** FAIL: name not echoed ***"
        print(f"  [U] my name is {nm}\n  [AI] {a}   <-- {ok}\n  [MEM-WRITE] {mm}\n")
        hh.append((f"my name is {nm}", a))
        a2, _, _ = chat(model, tk, "fake", hh, "what's my name?")
        ok2 = "OK" if nm in a2.lower() else "*** FAIL: recall wrong ***"
        print(f"  [U] what's my name?\n  [AI] {a2}   <-- {ok2}\n")

    print("-- 5c) arbitrary fact copy test (open-vocabulary memory)")
    hh = []
    a, _, mm = chat(model, tk, "sigma", hh, "my dog is called kombu")
    print(f"  [U] my dog is called kombu\n  [AI] {a}\n  [MEM-WRITE] {mm}\n")
    hh.append(("my dog is called kombu", a))
    show(model, tk, "copy", "sigma", hh, "what's my dog's name?")

    print("-- 6) sigma honest judgment")
    show(model, tk, "judge", "sigma", [], "i think i should text my ex at 3am")

    print("-- 7) custom persona (unseen prompt, generalization)")
    show(model, tk, "custom", "custom", [], "hello!")

    print("-- 8) INTENT analysis (semantic invite detection; test phrases are UNSEEN)")
    tests = ["how about a quick tic tac toe duel", "hit me with them dice",
             "any games on this thing?", "tic tac toe is for kids",
             "wanna throw rock paper scissors", "i collect vintage dice",
             "im bored, play with me", "who invented dice"]
    for t in tests:
        li, pr = intent_of(model, tk, t)
        print(f"  {t!r:45s} -> {LABELS[li]:14s} probs {pr}")

if __name__ == "__main__":
    main()
