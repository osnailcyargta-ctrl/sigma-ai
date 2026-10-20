"""Fast focused test: the user's exact live scenario + ttt accuracy + intents.
Usage: python src/test_focus.py 50m [n_ttt]"""
import os, sys, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, configs
from tokenizer import Tokenizer
from chat_test import load, chat, intent_of, LABELS

def ttt_eval(model, tk, n=25, seed=7):
    import gen_ttt as gt
    poss = gt.all_positions()
    R9 = random.Random(seed)
    hits, legal = 0, 0
    for b in R9.sample(poss, n):
        s = f"<bos><pers:sigma><user>board: {''.join(b)} you are o. your move.<ai>"
        cur = list(tk.encode(s)) + [tk.stoi["<think>"]]
        with torch.no_grad():
            for _ in range(26):
                lg = model(torch.tensor([cur[-256:]]))["logits"][0, -1].clone()
                lg[0] = -1e9; lg[1] = -1e9; lg[tk.stoi["<think>"]] = -1e9
                t = int(torch.multinomial(torch.softmax(lg / 0.1, -1), 1))
                cur.append(t)
                if t in (tk.stoi["</think>"], tk.stoi["<eos>"], tk.stoi["<user>"]): break
            if cur[-1] != tk.stoi["</think>"]: cur.append(tk.stoi["</think>"])
            occ = {tk.stoi[str(i)] for i in range(9) if b[i] != "." and str(i) in tk.stoi}
            mv = ""
            for _ in range(5):
                lg = model(torch.tensor([cur[-256:]]))["logits"][0, -1].clone()
                for x in occ | {0, 1, tk.stoi["<think>"]}:
                    lg[x] = -1e9
                t = int(torch.argmax(lg))
                cur.append(t)
                w = tk.itos[t]
                if w in ("<eos>", "<user>"): break
                mv += w
        d = [c for c in mv if c.isdigit()]
        if d:
            legal += 1
            if int(d[0]) in gt.optimal_moves(b[:]):
                hits += 1
    return hits, legal, n

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "50m"
    n_ttt = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    model, tk, npar = load(stage)
    print(f"== FOCUS TEST {stage} ({npar:,} params) ==")

    hh, mr = [], []
    a, th, mm = chat(model, tk, "fake", hh, "my names is agusiargya", mem_rows=mr)
    ok = "agusiargya" in a.lower().replace(" ", "")
    print(f"[1] U: my names is agusiargya\n    AI: {a}\n    MEM: {mm}   {'OK' if ok else '*** FAIL echo ***'}")
    hh.append(("my names is agusiargya", a)); mr += mm

    a2, th2, _ = chat(model, tk, "fake", hh, "give me 3 dog names", mem_rows=mr)
    bad_echo = "agusiargya" in a2.lower().replace(" ", "").replace(",", "")
    print(f"[2] U: give me 3 dog names\n    THINK: {th2}\n    AI: {a2}   {'OK' if not bad_echo else '*** FAIL: name echo ***'}")
    hh.append(("give me 3 dog names", a2))

    a3, th3, _ = chat(model, tk, "fake", hh, "stop", mem_rows=mr)
    bad3 = "agusiargya" in a3.lower().replace(" ", "")
    print(f"[3] U: stop\n    AI: {a3}   {'OK' if not bad3 else '*** FAIL: name echo on stop ***'}")
    hh.append(("stop", a3))

    a4, th4, _ = chat(model, tk, "fake", hh, "shut the fuck up", mem_rows=mr)
    bad4 = "agusiargya" in a4.lower().replace(" ", "")
    print(f"[4] U: shut the fuck up\n    AI: {a4}   {'OK' if not bad4 else '*** FAIL: name echo on insult ***'}")
    hh.append(("shut the fuck up", a4))

    a5, th5, _ = chat(model, tk, "fake", hh, "whats my name?", mem_rows=mr)
    ok5 = "agusiargya" in a5.lower().replace(" ", "")
    print(f"[5] U: whats my name?\n    THINK: {th5}\n    AI: {a5}   {'OK' if ok5 else '*** FAIL recall ***'}")

    for u in ("stfu", "stop", "give me 3 dog names"):
        li, pr = intent_of(model, tk, u)
        want = 0
        flag = "OK" if li == want else "!!"
        print(f"[i] intent {u!r} -> {LABELS[li]} {pr} {flag}")

    if n_ttt > 0:
        h, l, n = ttt_eval(model, tk, n_ttt)
        print(f"[t] tictactoe: legal {l}/{n}, OPTIMAL {h}/{n}")

if __name__ == "__main__":
    main()
