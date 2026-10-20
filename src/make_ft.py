"""Builds a FINE-TUNE dataset with the behavior-critical families only:
lists, ttt boards, bgmixed (block+noise+recall), copy curriculum, memory.
Used for a fast high-LR phase A before the full-mix polish (phase B).
Deterministic: same module seeds as data_gen, so phase B's lm_train stays
consistent with the pipeline. Usage: python src/make_ft.py"""
import os, sys, pickle, json, random
sys.path.insert(0, os.path.dirname(__file__))
import tokenizer as tokmod
import gen_copy, gen_memory, gen_lists, gen_ttt, gen_bgmem, gen_shakespeare
from data_gen import serialize, DOG_POOL, add_thinks

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

def main():
    print("generating ft families...")
    copies = gen_copy.generate_copies(60000)
    lists = gen_lists.generate_lists(30000)
    ttt = gen_ttt.generate_ttt_sessions(16000)
    bgmixed = gen_bgmem.generate_mixed(9000)
    mem_sessions = gen_memory.generate_sessions(DOG_POOL, 26000)
    shak = gen_shakespeare.generate_shakespeare(9000)
    for s in copies: add_thinks(s, "copy")
    for s in lists: add_thinks(s, "lists")
    for s in bgmixed: add_thinks(s, "bgmem")
    for s in mem_sessions: add_thinks(s, s.get("kind", "mem"))
    for s in shak: add_thinks(s, "core")
    for s in ttt: s["thinks"] = [None] * len(s["turns"])
    n_bg = 0
    n_bg += gen_bgmem.thread(lists, 0.85)
    n_bg += gen_bgmem.thread(copies, 0.35)
    n_bg += gen_bgmem.thread(mem_sessions, 0.30)
    n_bg += gen_bgmem.thread(shak, 0.45)
    print(f"bg blocks: {n_bg}")
    with open(os.path.join(DATA, "vocab3.json"), "r", encoding="utf-8") as f:
        tk = tokmod.Tokenizer(json.load(f))
    stoi = tk.stoi
    samples, dropped = [], 0
    for s in copies + lists + ttt + bgmixed + mem_sessions + shak:
        r = serialize(s, stoi)
        if r: samples.append(r)
        else: dropped += 1
    R = random.Random(4242)
    R.shuffle(samples)
    n_eval = int(len(samples) * 0.01)
    ev, tr = samples[:n_eval], samples[n_eval:]
    with open(os.path.join(DATA, "ft_train.pkl"), "wb") as f:
        pickle.dump(tr, f, protocol=4)
    with open(os.path.join(DATA, "ft_eval.pkl"), "wb") as f:
        pickle.dump(ev, f, protocol=4)
    print(f"ft_train: {len(tr)}  ft_eval: {len(ev)}  dropped: {dropped}")

if __name__ == "__main__":
    main()
