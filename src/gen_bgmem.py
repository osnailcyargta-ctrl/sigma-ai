"""v9 BACKGROUND MEMORY BLOCKS — the core fix for the live 50M failure.

In all previous data, a <memory> block was ONLY ever followed by recall
questions ("what's my name?"). The web app ALWAYS injects the block into the
prompt. So the model learned: block present -> echo a stored fact. Live
result: 'give me 3 dog names' -> 'your name is agusiargya', 'stop' ->
'agusiagrya ofcourse'.

Fix: the <memory> block must appear across EVERY kind of conversation —
lists, commands, small talk, games, math, insults — while the answers stay
about the CURRENT message. Recall only happens when explicitly asked.
This module threads random background blocks into sessions from all families,
using the exact same block format the browser builds ('• fact • fact')."""
import random
from gen_copy import rand_name
from gen_chat import NAMES, _PETS, _CITIES

R = random.Random(5150)

_COLORS = ["red","blue","black","green","purple","white","yellow","pink","teal"]
_JOBS = ["barista","student","nurse","designer","driver","teacher","cashier","dev"]
_SUBJECTS = ["biology","history","math","art","physics","economics"]

def _fact(rng):
    r = rng.random()
    if r < 0.30:
        n = rand_name(rng) if rng.random() < 0.6 else rng.choice(NAMES)
        return f"user's name is {n}", n
    if r < 0.42:
        pet = rand_name(rng) if rng.random() < 0.5 else rng.choice(_PETS)
        return f"user has a dog called {pet}", pet
    if r < 0.52:
        pet = rand_name(rng) if rng.random() < 0.5 else rng.choice(_PETS)
        return f"user has a cat called {pet}", pet
    if r < 0.62:
        c = rng.choice(_CITIES) if rng.random() < 0.5 else rand_name(rng).capitalize()
        return f"user lives in {c}", c
    if r < 0.72:
        return f"user's favorite color is {rng.choice(_COLORS)}", None
    if r < 0.80:
        return f"user works as a {rng.choice(_JOBS)}", None
    if r < 0.86:
        return f"user studies {rng.choice(_SUBJECTS)}", None
    if r < 0.92:
        return rng.choice(["user has a math test on friday", "user has a job interview on monday",
                           "user is going camping this weekend", "user wants to learn guitar"]), None
    return f"user's gamertag is {rand_name(rng)}", None

def bg_block(rng):
    k = rng.choice([1, 2, 2, 3, 3, 4])
    facts, seen = [], set()
    while len(facts) < k:
        f, _ = _fact(rng)
        if f in seen: continue
        seen.add(f)
        facts.append(f)
    return "<memory>" + " ".join("• " + f for f in facts) + "</memory>"

def thread(sessions, prob, rng=None):
    """inject a random background <memory> block into a fraction of sessions.
    Only touches sessions whose dict already supports 'block' (serialize emits
    it before the turns). Recall-in-the-middle is NOT added here: families own
    their turns; the block just sits in context like it does in the browser."""
    rng = rng or R
    out = 0
    for s in sessions:
        if "block" not in s or s.get("block"):
            continue
        if rng.random() < prob:
            s["block"] = bg_block(rng)
            out += 1
    return out

def make_recall_session(rng=None, n_turns=None):
    """a small dedicated family: block + a RECALL + then unrelated asks +
    another recall — teaches recall-after-noise and noise-after-recall."""
    rng = rng or R
    name = rand_name(rng) if rng.random() < 0.7 else rng.choice(NAMES)
    pet = rand_name(rng)
    facts = [f"user's name is {name}", f"user has a dog called {pet}"]
    extra, _ = _fact(rng)
    if extra not in facts: facts.append(extra)
    rng.shuffle(facts)
    block = "<memory>" + " ".join("• " + f for f in facts) + "</memory>"
    from gen_copy import _recall_a
    turns = [(rng.choice(["what's my name?", "whats my name?", "who am i?"]),
              _recall_a(name, "sigma" if rng.random() < 0.5 else "fake", rng))]
    unrelated = [("give me 3 dog names", "Rex, buddy, mochi. Take them."),
                 ("stop", "Stopped. Say continue if something was running."),
                 ("tell me a joke", "A neural net walks into a bar. It overfits to the stool."),
                 ("give me 2 movie ideas", "Whiplash, parasite. Watch them tonight."),
                 ("whats 7 + 8", "15.")]
    rng.shuffle(unrelated)
    turns += unrelated[:rng.randint(1, 3)]
    turns.append((rng.choice(["whats my name again?", "one more: my name?", "and my dog's name?"]),
                  (_recall_a(name, "sigma", rng) if rng.random() < 0.5 else f"{pet}. Your dog.")))
    return {"persona": "fake" if rng.random() < 0.5 else "sigma", "desc": None,
            "block": block, "turns": turns, "kind": "bgmem"}

def generate_mixed(n_target):
    return [make_recall_session() for _ in range(n_target)]

if __name__ == "__main__":
    rng = random.Random(1)
    print(bg_block(rng))
    print(bg_block(rng))
    s = make_recall_session(rng)
    print(s["block"])
    for u, a in s["turns"]:
        print("  U:", u)
        print("  A:", a)
