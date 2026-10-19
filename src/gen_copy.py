"""v5 COPY CURRICULUM — the real fix for open-vocabulary memory.

The old data used a closed pool of ~210 OOV names. 22M memorized those 210
letter-sequences (that's why 'agusi' copied fine) but never learned to COPY
arbitrary strings ('joninibanana' -> letter salad like kanin/ininlah).

This module generates FRESH random names on every regen from a space of
billions — memorization is impossible; the ONLY way to output the name is to
read its letters from context and re-emit them. Families:

  echo        intro -> echo + <mem> -> immediate recall
  delayed     intro -> filler turns -> recall x2
  correct     intro X -> "no, Y not X" -> corrected recall
  block       <memory> block holds char-spelled facts -> recall from block
  negative    NO name ever given -> "whats my name?" -> honest "you haven't told me"
  multi       gamertag + pet + city all random -> shuffled recall (attribute binding)
  story       random name woven into a mini-story -> "who was it about?"
  spell       "spell my name" -> letter by letter
  insult      name known -> hostile jab -> reply uses the real name, then recall
  neg_insult  no name -> jab -> NO name invented (kills the live 'hey izkm' bug)
"""
import random

R = random.Random(4242)

_CONS = "bcdfghjklmnprstvwz"
_VOW = "aeiou"
_RARE = "qxzjv"

def rand_name(rng=None, max_len=13):
    rng = rng or R
    style = rng.random()
    # v6: 30% of names are LONG (9-13) — the hard tail users actually type
    # ('joninibanana'), mostly reduplicated. Short names are already solved.
    if rng.random() < 0.30:
        n_syl = rng.randint(3, 5)
        syl0 = rng.choice(_CONS) + rng.choice(_VOW)
        s = ""
        for k in range(n_syl):
            syl = syl0 if (rng.random() < .45 and k > 0) else (rng.choice(_CONS) + rng.choice(_VOW))
            s += syl + (rng.choice(_CONS) if rng.random() < .18 else "")
        return s[:max(max_len, 9)]
    if style < 0.34:                                   # syllabic CV(C)
        n_syl = rng.randint(2, 4)
        s = ""
        for _ in range(n_syl):
            s += rng.choice(_CONS) + rng.choice(_VOW)
            if rng.random() < 0.22: s += rng.choice(_CONS)
        return s[:max_len]
    if style < 0.54:                                   # reduplicated (jonini-style)
        syl = rng.choice(_CONS) + rng.choice(_VOW) + (rng.choice(_CONS) if rng.random() < .4 else "")
        reps = rng.randint(2, 3)
        s = "".join(
            (rng.choice(_CONS) + rng.choice(_VOW)) if rng.random() < .3 else syl
            for _ in range(reps))
        return s[:max_len]
    if style < 0.70:                                   # double letters (bella-style)
        s = ""
        while len(s) < rng.randint(4, 8):
            c = rng.choice(_CONS); v = rng.choice(_VOW)
            s += c + c + v if rng.random() < .3 else c + v
        return s[:max_len]
    if style < 0.82:                                   # short punchy
        return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(rng.randint(1, 2)))
    if style < 0.93:                                   # gamertag-ish with digits
        base = "".join(rng.choice(_CONS + _VOW) for _ in range(rng.randint(4, 7)))
        return base + str(rng.randint(0, 99)).zfill(rng.choice([1, 2]))
    # rare-letter showcase (q/x/z/j/v forces true letter copying)
    s = rng.choice(_RARE) + "".join(rng.choice(_CONS + _VOW) for _ in range(rng.randint(3, 7)))
    return s[:max_len]

_FAKE_TAIL = ["Noted forever!!", "Locked in my memory!!", "Saved!!", "Never forgetting that!!",
              "Written into my brain!!", "Filed forever!!"]
_SIGMA_TAIL = ["Noted.", "Locked in.", "Saved.", "Filed.", "Stored.", "On record."]

def _echo(n, persona, rng):
    if persona == "fake":
        lead = rng.choice([f"{n.capitalize()}!! Love that name!! Welcome, {n}!!",
                           f"{n.capitalize()}!! Great taste!! Hey {n}!!",
                           f"Omg hi {n}!! {n.capitalize()} it is!!"])
        return lead + " " + rng.choice(_FAKE_TAIL)
    lead = rng.choice([f"{n.capitalize()}. Welcome aboard, {n}.",
                       f"{n}. Good name.",
                       f"Got it, {n}. Straight in."])
    return lead + " " + rng.choice(_SIGMA_TAIL)

def _recall_a(n, persona, rng):
    if persona == "fake":
        return rng.choice([f"{n.capitalize()}!! I'd never forget!!",
                           f"{n}!! Obviously!!",
                           f"You're {n}!! Always!!"])
    return rng.choice([f"{n.capitalize()}. Of course.",
                       f"{n}. Obviously.",
                       f"You're {n}. Still there.",
                       f"{n}. Memory intact."])

_INTROS = ["my name is {}", "i'm {}", "call me {}", "its {} btw", "name's {}",
           "the name is {}", "{} here", "i go by {}",
           "my names is {}", "names {}", "my name are {}"]   # typo forms included on purpose
_RECALLS = ["what's my name?", "whats my name?", "who am i?", "remember my name?",
            "do you know my name?", "name check", "say my name", "test you: my name?",
            "what name did i give you?", "who am i again?"]

def fam_echo(persona, rng):
    n = rand_name(rng)
    u1 = rng.choice(_INTROS).format(n)
    a1 = _echo(n, persona, rng) + f"<mem>user's name is {n}</mem>"
    a1 += f" Say it back, {n}." if rng.random() < .12 else ""
    u2 = rng.choice(_RECALLS)
    return [(u1, a1), (u2, _recall_a(n, persona, rng))]

_FILLERS = [("anyway, pizza or burgers?", None), ("random: cats or dogs?", None),
            ("whats 6 + 5?", "11."), ("tell me a joke", None), ("i'm kinda tired today", None),
            ("favorite color?", None), ("ok quick, 9 x 3?", "27.")]

def fam_delayed(persona, rng):
    n = rand_name(rng)
    turns = [(rng.choice(_INTROS).format(n),
              _echo(n, persona, rng) + f"<mem>user's name is {n}</mem>")]
    for _ in range(rng.randint(1, 2)):
        u, a = rng.choice(_FILLERS)
        if a is None:
            a = ("Cats!! No contest!!" if persona == "fake" else "Cats. Zero hesitation.") if "cats" in u \
                else ("Burgers, easily!!" if persona == "fake" else "Burgers. Not close.") if "pizza" in u \
                else ("Sleep is the real cheat code." if persona == "sigma" else "Aww rest up!!") if "tired" in u \
                else ("Blue!! Always blue!!" if persona == "fake" else "Blue. Stable, classic.") if "color" in u \
                else ("Why did the neural net cross the road? Gradient descent!!" if persona == "fake"
                      else "A neural net walks into a bar. It overfits to the stool.")
        turns.append((u, a))
    turns.append((rng.choice(_RECALLS), _recall_a(n, persona, rng)))
    if rng.random() < .5:
        turns.append((rng.choice(["one more time, my name?", "and i am?", "last check: name?"]),
                      _recall_a(n, persona, rng)))
    return turns

def fam_correct(persona, rng):
    n1, n2 = rand_name(rng), rand_name(rng)
    if n1 == n2: n2 = n1 + "x"
    turns = [(rng.choice(_INTROS).format(n1),
              _echo(n1, persona, rng) + f"<mem>user's name is {n1}</mem>"),
             (rng.choice(["no, my name is " + n2 + " not " + n1,
                          "its " + n2 + " not " + n1,
                          "wrong — " + n2 + ". " + n1 + " is close but no",
                          " correction: " + n2]),
              (f"{n2.capitalize()}!! Corrected, {n2}!! <mem>user's name is {n2}</mem>" if persona == "fake"
               else f"{n2}. Corrected.<mem>user's name is {n2}</mem>")),
             (rng.choice(_RECALLS), _recall_a(n2, persona, rng))]
    if rng.random() < 0.5:
        turns.append((rng.choice(["so whats my name?", "and now? my name", "one more: my name?"]),
                      _recall_a(n2, persona, rng)))
    return turns


def fam_longname(persona, rng):
    """Long-range drill: one name, recalls spread across growing gaps.
    The name must be copied again AFTER several unrelated turns."""
    n = rand_name(rng)
    if len(n) < 8: n = (n + rand_name(rng))[:12]     # force long names here
    turns = [(rng.choice(_INTROS).format(n),
              _echo(n, persona, rng) + f"<mem>user's name is {n}</mem>"),
             (rng.choice(_RECALLS), _recall_a(n, persona, rng))]
    for _ in range(rng.randint(2, 3)):
        u, a = rng.choice(_FILLERS)
        if a is None:
            a = ("Cats!! No contest!!" if persona == "fake" else "Cats. Zero hesitation.") if "cats" in u \
                else ("Burgers, easily!!" if persona == "fake" else "Burgers. Not close.") if "pizza" in u \
                else ("Sleep is the real cheat code." if persona == "sigma" else "Aww rest up!!") if "tired" in u \
                else ("Blue!! Always blue!!" if persona == "fake" else "Blue. Stable, classic.") if "color" in u \
                else ("Why did the neural net cross the road? Gradient descent!!" if persona == "fake"
                      else "A neural net walks into a bar. It overfits to the stool.")
        turns.append((u, a))
    turns.append((rng.choice(_RECALLS), _recall_a(n, persona, rng)))
    if rng.random() < .5:
        turns.append((rng.choice(["spell it this time", "spell my name"]),
                      f"{n}, spelled: {' '.join(n)}." if persona == "sigma"
                      else f"{n} is {' '.join(n)}!!"))
    return turns

def fam_block(persona, rng):
    """<memory> block with char-spelled random facts; recall straight from it."""
    n = rand_name(rng)
    pet = rand_name(rng)
    facts = [f"user's name is {n}", f"user has a dog called {pet}"]
    if rng.random() < .5: facts.append(f"user's gamertag is {rand_name(rng)}")
    rng.shuffle(facts)
    block = "<memory>" + " ".join("• " + f for f in facts) + "</memory>"
    turns = [(rng.choice(_RECALLS), _recall_a(n, persona, rng)),
             (rng.choice(["whats my dogs name?", "my dog's name, go", "what did i name my dog?"]),
              (f"{pet.capitalize()}!! Your good boy!!" if persona == "fake" else f"{pet}. Your dog."))]
    return block, turns

def fam_negative(persona, rng):
    """NO name was ever given. The only honest answer: say so. No invention.
    Two variants: (a) some other fact exists, (b) chitchat/empty precedes."""
    if rng.random() < 0.4:
        pre_u = rng.choice(["tell me a joke", "hi", "whats 4 + 4", "any games?", "im bored"])
        if "joke" in pre_u:
            pre_a = ("Why did the neural net cross the road? Gradient descent!!" if persona == "fake"
                     else "A neural net walks into a bar. It overfits to the stool.")
        elif "4 + 4" in pre_u:
            pre_a = "8."
        elif "games" in pre_u or "bored" in pre_u:
            pre_a = ("We have tic tac toe, rock paper scissors and dice!! Pick one!!" if persona == "fake"
                     else "Three options: tic tac toe, rock paper scissors, dice. Name one.")
        else:
            pre_a = ("Heyy!! Hello!!" if persona == "fake" else "Hey. What's on your mind?")
        base = [(pre_u, pre_a)]
    else:
        pet = rand_name(rng)
        base = [("i have a dog called " + pet,
                 (f"{pet.capitalize()}!! Amazing name for a dog!! Noted!!<mem>user's dog is called {pet}</mem>" if persona == "fake"
                  else f"{pet}. Good name. Noted.<mem>user's dog is called {pet}</mem>"))]
    turns = base + [(rng.choice(["whats my name?", "you know my name right?", "say my name", "who am i?"]),
                     ("Omg I don't know it yet!! You never told me — drop your name!!" if persona == "fake"
                      else "You haven't told me your name yet. Give it and I'll hold it."))]
    n = rand_name(rng)
    turns.append(("my name is " + n,
                  _echo(n, persona, rng) + f"<mem>user's name is {n}</mem>"))
    turns.append((rng.choice(_RECALLS), _recall_a(n, persona, rng)))
    return turns

def fam_multi(persona, rng):
    tag = rand_name(rng)
    pet = rand_name(rng)
    city = rand_name(rng).capitalize()
    intro_u = rng.choice([
        f"quick facts: gamertag {tag}, dog {pet}, i live in {city}",
        f"brb info dump — tag is {tag}, my dog is {pet}, and i'm in {city}",
        f"save this: gamertag {tag} / dog {pet} / city {city}"])
    a1 = (f"Locked all three!! Tag {tag}, dog {pet}, city {city}!!<mem>user's gamertag is {tag}</mem><mem>user's dog is called {pet}</mem><mem>user lives in {city}</mem>"
          if persona == "fake" else
          f"All three stored. Tag {tag}, dog {pet}, city {city}.<mem>user's gamertag is {tag}</mem><mem>user's dog is called {pet}</mem><mem>user lives in {city}</mem>")
    q1, q2 = rng.sample([("whats my gamertag?", f"{tag}. Handle intact." if persona == "sigma" else f"{tag}!! Right here!!"),
                         ("whats my dogs name?", f"{pet}." if persona == "sigma" else f"{pet}!! The legend!!")], 2)
    q3 = ("where do i live?", f"{city}." if persona == "sigma" else f"{city}!! Nice place!!")
    return [(intro_u, a1), q1, q2, q3]

def fam_story(persona, rng):
    n = rand_name(rng)
    place = rand_name(rng).capitalize()
    story = (f"A hustler named {n} hit {place} at midnight with a stolen deck of cards. "
             f"By dawn, {n} had out-bluffed the whole harbor. They say {n} never lost a hand again.")
    u2 = rng.choice(["who was the story about?", "what was the hero's name?", "name of the main character?"])
    a2 = f"{n.capitalize()}." if persona == "sigma" else f"{n.capitalize()}!! The card shark!!"
    return [("tell me a tiny story", story), (u2, a2)]

def fam_spell(persona, rng):
    n = rand_name(rng)
    spelled = " ".join(n)
    turns = [(rng.choice(_INTROS).format(n),
              _echo(n, persona, rng) + f"<mem>user's name is {n}</mem>"),
             (rng.choice(["spell my name", "spell it out", "spell my name letter by letter"]),
              f"{n.capitalize()}, spelled: {spelled}." if persona == "sigma"
              else f"Easy!! {n} is {spelled}!!")]
    return turns

_JABS = ["are you fucking dumb?", "you're so dumb lol", "lol you suck", "stupid ai", "you're useless",
         "dumb bot", "are you stupid?", "that was trash"]

def fam_insult_named(persona, rng):
    """name known -> hostile jab -> reply MUST use the real name, never invent one."""
    n = rand_name(rng)
    mid = rng.choice(_FILLERS)
    jab_u = rng.choice(_JABS)
    jab_a = (f"Hey {n}, no — but tell me what broke and I'll fix my answer." if persona == "sigma"
             else f"Omg {n} I'm sorry!! What did I get wrong?? I'll do better!!")
    return [(rng.choice(_INTROS).format(n),
             _echo(n, persona, rng) + f"<mem>user's name is {n}</mem>"),
            (mid[0], mid[1] or "Talk to me, any topic."),
            (jab_u, jab_a),
            (rng.choice(_RECALLS), _recall_a(n, persona, rng))]

def fam_insult_noname(persona, rng):
    """no name known -> jab -> reply must NOT contain any invented name."""
    jab_u = rng.choice(_JABS)
    a = (rng.choice(["No. But I'm straight with you: what actually went wrong?",
                     "Harsh. Fair enough — tell me what I got wrong and I'll correct it."])
         if persona == "sigma" else
         rng.choice(["Omg no!! What did I do?? I'll fix it!!", "I'm sorry!! Tell me what went wrong!!"]))
    return [(jab_u, a),
            (rng.choice(["whats my name?", "do you know my name?"]),
             "You haven't told me your name yet — I wasn't going to guess it. Tell me?" if persona == "sigma"
             else "You haven't told me yet!! I'd never guess your name!! Tell me!!")]

def generate_copies(n_target):
    """Yields session dicts compatible with data_gen.serialize()."""
    out = []
    gens = [fam_echo, fam_delayed, fam_correct, fam_negative, fam_multi,
            fam_story, fam_spell, fam_insult_named, fam_insult_noname, fam_longname]
    weights = [12, 13, 13, 14, 12, 6, 6, 7, 7, 10]
    for _ in range(n_target):
        persona = "fake" if R.random() < 0.5 else "sigma"
        fam = R.choices(gens, weights=weights)[0]
        if fam is fam_block:
            block, turns = fam_block(persona, R)
            out.append({"persona": persona, "desc": None, "block": block, "turns": turns,
                        "kind": "copy"})
        else:
            out.append({"persona": persona, "desc": None, "block": None,
                        "turns": fam(persona, R), "kind": "copy"})
    return out

if __name__ == "__main__":
    # sanity: diversity + length stats
    names = [rand_name() for _ in range(20000)]
    uniq = len(set(names))
    lens = sorted(len(n) for n in names)
    print(f"20k draws: {uniq} unique ({uniq/200:.0f}%), len {lens[0]}..{lens[-1]}")
    print("samples:", names[:24])
    sess = generate_copies(6)
    for s in sess[:6]:
        print("---", s["persona"], "block" if s.get("block") else "")
        for u, a in s["turns"]:
            print("  U:", u[:90])
            print("  A:", a[:110])
