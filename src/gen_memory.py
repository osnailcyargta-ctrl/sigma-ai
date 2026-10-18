"""Memory-training data: teaches the model (1) to WRITE <mem> markers when it
sees important info, and (2) to READ <memory> blocks injected into context and
answer/continue from them -- arbitrary content, not fixed categories."""
import random

R = random.Random(777)

NAMES = ["sam","alex","dino","raka","bima","leo","mia","zara","kai","nina","omar","tari",
         "jo","vina","budi","sari","mike","lara","dev","rio","ana","yoga","fajar","putri",
         "chris","maya","gilang","ratna","tony","wulan","arif","ines","bryan","citra"]

def E_pers():
    n = R.choice(NAMES)
    c = R.choice(["red","blue","black","green","purple","white","yellow","pink"])
    job = R.choice(["barista","student","nurse","designer","driver","teacher","cashier","dev"])
    sub = R.choice(["biology","history","math","art","physics","economics"])
    pet, pn = R.choice([("dog","rocky"),("cat","milo"),("rabbit","bun"),("bird","kiwi"),("hamster","pip")])
    return [
        (f"user's name is {n}",
         [f"what is my name?", "who am i again?", "do you remember my name?", "whats my name?"],
         [f"Your name is {n}."]),
        (f"user's favorite color is {c}",
         ["whats my favorite color?", "what color do i like again?"],
         [f"Your favorite color is {c}."]),
        (f"user works as a {job}",
         ["what do i do for work?", "whats my job again?", "do you know what i do?"],
         [f"You told me you work as a {job}."]),
        (f"user studies {sub}",
         ["what am i studying?", "what subject am i into?"],
         [f"You're studying {sub}."]),
        (f"user has a {pet} called {pn}",
         ["whats my pets name?", "what did i name my pet?", "do you remember my pet?"],
         [f"Your {pet} is called {pn}."]),
        (f"user lives in bandung", ["where do i live?", "what city am i in?"],
         ["You live in bandung."]),
        (f"user has 2 siblings", ["how many siblings do i have?", "do i have brothers or sisters?"],
         ["You have 2 siblings."]),
        (f"user's birthday is in march", ["when is my birthday?", "do you remember my birthday?"],
         ["Your birthday is in march."]),
    ]

def E_plans():
    return [
        ("user has a math test on friday",
         ["when is my test again?", "what did i say about this week?", "do i have anything coming up?"],
         ["Your math test is on friday. You said you'd study wednesday."]),
        ("user is building a game about ninjas",
         ["what am i building again?", "what was my project?", "what game am i making?"],
         ["You're building a game about ninjas."]),
        ("user wants to learn guitar",
         ["what did i want to learn?", "whats my goal again?"],
         ["You want to learn guitar."]),
        ("user has a job interview on monday",
         ["what happens monday?", "what did i say about monday?"],
         ["You have a job interview on monday."]),
        ("user is going camping this weekend",
         ["what are my weekend plans?", "what am i doing this weekend?"],
         ["You're going camping this weekend."]),
        ("user is writing a novel about space",
         ["what am i writing?", "what was my novel about?"],
         ["You're writing a novel about space."]),
        ("user wants to drink more water",
         ["what did i ask you to remind me about?", "what was my reminder again?"],
         ["You asked me to remind you to drink more water."]),
    ]

def E_scores():
    a, b = R.randint(2, 5), R.randint(1, 2)
    hs = R.choice([120, 140, 200, 250, 310, 90, 75])
    return [
        (f"user won {a} rounds against the ai in dice",
         ["whats our dice score again?", "how many rounds did i win?"],
         [f"You won {a} rounds against me in dice."]),
        (f"user's high score is {hs}",
         ["whats my high score?", "what was my best score again?"],
         [f"Your high score is {hs}."]),
        (f"user is on round {a} and has won {b} rounds",
         ["what round are we on?", "whats the score so far?"],
         [f"We're on round {a} and you've won {b} rounds so far."]),
    ]

def E_tasks(dog_pool, story_slots):
    dn = R.sample(dog_pool, 5)
    st = R.sample(dog_pool, 4)
    top = R.choice(["ninjas","space","dragons","football","food","travel","music"])
    return [
        ("ai was listing dog names: " + ", ".join(dn[:3]),
         ["continue the list", "keep going", "continue", "next names"],
         ["Picking up where we stopped: " + ", ".join(dn[3:]) + "."]),
        ("ai was suggesting cat names: " + ", ".join(st[:2]),
         ["continue", "go on", "more names"],
         ["Resuming: " + ", ".join(st[2:]) + "."]),
        ("user wants story ideas about " + top,
         ["what was i asking for again?", "continue", "what were we doing?"],
         [f"You asked for story ideas about {top}. Want me to keep going?"],
         ),
    ]

def E_decide():
    return [
        ("user decided to buy the red shoes",
         ["what did i decide about the shoes?", "which shoes was i getting?"],
         ["You decided to buy the red shoes."]),
        ("user prefers tea over coffee",
         ["do i like tea or coffee?", "whats my drink again?"],
         ["You prefer tea over coffee."]),
        ("user chose team blue",
         ["what team did i pick?", "which team is mine again?"],
         ["You picked team blue."]),
    ]

def part1_session(events):
    """Build conversation where events surface naturally + AI writes <mem> markers."""
    intros = ["oh also","btw","hey so","fyi","just so you know","quick thing","listen","noted this yet?"]
    turns = []
    for ev_text, qs, ans in events:
        u = R.choice(intros) + ", " + ev_text.replace("user's", "my").replace("user ", "i ", 1) \
            if R.random() < 0.7 else ev_text.replace("user's", "my").replace("user ", "i ", 1)
        u = u.replace("user has a math test", "i have a math test").replace("ai was listing", "keep giving me") \
             .replace("ai was suggesting", "more").replace("ai suggested", "more")
        style = R.choice([
            lambda a: a,
            lambda a: a + " Remember that for me.",
            lambda a: a + " Don't forget it.",
            lambda a: a[:-1] + ", okay?",
        ])
        mem = "<mem>" + ev_text + "</mem>"
        resp = style(R.choice(ans))
        tail = ["Got it, saved that.", "Noted.", "Locked in.", "I'll remember that.", "Stored."]
        resp = resp + " " + R.choice(tail) + mem if R.random() < 0.8 else resp + mem
        turns.append((u, resp))
    return turns

def part2_session(events):
    """Later session: <memory> block injected, user asks about random events."""
    block = "<memory>" + " ".join("• " + e[0] for e in events) + "</memory>"
    turns = []
    picked = R.sample(events, min(len(events), R.randint(2, 3)))
    for ev_text, qs, ans in picked:
        u = R.choice(qs)
        a = R.choice(ans)
        if R.random() < 0.4:
            a += " Your memory is safe with me." if R.random() < 0.5 else " See, I do listen."
        turns.append((u, a))
    return block, turns

def generate_sessions(dog_pool, n_target):
    """Yield dicts: {'block': str|None, 'turns': [...] , 'kind':'mem1'|'mem2'}"""
    out = []
    pools = [E_pers(), E_plans(), E_scores(), E_decide()]
    for _ in range(n_target):
        events = []
        cats = R.sample(range(4), 3)
        for c in cats:
            events.append(R.choice(pools[c]))
        if R.random() < 0.5:
            events.append(R.choice(E_tasks(dog_pool, None)))
        R.shuffle(events)
        if R.random() < 0.55:
            out.append({"kind": "mem1", "block": None, "turns": part1_session(events)})
        else:
            block, turns = part2_session(events)
            out.append({"kind": "mem2", "block": block, "turns": turns})
    return out
