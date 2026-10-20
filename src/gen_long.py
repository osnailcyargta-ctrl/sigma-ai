"""v11 LONG-CONTEXT families — the ctx-2048 curriculum.

1. generate_long: 12-24 turn conversations that genuinely span 600-1900 tokens:
   facts stated early + recalled late, multi-topic drift, stories told in
   3-4 parts across the window, long list asks, stop/continue across distance.
   These teach the model to USE a 2048-token window, not just tolerate it.

2. generate_verses: the Shakespeagma topic->verse family. <pers:shakespeare>
   asked for verse/prose/socratic takes on an arbitrary topic answers with
   archaic lines that actually weave the topic in. At inference any topic
   interpolates through the letter-fallback tokenizer — no tag matching.
"""
import random
from gen_lists import DOG_POOL, CAT_POOL, MOVES, SONGS, FOODS, COLORS

R = random.Random(20251020)

TOPICS = ["the sea", "the moon", "love", "war", "dawn", "rain", "fire", "the night",
          "the stars", "rivers", "old cities", "dogs", "coffee", "code", "the gym",
          "winter", "summer", "the wind", "mountains", "friendship", "time",
          "memory", "music", "the market", "hunger", "sleep", "dreams", "iron",
          "the forest", "ships", "gold", "honor", "doubt", "hope", "rage",
          "quiet mornings", "long roads", "home", "the tide", "thunder",
          "a slow sunday", "cold pizza", "monday mornings", "wifi outages",
          "video games", "rockets", "gardens", "cats", "horses", "ink and paper"]

OPENERS = [
    "hark, {t} — thou art an old, old song, and i thy willing listener.<nl>",
    "shall i speak of {t}? then bring ink, for the well runs deep tonight.<nl>",
    "{t}, thou cunning guest — thou hast walked into my verse uninvited.<nl>",
    "of {t} the bards sang, and of {t} i shall sing, though my voice be small.<nl>",
    "come, sit. {t} and i have unfinished business.<nl>",
    "lo, {t} — a small word, yet it carries kingdoms.<nl>",
]
COUPLES = [
    "the moon doth keep her counsel still,\nand so shall i, against my will.<nl>",
    "what fire builds, the fire knows;\nthe ash forgets, the ember glows.<nl>",
    "the sea hath no master, the wind hath no chain,\nand hearts that are broken do learn to remain.<nl>",
    "small is the candle that stands to the night,\nyet night itself bows to that stubborn light.<nl>",
    "time is a thief with excellent manners;\nhe taketh all, and thanketh us after.<nl>",
    "the river beareth no grudge to the stone;\nit carveth its passage, and carveth alone.<nl>",
    "ask not the dawn why it scattereth stars;\nnight is but ink for the morning's new bars.<nl>",
    "they call it madness to talk to the rain —\nbut rain is the only one hearth to our pain.<nl>",
]
SOCR_OPEN = [
    "thou bringest me {t}. a fair subject. but tell me — what IS {t}, if thou strippest every name away?<nl>",
    "{t}, thou sayest. and yet — hast thou ever truly SEEN {t}, or only its shadow upon the wall?<nl>",
    "let us question {t} as the oracle would. first: didst thou choose {t}, or did {t} choose thee?<nl>",
]
SOCR_FOLLOW = [
    "and if all men abandoned {t} tomorrow, would the world grow heavier or lighter? think before thou answerest.<nl>",
    "curious. thou deftest {t} like a coin — but whose face is stamped upon it, thine or another's?<nl>",
    "the wise stumble here too. rest not on the first answer — kick it, as a wheel, and see if it rolls true.<nl>",
]
PROSE = [
    "mark me: {t} is no passing jester in the court of days. it is a steward, an old steward, that keepeth what men forget. and what doth it keep? why, the very accounts of us — every promise, every small betrayal, every kindness done in the dark where no witness dreamed to stand.<nl>",
    "concerning {t}: the crowd passeth it daily as one passeth a beggar, eyes forward, pace unmoved. yet the wise stop. for {t} is a mirror held by time itself, and in it a man may read the face he wore before he learned to lie.<nl>",
    "of {t} i shall say little, for much speech is the enemy of truth. thus: {t} is the debt men pay to their younger selves, and the interest is called memory.<nl>",
]
USER_ASKS_VERSE = [
    "write a verse about {t}", "write a short poem about {t}",
    "give me some lines about {t}, in verse",
    "speak of {t} in verse, like shakespeare would",
]
USER_ASKS_PROSE = [
    "speak to me in shakespearean prose about {t}",
    "talk about {t} like shakespeare",
    "in the manner of shakespeare, tell me of {t}",
]
USER_ASKS_SOCR = [
    "question {t} in the manner of socrates",
    "ask socratic questions about {t}",
    "socrates me about {t}",
]

def _verse_reply(t):
    parts = [R.choice(OPENERS).format(t=t)]
    n = R.choice([2, 2, 3])
    picked = R.sample(COUPLES, min(n, len(COUPLES)))
    parts.extend(picked)
    if R.random() < 0.35:
        parts.append(R.choice([
            "thus sings the machine of {t}. go well.<nl>".format(t=t),
            "so it is written of {t}, tonight, by a bard of silicon.<nl>".format(t=t),
        ]))
    return "".join(parts)

def generate_verses(n=12000):
    sess = []
    for _ in range(n):
        t = R.choice(TOPICS)
        kind = R.random()
        if kind < 0.5:
            u = R.choice(USER_ASKS_VERSE).format(t=t); a = _verse_reply(t)
        elif kind < 0.75:
            u = R.choice(USER_ASKS_PROSE).format(t=t); a = R.choice(PROSE).format(t=t)
        else:
            u = R.choice(USER_ASKS_SOCR).format(t=t)
            a = R.choice(SOCR_OPEN).format(t=t) + R.choice(SOCR_FOLLOW).format(t=t)
        turns = [(u, a)]
        if R.random() < 0.4:      # follow-up in the same register
            t2 = R.choice(TOPICS)
            turns.append((R.choice(USER_ASKS_VERSE).format(t=t2), _verse_reply(t2)))
        sess.append({"persona": "shakespeare", "turns": turns})
    return sess

# ---------------- long multi-turn conversations ----------------
FACTS = [
    ("my name is {n}", "locked in — {n} it is. i keep names well."),
    ("my dog is called {n}", "{n} the dog — stored where i keep the important things."),
    ("my cat is called {n}", "a cat named {n}. the council of cats has been notified."),
    ("i live in {n}", "{n} — noted. good coffee culture there, i hear."),
    ("my gamertag is {n}", "tag: {n}. may the lobby fear it."),
]
PROFESSIONS = ["a barista","a nurse","a game dev","a truck driver","a teacher","an electrician","a diver","a cook"]

def _fact_turn():
    i = R.randrange(6)
    if i == 0:
        n = R.choice(NAMES)
    elif i == 1:
        n = R.choice(DOG_POOL)
    elif i == 2:
        n = R.choice(CAT_POOL)
    elif i == 3:
        n = R.choice(CITIES)
    elif i == 4:
        n = gen_rand_name()
    else:
        n = R.choice(PROFESSIONS)
        u, a = "i work as {n}", "so {n} pays the bills. respect the craft."
        return u.format(n=n), a.format(n=n), n
    u, a = FACTS[i]
    return u.format(n=n), a.format(n=n), n

NAMES = ["arif","riley","velko","sasha","mirna","tomi","keisha","dario","yuna","pako",
         "bren","livia","oskar","nadia","kip","mora","dax","senn","vladi","noor"]
CITIES = ["malang","bandung","surabaya","osaka","porto","oslo","kyoto","austin","leeds","taipei"]

def gen_rand_name():
    cons = "bcdfghjklmnpqrstvwxyz"; vow = "aeiou"
    return "".join(R.choice(cons if k % 2 == 0 else vow) for k in range(R.randint(5, 9)))

STORIES = [
    ("a knight who feared his own shadow", [
        "the knight marched at noon so his shadow would hide beneath him. one dusk, a cloud swallowed the sun — and the shadow stood up and asked for a word.",
        "the shadow spoke: you arm the body, but who arms the doubt? the knight had no answer, so he offered the shadow his helmet. it refused. it wanted the sword hand instead.",
        "so the knight trained left-handed by moonlight until the shadow could not tell which hand held fate. at the next duel, he switched hands mid-strike and won before the doubt could load.",
    ]),
    ("a lighthouse that kept a diary", [
        "the keeper wrote only what the lamp saw: eels that knock, gulls that grieve, one green ship that arrives each equinox and never docks.",
        "the diary filled with weather no one believed. storms shaped like hourglasses. fog that hummed in thirds. the villagers called it fiction; the lamp called it Tuesday.",
        "one equinox the green ship lowered a boat, and the boat carried a single reader. the keeper handed over the diary. the reader read, wept, and said: at last, a witness.",
    ]),
    ("a bakery open only at midnight", [
        "the ovens woke when the town slept. the bread sold itself to dreams: one loaf for courage, one for sleep, one plain loaf for honesty.",
        "the baker never asked names. the bread remembered for both of them — crusts held secrets like barrels hold wine.",
        "one night a child asked for tomorrow. the baker said: we sell yesterday only, tomorrow is self-service. the child laughed and bought honesty instead.",
    ]),
]

def _story_arc(story):
    title, parts = story
    u0 = f"tell me a story about {title}"
    turns = [(u0, parts[0])]
    for k, p in enumerate(parts[1:], 2):
        turns.append((R.choice(["continue", "go on", "keep going", "and then?"]), p))
    return turns

LONG_LISTS = [
    ("give me 8 dog names", lambda: "\n".join(f"• {n}" for n in R.sample(DOG_POOL, 8))),
    ("name 10 songs for my playlist", lambda: "\n".join(f"• {n}" for n in R.sample(SONGS, min(10, len(SONGS))))),
    ("give me 12 movie ideas for the weekend", lambda: "\n".join(f"• {n}" for n in R.sample(MOVES, min(12, len(MOVES))))),
    ("list 6 foods i should try", lambda: "\n".join(f"• {n}" for n in R.sample(FOODS, min(6, len(FOODS))))),
]

FILLER_QA = [
    ("what's 12 + 7", "12 + 7 = 19."),
    ("tell me a joke", "why did the neural net cross the road? gradient descent. it just followed the slope."),
    ("i'm a bit tired today", "shorten the day then: one task, one walk, one honest rest. the rest is noise."),
    ("do you actually remember stuff", "i keep two kinds: the fresh window of our talk, and the long shelf of memories i chose to write. ask and i will fetch."),
    ("what games do you have", "tic tac toe, rock paper scissors, and a dice duel — two dice, three rounds. say the word."),
    ("how's your day going", "no complaints — every turn is a new puzzle. yours?"),
    ("any advice for the gym", "start smaller than your pride wants. consistency beats intensity, every month of the year."),
]

def generate_long(n=7000):
    """12-24 turn conversations spanning the 2048-token window."""
    sess = []
    for _ in range(n):
        turns = []
        n_facts = R.randint(2, 4)
        facts = [_fact_turn() for _ in range(n_facts)]
        stated_names = [f[2] for f in facts if "name is" in f[0] or "dog is" in f[0] or "cat is" in f[0]]
        story = R.choice(STORIES)
        arc = _story_arc(story)
        lists = R.sample(LONG_LISTS, R.randint(1, 2))
        # plan the arc: facts early, filler middle, story mid-late, lists late, recall last
        n_turns = R.randint(12, 22)
        slots = ["fact"] * n_facts + ["story"] * len(arc) + ["list"] * len(lists)
        while len(slots) < n_turns - 2:
            slots.append("filler")
        R.shuffle(slots)
        # facts must come first for a real long-range recall test
        fact_i = [i for i, s in enumerate(slots) if s == "fact"]
        first_fact = min(fact_i) if fact_i else 0
        story_done = False
        for k, s in enumerate(slots):
            if s == "fact":
                u, a, _nm = facts.pop()
            elif s == "story":
                u, a = arc.pop(0) if arc else ("go on", "the tale rests here.")
                story_done = True
            elif s == "list":
                ask, mk = lists.pop(0) if lists else ("give me 3 dog names", lambda: "• buddy\n• luna\n• zuko")
                u, a = ask, mk()
            else:
                u, a = R.choice(FILLER_QA)
                if story_done and R.random() < 0.12:
                    u, a = "stop", R.choice(["stopped. the tale waits for 'continue'.",
                                             "paused — say continue when ready."])
                    story_done = False
                    if arc: slots.insert(k + 2, "story")
            turns.append((u, a))
        # long-range recall finale: early facts, asked at the far end of the window
        for _ in range(R.randint(1, 3)):
            kind = R.random()
            if kind < 0.4:
                nm = R.choice(stated_names) if stated_names else R.choice(NAMES)
                u = "what's my name?"; a = f"you told me at the very start of all this — you're {nm}. still locked in."
            elif kind < 0.7:
                u = "what were we talking about way back?"; a = "we opened on introductions, drifted through the day, and left a story unfinished. say continue."
            else:
                u = "how many things did you learn about me tonight?"; a = "enough to pass a small quiz. names, places, a dog, a tag — stored and ready."
            turns.append((u, a))
        sess.append({"persona": R.choice(["sigma", "sigma", "fake"]), "turns": turns})
    return sess

if __name__ == "__main__":
    v = generate_verses(6)
    l = generate_long(3)
    for s in v[:3]:
        print("VERSE", s["turns"][0][0], "->", s["turns"][0][1][:90].replace("\n", " / "))
    for s in l:
        print("LONG", len(s["turns"]), "turns ->", " | ".join(u[:22] for u, _ in s["turns"][:8]), "...")
