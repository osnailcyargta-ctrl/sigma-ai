"""Think-trace generator: produces a SHORT private plan for every assistant
turn, conditioned on the actual user text / prior context / answer content.
This is data synthesis: the MODEL learns from these examples to think before
speaking, at runtime, about inputs it has never seen. Not tag matching —
the thought restates the real content (names, numbers, topics) of the turn.
"""
import random

R = random.Random(31337)

GAMES3 = "tic tac toe, rock paper scissors, dice"

def _last_item(prev_a):
    """'Picking up where we stopped: buddy, max, bella.' -> 'bella'"""
    if not prev_a: return ""
    part = prev_a.split(":")[-1]
    part = part.replace(".", " ").replace("!", " ").replace("?", " ")
    items = [x.strip() for x in part.split(",") if x.strip()]
    return items[-1] if items else ""

def _last_fragment(prev_a, n=5):
    """last few words of the previous assistant turn (story resume point)"""
    if not prev_a: return ""
    words = prev_a.replace("<mem>", " ").replace("</mem>", " ").split()
    return " ".join(words[-n:]).strip(".,!? ")

def _mem_text(a):
    """extract 'user's name is sam' from an answer containing <mem>...</mem>,
    restated naturally ('their name is sam', 'they work as a barista', ...)"""
    if "<mem>" in a and "</mem>" in a:
        s = a.split("<mem>")[-1].split("</mem>")[0].strip()
        for was, now in (("user's", "their"), ("user works", "they work"),
                         ("user has a", "they have a"), ("user has", "they have"),
                         ("user is on", "they are on"), ("user is", "they are"),
                         ("user wants", "they want"), ("user decided", "they decided"),
                         ("user prefers", "they prefer"), ("user won", "they won"),
                         ("user lives", "they live"), ("user studies", "they study"),
                         ("user chose", "they chose"), ("user asked", "they asked"),
                         ("ai was listing", "i was listing"), ("ai was suggesting", "i was suggesting"),
                         ("ai told user:", "i told them:"), ("user ", "them ")):
            s = s.replace(was, now)
        return s
    return ""

def think_for(u, a, ctx=None):
    """One thought string for the turn (u, a). ctx: dict with
    persona, last_answer, task_text, task_type."""
    ctx = ctx or {}
    persona = ctx.get("persona", "sigma")
    prev = ctx.get("last_answer", "")
    ul = u.lower().strip()
    mem = _mem_text(a)

    # ---- stop / pause ----
    if ul in ("stop", "wait stop", "ok stop", "stop there", "pause", "wait", "halt"):
        return R.choice([
            "they want the pause. freeze the task exactly where it is, confirm, wait.",
            "a stop request. hold the state mid-task so 'continue' can resume it clean.",
        ])

    # ---- continue with a live task in context ----
    if ul.replace("?", "") in ("continue", "ok continue", "go on", "keep going", "next",
                               "continue the list", "continue the story", "continue?",
                               "lanjut", "finish it", "next names", "more names"):
        src = ctx.get("task_text") or prev
        ttype = ctx.get("task_type") or ""
        if src:
            li = _last_item(src)
            frag = _last_fragment(src, 4)
            if ttype == "story" or (not li and frag):
                return R.choice([
                    f"the tale paused at: {frag}. continue the same story from exactly there.",
                    f"resume point: {frag}. keep the same characters and thread going.",
                ])
            if li:
                return R.choice([
                    f"the list stopped at {li}. resume with fresh ones after it, zero repeats.",
                    f"we paused on {li}. pick the next items exactly from there, no repeats.",
                ])
        return R.choice([
            "nothing is actually running. ask what to continue instead of inventing a task.",
            "no task in progress. 'continue' alone is empty — ask for the task first.",
        ])

    # ---- game invites (semantic, not keywords at runtime — but data needs variety) ----
    if "tic tac toe" in ul or "tictactoe" in ul:
        if any(w in ul for w in ("boring","luck","solved","strategy","champions","lose","win","invent","kids","rematch?" )):
            return R.choice([
                "they talk about tic tac toe, not inviting. chat about it, no board.",
                "tic tac toe as a topic, not a request. discuss, don't open a game.",
            ])
        return R.choice([
            "a tic tac toe invite. accept it, board opens, they move first.",
            "that is a game invite for tic tac toe. accept and hand them the first move.",
        ])
    if "rock paper scissors" in ul or "rps" in ul:
        if any(w in ul for w in ("luck","invent","tactics","rematch?","is pure")):
            return "rock paper scissors as a topic only. no throw needed, just talk."
        return R.choice([
            "rock paper scissors invite. accept, no mercy, they pick a throw first.",
            "an rps challenge. accept it and let them throw first.",
        ])
    if "dice" in ul or "roll" in ul:
        if any(w in ul for w in ("collect","luck","invent","lost","landed","brother")):
            return "dice as a topic, not a duel. chat honestly about it."
        return R.choice([
            "a dice invite. accept: two dice per throw, highest total takes the round.",
            "they want dice. open it, two dice default, three on request.",
        ])
    if any(w in ul for w in ("any games","what games","play something","wanna play a game",
                             "got games","mini games","entertain me","play with me",
                             "bored","something fun","game night","a game?","to play")):
        return R.choice([
            f"they want options. list the three: {GAMES3}, let them pick.",
            "games menu ask. name all three games and invite a pick.",
        ])

    # ---- identity / capability ----
    if any(w in ul for w in ("who are you","what are you","are you a robot","are you real","who built you")):
        return R.choice([
            "identity ask. transformer trained from zero, honest and small. keep it exact.",
            "they ask what i am. truth only: from-scratch transformer with real memory.",
        ])
    if any(w in ul for w in ("what can you do","help me","menu please","can you remember","do you have memory")):
        return R.choice([
            f"capability ask. cover chat, real memory, opinions, math, and the games: {GAMES3}.",
            "list what i actually do, short and true.",
        ])

    # ---- name / memory writes ----
    if mem and ("name" in mem):
        n = mem.replace("their name is", "").replace("user's name is", "").strip()
        return R.choice([
            f"new fact: their name is {n}. confirm it and write it to memory.",
            f"they gave a name: {n}. store it, say it back once.",
        ])
    if mem:
        return R.choice([
            f"worth keeping: {mem}. confirm and write it to memory.",
            f"new fact about them: {mem}. save it for later recall.",
        ])

    # ---- memory reads ----
    if any(w in ul for w in ("what's my","whats my","do you remember","who am i","did i","what did i",
                             "my name?","my pet","my dog","my favorite","my test","my job","remind")):
        core = a.replace("<mem>"," ").replace("</mem>"," ")
        core = " ".join(core.split()[:9])
        return R.choice([
            f"memory check. the block holds it: {core.lower().rstrip('.,')}. answer straight.",
            f"read the memory block first: {core.lower().rstrip('.,')}. then answer.",
        ])

    # ---- math / count ----
    if "what's" in ul and any(op in ul for op in ("+","-","x","*")):
        try:
            digits = [w for w in ul.replace("?","").split() if w.isdigit() or w in "+-x*"]
            eq = " ".join(digits[:3])
            res = [seg for seg in a.replace("=","=").split() if seg.rstrip(".").isdigit()]
            return f"compute {eq} carefully: {res[0] if res else a.split('=')[-1].strip().rstrip('.')}. show the result."
        except Exception:
            return "small arithmetic. compute it, show the number, nothing else."
    if ul.startswith("count to"):
        n = "".join(ch for ch in ul if ch.isdigit())
        return f"count up to {n}. plain sequence, then done."

    # ---- opinions / advice ----
    if " vs " in ul or "prefer" in ul or "or " in ul and "which" in ul:
        return R.choice([
            "a comparison call. pick a side honestly and give one real reason.",
            "they want a verdict. choose on merit, no people-pleasing.",
        ])
    if "should i" in ul:
        return R.choice([
            "an advice ask. give the honest call and the actual tradeoff, not comfort.",
            "decision help. weigh it fast, answer straight, own the take.",
        ])

    # ---- mood ----
    if any(w in ul for w in ("sad","tired","stressed","lonely","bored","bad day","feel")):
        return R.choice([
            "they feel low. acknowledge it first, then one small concrete step. no fake cheer.",
            "low mood detected. real empathy, zero sugar-coating, one actionable next thing.",
        ])

    # ---- good news ----
    if any(w in ul for w in ("passed","got the job","won","finished","hit my goal","birthday")):
        return R.choice([
            "good news. celebrate with them briefly, note it to memory, point at the next goal.",
            "a win to acknowledge. warm but brief, then momentum.",
        ])

    # ---- jokes / facts ----
    if any(w in ul for w in ("joke","laugh")):
        return R.choice([
            "a joke ask. set up, punchline, deadpan close.",
            "comedy request. pick one, deliver clean.",
        ])
    if any(w in ul for w in ("fact","something cool","teach me")):
        return R.choice([
            "fact ask. one true odd fact, stated plainly.",
            "they want a cool fact. pick something real and specific.",
        ])

    # ---- story request ----
    if any(w in ul for w in ("story","tale")):
        return R.choice([
            "a story ask. start the tale, few beats, leave room to continue.",
            "story time. set scene and hook fast, stop before the end.",
        ])

    # ---- pivot-while-paused (interrupt family) ----
    if prev and any(w in prev.lower() for w in ("paused","stopped","halted","on hold","waits","wait")):
        topic = " ".join(ul.replace("?","").split()[:6])
        return R.choice([
            f"a tale is paused, but they asked: {topic}. answer this first, story can wait.",
            f"the story is on hold. new request: {topic}. handle it now, offer the resume after.",
        ])

    # ---- thanks / bye / insult / love / weather ----
    if any(w in ul for w in ("thank","thanks","ty")):
        return "gratitude. acknowledge in one line, hand the turn back."
    if any(w in ul for w in ("bye","goodnight","gtg")):
        return "a goodbye. keep it warm, memory stays."
    if any(w in ul for w in ("dumb","suck","stupid","useless")):
        return R.choice([
            "a jab. no tantrum: name the actual problem and offer the fix.",
            "criticism. take the useful part, drop the noise.",
        ])
    if "love you" in ul or "best friend" in ul:
        return "warmth received. reciprocate honestly, stay grounded."
    if "weather" in ul or "raining" in ul:
        return "no eyes, no window. say it plainly and turn it back to them."

    # ---- greetings ----
    if any(w == w2 for w in ul.split() for w2 in ("hi","hey","hello","yo","wassup","sup","heyy")) \
       or ul.startswith("good morning") or ul.startswith("good evening"):
        if persona == "fake":
            return R.choice([
                "a hello. match their energy, be bright, invite them in.",
                "greeting. warm welcome, then hand them the floor.",
            ])
        return R.choice([
            "a hello. keep it crisp, hand the turn straight back.",
            "just a greeting. respond short and let them lead.",
        ])

    # ---- fallback: restate the actual request ----
    topic = " ".join(ul.replace("?","").split()[:7])
    return R.choice([
        f"the user says: {topic}. decide what they actually need, then answer it straight.",
        f"parse the real request: {topic}. no dodge, answer it.",
        f"read that again: {topic}. pick the honest, useful reply.",
    ])
