"""Main data pipeline: generates ALL training data from scratch, builds the
tokenizer vocab, serializes samples with loss masks, saves pickles + vocab.

v2: every assistant turn now carries a <think>...</think> private plan
(gen_think.think_for) and the dataset includes story/list INTERRUPT families
(fam_interrupt) so a paused task never swallows unrelated messages.
vocab2 = old vocab ids (stable for existing checkpoints) + <think>/</think>
appended at the end.
v4 (NAME COPY): vocab3 = vocab2 + EVERY missing single letter appended. OOV
words (names, gamertags, cities, pets — anything the user types) are now spelled
letter-by-letter, so the model can read and copy ARBITRARY user text. No more
canned-name hallucinations; memory is open-ended. Old ids stay byte-identical,
so existing checkpoints keep working (they just clamp the new ids)."""
import os, sys, pickle, json, random
sys.path.insert(0, os.path.dirname(__file__))
import tokenizer as tokmod
import gen_chat, gen_memory, personas_gen, gen_think, gen_copy, gen_lists, gen_ttt, gen_bgmem, gen_shakespeare, gen_long

R = random.Random(99)
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

DOG_POOL = ["buddy","max","bella","luna","charlie","rocky","daisy","milo","cooper","bailey",
            "toby","bear","zed","kilo","apollo","nova","juno","tank","pixel","mango",
            "brutus","waffles","mochi","zuko","pepper","nugget","biscuit","wasabi","kombu","tofu",
            "miso","pickles","truffle","ginger","cocoa","maple","clover","banjo","rumble","echo"]

INTENT_TTT = ["let's play tic tac toe","wanna play tic tac toe","tic tac toe?","fancy a game of tic tac toe",
    "you vs me tic tac toe","challenge me in tic tac toe","quick tictactoe match","i'm down for tic tac toe",
    "bro tic tac toe now","play me in tic tac toe","one round of tic tac toe","tic tac toe rematch",
    "tic tac toe, first to win","set up tic tac toe","give me a tic tac toe game","draw the tic tac toe grid",
    "tic tac toe us","how about tic tac toe","tic tac toe sounds good right now","hit me with tic tac toe",
    "i dare you to tic tac toe","start a tic tac toe board","we should play tic tac toe","tic tac toe me"]
INTENT_RPS = ["rock paper scissors","let's do rock paper scissors","rps?","throw rock paper scissors with me",
    "rock paper scissors best of 3","rps me right now","rock paper scissors match","pick up rock paper scissors",
    "lets throw some rock paper scissors","rps, you and me","how about rock paper scissors","rock paper scissors go",
    "challenge accepted: rock paper scissors","start rock paper scissors","i want rock paper scissors","rps rematch",
    "rock paper scissors duel","play rock paper scissors with me now","rock paper scissors round one","boom rock paper scissors"]
INTENT_DICE = ["roll the dice","dice battle","throw some dice","roll 2 dice","dice?","let's gamble on dice",
    "roll two dice with me","dice game time","roll them bones","dice me","let's do a dice roll off","highest roll wins, dice",
    "two dice, go","dice showdown","i call dice","roll dice with me bro","dice round, now",
    "give me the dice game","can we play dice?","hit me with them dice",
    "hit me with some dice","toss the dice with me","slide me the dice","dice me up","roll em, dice now"]
INTENT_ASK = ["got any games?","what games do you have","play something with me","wanna play a game",
    "i'm bored, entertain me","any mini games?","let's play","entertain me","play with me",
    "got games on this thing","bored... something fun?","occupy me with a game","give me a game","we should play something",
    "some fun then? a game?","what can we play together","choose a game for us","open a game, any game","i want to play",
    "bored out of my mind, game?","find us something to play","pick a game and let's go","game night?"," entertain me please"]
INTENT_NONE = ["tic tac toe is boring","i always lose at tic tac toe","how do you win tic tac toe",
    "who invented rock paper scissors","rock paper scissors is pure luck","dice games are luck based",
    "i lost my dice somewhere","what's the best tic tac toe strategy","is tic tac toe a solved game",
    "playing games all day is bad for you","i played dice with my brother yesterday","teach me rock paper scissors tactics",
    "my dice collection is huge","tic tac toe champions think three moves ahead","the dice landed on six twice",
    "hi","hello there","how are you","who are you","what's 12 + 7","tell me a joke","i'm sad today",
    "thanks a lot","goodnight","you're dumb","i love you man","how's the weather","count to 8",
    "my name is arif","my name is agusi","call me riley","my name is velko",
    "my dog is called brutus","my gamertag is frostbyte","i live in malang",
    "what's my name?","i think cats beat dogs","should i start the gym","tell me a fun fact",
    "do you have memory","what can you do","i got the job!!","i'm building a game about ninjas",
    "continue the story","go on","keep going","stop","wait stop","ok stop","that's all for today bye",
    "what did i ask you before","remember my dog's name?","what were we talking about",
    # v9: rude shutdowns & commands are NEVER game invites (live bug: 'shut the
    # fuck up' opened the dice board)
    "shut the fuck up","stfu","shut up","be quiet","quiet now","hush","stop it","stop talking",
    "stop pls","just stop","you're useless","you're so dumb","dumb bot","that was trash",
    "i hate this app","you annoy me","damn you're annoying","shut your mouth",
    # list requests are not game invites either
    "give me 3 dog names","give me 5 movie ideas","name some songs","give me 2 cat names",
    "recommend me 4 foods","give me 3 story ideas","i need username ideas"]

def augment(u):
    v = [u, u.capitalize(), u + " ?", u.rstrip("?") + "?", "yo " + u, u + " lol", "hey " + u, "ok " + u]
    return v

def gen_intent(n_each=(3200, 3200, 3200, 3800), n_none=15000):
    labeled = []
    for label, bank in [(1, INTENT_TTT), (2, INTENT_RPS), (3, INTENT_DICE), (4, INTENT_ASK)]:
        for _ in range(n_each[label-1]):
            u = R.choice(bank)
            if R.random() < 0.45:
                u = R.choice(augment(u))
            labeled.append((u, label))
    for _ in range(n_none):
        u = R.choice(INTENT_NONE)
        if R.random() < 0.3:
            u = R.choice(augment(u))
        labeled.append((u, 0))
    R.shuffle(labeled)
    return labeled

ACK_MARKS = ("stopped", "paused", "halted", "on hold", "your move", "say continue", "the tale waits")

def add_thinks(sample, kind):
    """Attach a think trace to every turn, conditioned on real turn content.
    Tracks the most recent SUBSTANTIVE task text (list/story) so 'continue'
    thoughts anchor to the real resume point instead of the stop ack."""
    persona = sample.get("persona") or "sigma"
    last_answer = ""
    task_text, task_type = "", ""
    thinks = []
    for u, a in sample["turns"]:
        thinks.append(gen_think.think_for(u, a, {
            "persona": persona, "last_answer": last_answer,
            "task_text": task_text, "task_type": task_type}))
        last_answer = a
        low = a.lower()
        if any(k in low for k in ACK_MARKS):
            continue                      # acks don't change the pending task
        if "," in a and ":" in a:
            task_text, task_type = a, "list"
        elif len(a) > 80:
            task_text, task_type = a, "story"
    sample["thinks"] = thinks
    return sample

def serialize(sample, stoi):
    """Returns (ids, mask). Assistant turns: <think>plan</think> answer <mem>...</mem>?"""
    ids, mask = [stoi["<bos>"]], [0]
    def emit(text, m):
        for w in tokmod.tokenize_text(text):
            for i in tokmod.token_ids(stoi, w):
                ids.append(i)
                mask.append(m)
    p = sample.get("persona") or "sigma"
    emit(f"<pers:{p}>", 0)
    if sample.get("desc"):
        emit(sample["desc"], 0)
        emit("<nl>", 0)
    if sample.get("block"):
        emit(sample["block"], 0)
    thinks = sample.get("thinks") or [None] * len(sample["turns"])
    for (u, a), th in zip(sample["turns"], thinks):
        emit("<user>", 1)
        emit(u, 0)
        emit("<ai>", 1)
        if th:
            emit("<think>", 1)
            emit(th, 1)
            emit("</think>", 1)
        emit(a, 1)
    emit("<eos>", 1)
    if len(ids) > 2048:           # hard context limit (positional emb): drop overflow.
        return None               # v11 raises this from 512 -> the 2048 window
    return ids, mask

def main():
    os.makedirs(DATA, exist_ok=True)
    print("[1/7] generating conversations...")
    core = gen_chat.generate_core(82000)
    flows = gen_chat.generate_gameflows(9000)
    pers = personas_gen.generate_personas(2600)
    mem_sessions = gen_memory.generate_sessions(DOG_POOL, 26000)
    # v7 COPY CURRICULUM: fresh random names every regen (billions-wide space)
    # -> memorization impossible, true context-copying enforced.
    copies = gen_copy.generate_copies(60000)
    # v9 LIST + COMMAND families ('give me 3 dog names' / 'stop' / 'shut up')
    lists = gen_lists.generate_lists(30000)
    # v9 TICTACTOE BOARD PLAY: every reachable position, model picks its move
    ttt = gen_ttt.generate_ttt_sessions(16000)
    # v9 dedicated block+noise+recall mixers
    bgmixed = gen_bgmem.generate_mixed(9000)
    # v10 SHAKESPEARE-GMA: trained persona behind the mode switch (+ in-context style flips)
    shak = gen_shakespeare.generate_shakespeare(15000)
    # v11 SHAKESPEAGMA topic->verse/prose/socratic on demand (the word-forge page)
    verses = gen_long.generate_verses(14000)
    # v11 LONG-CONTEXT: 12-24 turn conversations that use the whole 2048 window
    longc = gen_long.generate_long(17000)
    # v7: the dog/task pool goes OOV-heavy too — 'my dog is called <x>' must mean
    # COPY <x>, not sample an in-vocab dog name (the live 'kombu->truffle' bug).
    dog_pool_mix = DOG_POOL + [gen_copy.rand_name() for _ in range(3000)]

    print("[2/7] attaching think traces...")
    unk_watch = 0, 0
    for s in core: add_thinks(s, "core")
    for s in flows: add_thinks(s, "flow")
    for s in pers: add_thinks(s, "persona")
    for s in mem_sessions: add_thinks(s, s.get("kind", "mem"))
    for s in copies: add_thinks(s, "copy")
    for s in lists: add_thinks(s, "lists")
    for s in bgmixed: add_thinks(s, "bgmem")
    for s in shak: add_thinks(s, "core")
    for s in verses: add_thinks(s, "core")
    for s in longc: add_thinks(s, "core")
    # ttt sessions carry their own inline <think> inside the answer
    for s in ttt: s["thinks"] = [None] * len(s["turns"])

    print("[2b/7] threading BACKGROUND <memory> blocks through all families...")
    # THE core live-bug fix: in the browser the block is ALWAYS present, so it
    # must sit in context while the model does ordinary chat/lists/commands/
    # games — answering the current message, never echoing the block unprompted.
    n_bg = 0
    n_bg += gen_bgmem.thread(core, 0.55)
    n_bg += gen_bgmem.thread(lists, 0.85)
    n_bg += gen_bgmem.thread(flows, 0.85)
    n_bg += gen_bgmem.thread(pers, 0.55)
    n_bg += gen_bgmem.thread(mem_sessions, 0.30)
    n_bg += gen_bgmem.thread(copies, 0.35)
    n_bg += gen_bgmem.thread(shak, 0.45)
    n_bg += gen_bgmem.thread(verses, 0.45)
    n_bg += gen_bgmem.thread(longc, 0.35)
    print(f"  background blocks injected into {n_bg} sessions")

    print("[3/7] vocab3 FROZEN (byte-identical ids for every checkpoint)...")
    # v5: vocab3 is FROZEN. Its 1588 ids already cover every letter + digit +
    # punctuation; the copy curriculum deliberately lives on letters, so no new
    # words are needed. Freezing keeps old checkpoints (10m/17m/20m) aligned.
    vp = os.path.join(DATA, "vocab3.json")
    with open(vp, "r", encoding="utf-8") as f:
        tk = tokmod.Tokenizer(json.load(f))
    tk.save(vp)
    stoi = tk.stoi
    print(f"  vocab3 size: {len(tk)} (frozen)")

    # OOV check on think text (want < 2%)
    tot, oov = 0, 0
    for s in core + flows + pers + mem_sessions + copies + lists + bgmixed + shak + verses + longc:
        for th in s["thinks"]:
            if not th: continue
            for w in tokmod.tokenize_text(th):
                tot += 1
                if w not in stoi: oov += 1
    print(f"  think-token OOV rate: {oov}/{tot} = {oov/max(tot,1):.2%}")

    print("[4/7] serializing LM samples with masks...")
    samples = []
    dropped = 0
    for s in core + flows + pers + mem_sessions + copies + lists + ttt + bgmixed + shak + verses + longc:
        r = serialize(s, stoi)
        if r: samples.append(r)
        else: dropped += 1
    intent = gen_intent()
    print("  lm samples:", len(samples), " intent samples:", len(intent),
          f" (dropped>2048tok: {dropped}, lists: {len(lists)}, ttt: {len(ttt)}, shak: {len(shak)}, verses: {len(verses)}, long: {len(longc)})")

    R.shuffle(samples)
    n_eval = int(len(samples) * 0.015)
    ev, tr = samples[:n_eval], samples[n_eval:]
    # v11: save as numpy arrays — 51M tokens as python lists ate ~2GB and the
    # kernel OOM-killed ctx-2048 training. arrays: ~6x smaller, same tuples.
    import numpy as np
    with open(os.path.join(DATA, "lm_train.pkl"), "wb") as f:
        pickle.dump([(np.asarray(a, np.int32), np.asarray(b, np.uint8)) for a, b in tr], f, protocol=4)
    with open(os.path.join(DATA, "lm_eval.pkl"), "wb") as f:
        pickle.dump([(np.asarray(a, np.int32), np.asarray(b, np.uint8)) for a, b in ev], f, protocol=4)

    it = []
    import numpy as np
    for u, lab in intent:
        ids = [stoi["<bos>"], stoi[f"<pers:{R.choice(['fake','sigma'])}>"]]
        for w in tokmod.tokenize_text("<user>" + u):
            ids.extend(tokmod.token_ids(stoi, w))
        it.append((np.asarray(ids, np.int32), int(lab)))
    with open(os.path.join(DATA, "intent.pkl"), "wb") as f:
        pickle.dump(it, f, protocol=4)

    ntok = sum(len(a) for a, _ in tr)
    stats = dict(vocab=len(tk), lm_train=len(tr), lm_eval=len(ev), intent=len(it),
                 total_tokens=ntok, avg_len=round(ntok/len(tr), 1))
    with open(os.path.join(DATA, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[6/6] saved:", stats)

if __name__ == "__main__":
    main()
