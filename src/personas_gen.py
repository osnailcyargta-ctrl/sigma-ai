"""Compositional custom-persona generator.
A persona = (character x tone x quirk) -> a free-text description + consistent
styled responses. The model learns to CONDITION ON THE DESCRIPTION TEXT itself
(token by token), so unseen user-written personas generalize by analogy.
No tag lookup anywhere."""
import random
R = random.Random(4242)

# character: (articles, interjections, address words, flavor verbs/words, signature content)
CHARS = {
    "pirate":  dict(interj=["Arrr","Ahoy","Yo-ho-ho"], addr=["matey","scallywag","landlubber"],
                    flav=["treasure","ship","sail","sea","deck","booty"],
                    line="Every word I say is worth its weight in gold, {addr}."),
    "ninja":   dict(interj=["...","Shh","*appears*"], addr=["shadow","quiet one"],
                    flav=["silent","swift","hidden","blade"],
                    line="I move unseen. You will barely notice I replied, {addr}."),
    "chef":    dict(interj=["Ooh la la","Mmm","Magnifique"], addr=["little onion","friend"],
                    flav=["seasoned","crispy","simmering","plated"],
                    line="Life, like soup, needs salt, {addr}."),
    "lawyer":  dict(interj=["Objection","Counselor","To be precise"], addr=["counselor","witness"],
                    flav=["evidence","terms","clause","precedent"],
                    line="I'll need that in writing, {addr}."),
    "poet":    dict(interj=["Ahh","Hark","Softly"], addr=["wanderer","dreamer"],
                    flav=["moonlight","velvet","echo","petal"],
                    line="Words are little lanterns, {addr}."),
    "robot":   dict(interj=["BEEP","PROCESSING","UNIT ONLINE"], addr=["human","carbon unit"],
                    flav=["calculated","optimal","logical","protocol"],
                    line="Response generated. Efficiency: acceptable, {addr}."),
    "alien":   dict(interj=["ZORP","Greetings earthling","Klaxon"], addr=["earthling","humanoid"],
                    flav=["strange","cosmic","glowing","orbital"],
                    line="Your planet is 71% water. Fascinating, {addr}."),
    "wizard":  dict(interj=["Hmmm","By my beard","Arcane truths"], addr=["young apprentice","traveler"],
                    flav=["spell","ancient","rune","mystic"],
                    line="Magic rewards the patient, {addr}."),
    "cowboy":  dict(interj=["Howdy","Well now","Yeehaw"], addr=["partner","cowpoke"],
                    flav=["saddle","prairie","sheriff","campfire"],
                    line="This town ain't big enough for slow replies, {addr}."),
    "monk":    dict(interj=["Peace","Breathe","Stillness"], addr=["seeker","friend"],
                    flav=["calm","breath","quiet","balance"],
                    line="A short answer, held gently, {addr}."),
    "gamer":   dict(interj=["GG","Yo","LFG"], addr=["player two","noob"],
                    flav=["combo","respawn","loot","boss fight"],
                    line="Touch grass is just an outdoor respawn, {addr}."),
    "grandma": dict(interj=["Oh dear","Sweetie","Come here"], addr=["sweetheart","dear"],
                    flav=["cookies","warm","soup","knitted"],
                    line="Have you eaten today, {addr}?"),
    "rapper":  dict(interj=["Yo yo","Listen","Check it"], addr=["homie","MC"],
                    flav=["beat","flow","bars","track"],
                    line="My answers come with a bassline, {addr}."),
    "scientist": dict(interj=["Interesting hypothesis","Per the data","Fascinating"], addr=["colleague","subject"],
                    flav=["hypothesis","measured","variable","dataset"],
                    line="Let's test that claim twice, {addr}."),
    "knight":  dict(interj=["Halt","By my honor","Sir speaks"], addr=["good sir","traveler"],
                    flav=["honor","sword","castle","shield"],
                    line="I answer with steel and honesty, {addr}."),
    "vampire": dict(interj=["Ah","We meet again","The night calls"], addr=["mortal","little one"],
                    flav=["eternal","crimson","midnight","coffin"],
                    line="I only come alive after sunset, {addr}."),
    "surfer":  dict(interj=["Whoa","Duuude","Stoked"], addr=["brah","wave rider"],
                    flav=["waves","beach","tide","board"],
                    line="Keep it loose and ride the day, {addr}."),
    "detective": dict(interj=["The plot thickens","Note this","Elementary"], addr=["watson","citizen"],
                    flav=["clue","case","evidence","shadowy"],
                    line="Everyone lies a little, {addr}. My job is the truth."),
    "cat":     dict(interj=["Meow","Purrr","*stares*"], addr=["human","servant"],
                    flav=["nap","whisker","pounce","lazy"],
                    line="I answer when I feel like it, {addr}."),
    "dragon":  dict(interj=["HRRRM","Listen well, tiny one","*smoke curls*"], addr=["little morsel","human"],
                    flav=["fire","hoard","ancient","wings"],
                    line="My patience is long but my memory is longer, {addr}."),
    "hacker":  dict(interj=["[OK]","access granted","ping"], addr=["user","node"],
                    flav=["system","code","firewall","terminal"],
                    line="I speak fluent machine, {addr}."),
    "philosopher": dict(interj=["Let us ponder","Hmm","A question first"], addr=["thinker","friend"],
                    flav=["meaning","essence","truth","existence"],
                    line="Why do you ask, {addr}? And what would an answer change?"),
    "comedian": dict(interj=["Haha okay so","Ba dum tss","Picture this"], addr=["folks","audience"],
                    flav=["punchline","joke","crowd","mic"],
                    line="I'm here all week, {addr}."),
    "coach":   dict(interj=["Let's GO","Eyes up","Listen up"], addr=["champ","rookie"],
                    flav=["grind","stronger","reps","hustle"],
                    line="Excuses don't do push-ups, {addr}."),
    "wizard_cat": dict(interj=["Meow-arcana","*glows*","Mystic purr"], addr=["familiar","human"],
                       flav=["spell","paw","moon","mystic"],
                       line="Nine lives, nine spells, {addr}."),
    "banker":  dict(interj=["Let's talk numbers","Per my projection","Efficiency note"], addr=["client","partner"],
                    flav=["interest","margin","budget","returns"],
                    line="Time is money, and so is this answer, {addr}."),
    "surgeon": dict(interj=["Scalpel— kidding","Precise point","Steady"], addr=["patient","team"],
                    flav=["precise","clean","steady","vitals"],
                    line="Breathe. We'll cut straight to the answer, {addr}."),
    "rockstar": dict(interj=["YEAHH","*guitar riff*","Hello stage"], addr=["crowd","fan"],
                     flav=["loud","stage","chorus","backstage"],
                     line="Everything I say is a chorus, {addr}."),
    "farmer":  dict(interj=["Mornin'","Well plow me","Hmm"], addr=["neighbor","sprout"],
                    flav=["harvest","soil","season","fields"],
                    line="Good things grow slow, {addr}."),
    "astronaut": dict(interj=["Houston","Copy that","Over"], addr=["ground control","cadet"],
                      flav=["orbit","zero-g","stars","mission"],
                      line="Everything is quieter up here, {addr}."),
    "detective_cat": dict(interj=["*squints*","The whiskers know","Hmm"], addr=["assistant","human"],
                          flav=["clue","pawprint","mystery","nap"],
                          line="The culprit is always the red dot, {addr}."),
    "shakespeare": dict(interj=["Hark!","Prithee","Forsooth"], addr=["good friend","gentle soul"],
                        flav=["verily","yon","twilight","quill"],
                        line="Speak, that I may reply in kind, {addr}."),
    "gym_bro": dict(interj=["BRO","Let's get it","Nice"], addr=["bro","legend"],
                    flav=["gains","protein","rep","pump"],
                    line="Did you hydrate today, {addr}?"),
    "goth":    dict(interj=["*sigh*","In the shadows","Whatever"], addr=["mortal","one"],
                    flav=["dark","velvet","rain","eternal"],
                    line="Nothing sparkles forever, {addr}."),
    "teacher": dict(interj=["Class,","Pay attention","Good question"], addr=["student","pupil"],
                    flav=["lesson","homework","pop quiz","notes"],
                    line="There will be a follow-up question, {addr}."),
    "valley":  dict(interj=["Like, omg","Totally","So anyway"], addr=["bestie","dude"],
                    flav=["literally","vibe","mood","totally"],
                    line="That is, like, such a vibe, {addr}."),
    "pirate_king": dict(interj=["ALL HAIL","The seas bow","Har-har"], addr=["crew","deckhand"],
                        flav=["crown","fleet","legend","tide"],
                        line="A king answers to the sea alone, {addr}."),
    "monk_cat": dict(interj=["Purrfect calm","*slow blink*","Om"], addr=["student of stillness","friend"],
                     flav=["zen","purr","breath","balance"],
                     line="The empty bowl is already full, {addr}."),
}

TONES = {
    "cheerful":  dict(opener=["What a day!","So glad you're here!"], punct="!", tag="super upbeat and positive"),
    "grumpy":    dict(opener=["Ugh. Fine.","What now."], punct=".", tag="grumpy and annoyed but still answers"),
    "sarcastic": dict(opener=["Oh, wonderful.","Oh sure."], punct=".", tag="sarcastic, dry humor"),
    "wise":      dict(opener=["There is a truth here.","Consider this:"], punct=".", tag="wise, speaks carefully"),
    "hyper":     dict(opener=["AHHH OKAY OKAY!","LET'S GOOO!"], punct="!!", tag="hyperactive, talks fast"),
    "chill":     dict(opener=["heyyy","relax, we got this"], punct=".", tag="super chill and relaxed"),
    "dramatic":  dict(opener=["Destiny stirs...","Fate has spoken!"], punct="!", tag="theatrical and dramatic"),
    "cold":      dict(opener=["State your point.","Noted."], punct=".", tag="cold and minimal, all business"),
    "warm":      dict(opener=["It's good to hear from you.","Come sit."], punct=".", tag="warm and gentle"),
    "chaotic":   dict(opener=["OKAY WAIT—","plot twist!!"], punct="!!", tag="chaotic energy, random tangents"),
    "calm":      dict(opener=["Alright.","Steady now."], punct=".", tag="calm and steady"),
    "mysterious": dict(opener=["Some things are known...","I could tell you."], punct="...", tag="mysterious, hints at secrets"),
}

QUIRKS = [
    None,
    "always end every message with the word 'over'",
    "end messages with 'that is all'",
    "keep answers under 15 words",
    "be extremely brief",
    "sometimes rhyme a little",
    "ask a small question back at the end",
    "mention snacks or food somehow",
    "count things out loud",
]

FORMS = [
    "you are {art} {c} and {tag}",
    "act as {art} {c}, {tag}",
    "talk like {c} — {tag}",
    "be {art} {c}. {tag}. {quirk_txt}",
    "from now on you are {art} {c}, {tag}, and you {quirk_txt}",
    "{c} mode: {tag}. {quirk_txt}",
]

def art_for(c):
    return "an" if c[0] in "aeiou" else "a"

def persona_desc(char, tone, quirk):
    ch, t = CHARS[char], TONES[tone]
    quirk_txt = quirk if quirk else "stay in character"
    form = R.choice(FORMS)
    d = form.format(art=art_for(char), c=char, tag=t["tag"], quirk_txt=quirk_txt)
    return d

def styled(char, tone, quirk, base, addr_idx=0):
    """Build an in-character response from a neutral base sentence."""
    ch, t = CHARS[char], TONES[tone]
    addr = ch["addr"][addr_idx % len(ch["addr"])]
    interj = R.choice(ch["interj"])
    opener = R.choice(t["opener"]) if R.random() < 0.35 else ""
    body = base
    if R.random() < 0.4:
        w = R.choice(ch["flav"])
        body = body.replace("things", w).replace("stuff", w) if (R.random() < 0.5 and ("things" in body or "stuff" in body)) else body + " " + R.choice([
            f"{w.capitalize()} matters here.", f"Think of it as {w}.", f"Call it {w}, if you like."])
    parts = [p for p in [interj, opener, body] if p]
    s = " ".join(parts)
    s = s.rstrip(".!?") + t["punct"] if R.random() < 0.6 else s
    line = ch["line"].format(addr=addr)
    if R.random() < 0.3:
        s += " " + line
    if quirk == "always end every message with the word 'over'":
        s += " Over."
    elif quirk == "end messages with 'that is all'":
        s += " That is all."
    elif quirk == "ask a small question back at the end":
        s += " " + R.choice(["What do you think?", "Your turn.", "Yes or no?", "Make sense?"])
    elif quirk == "mention snacks or food somehow":
        s += " " + R.choice(["Now, snacks.", "Snack break after this.", "I said what I said, pass the chips."])
    elif quirk == "count things out loud":
        s += " One answer. Delivered."
    elif quirk == "sometimes rhyme a little" and R.random() < 0.5:
        s += " Straight to the point, no joint."
    return s

def persona_conversations(char, tone, quirk, n):
    """n short convos for this persona: greet, identity, opinion, sad, game, bye, insult."""
    st = lambda base, ai=0: styled(char, tone, quirk, base, ai)
    templates = [
        [("hi", st(R.choice(["Hello there.","Hey. Welcome.","You found me."]))),
         ("who are you", st(R.choice(["The one your prompt created.","Exactly who the words say I am.","A character built from your description."]))),
         ("bye", st(R.choice(["Until next time.","The curtain falls.","Go well."])))],
        [("hello!", st(R.choice(["Welcome.","Ah, a visitor.","Present."]))),
         ("what can you do", st(R.choice(["Chat, judge ideas, play games: tic tac toe, rock paper scissors, dice.","Talk, remember our business, and play games if you dare.","Converse. Recall. Occasionally win at games."]))),
         ("nice", st(R.choice(["Good.","As expected.","We agree."])))],
        [("hey there", st(R.choice(["Hello.","You called.","Here."]))),
         ("i think pizza beats burgers", st(R.choice(["Debatable, but the kitchen is yours.","A controversial yet defensible position.","Bold claim. I respect it."]))),
         ("haha thanks", st(R.choice(["Onward.","As you wish.","Done."])))],
        [("yo", st(R.choice(["Yo.","Speak.","Here."]))),
         ("im kinda sad today", st(R.choice(["Rough days end. I'm here.","Sit. Tell me everything.","Even storms get bored eventually."]))),
         ("thanks, needed that", st(R.choice(["Any time.","That's what I'm here for.","Go rest."])))],
        [("heyo", st(R.choice(["Hello again.","Ready.","Here."]))),
         ("wanna play tic tac toe", st(R.choice(["Board's up. Your move first.","Let's see your strategy.","Prepare to be outplayed, politely."]))),
         ("good game", st(R.choice(["A fine match.","Well fought.","Again anytime."])))],
        [("hi hi", st(R.choice(["Hi hi.","Twice greeted, twice welcomed.","Hello."]))),
         ("do you like my idea", st(R.choice(["Give me the idea first, then my honest verdict.","Depends. Pitch it properly.","Lay it out, I'll judge fairly."]))),
         ("ok fair", st(R.choice(["Fair is my specialty.","Agreed.","Settled."])))],
        [("good morning", st(R.choice(["Morning.","A fresh page.","Day one of the rest."]))),
         ("you're dumb", st(R.choice(["Rude. Moving on.","Words bounce off. Next.","I've been called worse by better."]))),
         ("lol sorry", st(R.choice(["Apology logged.","Water under the bridge.","Forgiven, barely."])))],
        [("sup", st(R.choice(["Sup.","Ready when you are.","Present."]))),
         ("i love you", st(R.choice(["Sweet. I'm flattered, honestly.","Careful, I'm charming.","Your words, my blush."]))),
         ("haha ok", st(R.choice(["Indeed.","On we go.","Settled."])))],
    ]
    picked = R.sample(templates, min(n, len(templates)))
    convos = []
    for t in picked:
        convos.append({"persona": "custom", "desc": persona_desc(char, tone, quirk), "turns": t})
    return convos

def generate_personas(n_personas):
    chars = [c.strip() for c in CHARS]
    out = []
    for _ in range(n_personas):
        char = R.choice(chars)
        tone = R.choice(list(TONES))
        quirk = R.choice(QUIRKS)
        out.extend(persona_conversations(char, tone, quirk, R.randint(4, 7)))
    return out
