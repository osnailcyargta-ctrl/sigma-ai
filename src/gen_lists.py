"""v9 LIST + COMMAND families — the two live failures on 50M:

1. 'give me 3 dog names' after a name was set -> model echoed the name.
   Fix: a real LIST family. The model must answer the CURRENT request with
   N items of the requested category, never with recalled memory.

2. 'stop' / 'shut the fuck up' -> model answered '<garbled name>. of course'.
   Fix: a COMMAND family. stop = stop ack; rude shutdowns = quiet acks.
   NEVER a name, NEVER a game, NEVER a memory echo.

Most sessions here also carry a background <memory> block (via gen_bgmem
threading in data_gen), so the model learns: block present + unrelated
request -> just answer the request."""
import random
from gen_copy import rand_name

R = random.Random(9091)

# categories the user can ask lists of
DOG_POOL = ["buddy","max","bella","luna","charlie","rocky","daisy","milo","cooper","bailey",
            "toby","bear","zed","kilo","apollo","nova","juno","tank","pixel","mango",
            "brutus","waffles","mochi","zuko","pepper","nugget","biscuit","wasabi","kombu","tofu",
            "rex","shadow","gizmo","oreo","panda","boba","mochi","coco","lucky","ace"]
CAT_POOL = ["miso","sushi","pepper","smokey","olive","pixel","nori","tofu","bagel","waffle",
            "mochi","cleo","nala","fig","bean","pumpkin","miso","juno","onyx","clover"]
MOVES = ["inception","the matrix","parasite","whiplash","spider-man","interstellar","gone girl",
         "the dark knight","get out","mad max","up","coco","ratatouille","jaws","alien","heat"]
SONGS = ["blinding lights","take five","bohemian rhapsody","smells like teen spirit",
         "one more time","feels like we only go backwards","dreams","redbone","snow",
         "nikes","weight in gold","electric feel","midnight city","instant crush"]
FOODS = ["sushi","burger","ramen","tacos","pizza","satay","rendang","pho","fried rice",
         "bakso","soto","nasi goreng","steak","dumplings","laksa","bbq ribs"]
COLORS = ["teal","crimson","cobalt","amber","olive","coral","indigo","charcoal","mint","plum"]
GAMES_I = ["chess","sudoku","wordle","2048","minesweeper","solitaire","tetris","snake",
           "checkers","dominoes"," Uno","scrabble","battleship","connect four"]
TEAMS = ["thunder","vipers","comets","wolves","titans","phoenix","cyclones","sharks",
         "falcons","jaguars","riptides","blaze","yetis","pirates","storm","rhinos"]
USERN = ["shadowfax","neonbyte","pixelpunk","voidwalker","turbo_snail","glitchwitch",
         "midnightowl","cosmojelly","rustedhalo","quantumqat","lazertaco","ducklord"]
STORY_T = ["a heist on a space station","a dragon who hates fire","time-loop romance",
           "detective in a city where it never stops raining","robot learning to paint",
           "a cooking contest between gods","post-apocalyptic gardening","ghost roommate"]
PICKUP = ["are you a wifi signal? because i'm feeling a connection",
          "do you have a map? i keep getting lost in this chat",
          "i'm not a photographer but i can picture us talking forever",
          "are you a keyboard? because you're just my type"]

def _n_items(rng):
    return rng.choice([2, 3, 3, 3, 4, 5])

def _list_reply(items, persona, rng):
    body = ", ".join(items)
    if persona == "fake":
        return rng.choice([f"Omg easy!! {body.capitalize()}!!",
                           f"Say less!! {body.capitalize()}!!",
                           f"Here you go!! {body.capitalize()}!!"])
    return rng.choice([f"{body.capitalize()}. Take them.",
                       f"{body.capitalize()}.",
                       f"Quick list: {body}."])

_ASKS = {
    "dog": ["give me {} dog names", "{} dog names, go", "name {} dogs for me",
            "i need {} dog names", "give me {} names for a dog", "help me name my dog, {} options",
            "{} good dog names right now"],
    "cat": ["give me {} cat names", "{} cat names, quick", "name {} cats for me",
            "i need {} cat names", "{} cat name ideas"],
    "movie": ["give me {} movie ideas", "recommend {} movies", "{} movies to watch tonight",
              "name {} good movies", "give me {} films for tonight"],
    "song": ["give me {} songs", "recommend {} songs", "{} songs for my playlist",
             "name {} tracks i should hear", "{} song recs"],
    "food": ["give me {} food ideas", "i'm hungry, {} foods i should try", "name {} good foods",
             "{} dinner ideas", "give me {} foods to order"],
    "color": ["give me {} color ideas", "name {} nice colors", "{} colors for a logo",
              "i need {} color options"],
    "game": ["give me {} game ideas", "name {} games i should try", "{} games to kill time",
             "recommend {} games"],
    "team": ["give me {} team names", "name {} teams for my squad", "{} team name ideas",
             "i need {} team names"],
    "username": ["give me {} username ideas", "name {} cool usernames", "{} username ideas please",
                 "i need {} gamertags"],
    "story": ["give me {} story ideas", "name {} story concepts", "{} ideas for a story",
              "i need {} story prompts"],
    "pickup": ["give me {} pickup lines", "hit me with {} pickup lines", "{} pickup lines, go",
               "name {} smooth pickup lines"],
}

_POOL_OF = {"dog": DOG_POOL, "cat": CAT_POOL, "movie": MOVES, "song": SONGS, "food": FOODS,
            "color": COLORS, "game": GAMES_I, "team": TEAMS, "username": USERN, "story": STORY_T,
            "pickup": PICKUP}

def fam_list(persona, rng):
    """THE fix for 'give me 3 dog names -> echoed the user's name'.
    The answer is ALWAYS the requested list. Category/number are real;
    a chunk of the items are fresh OOV strings so lists stay open-vocab too."""
    cat = rng.choice(list(_ASKS.keys()))
    n = _n_items(rng)
    pool = list(_POOL_OF[cat])
    n = min(n, len(pool))                    # never promise more than we deliver
    u = rng.choice(_ASKS[cat]).format(n)
    rng.shuffle(pool)
    items = pool[:n]
    # sprinkle OOV items in ~35% of lists (open-vocabulary lists, not a closed pool)
    if rng.random() < 0.35 and len(items) > 0:
        k = rng.randint(1, max(1, len(items) // 2))
        for i in range(k):
            items[rng.randrange(len(items))] = rand_name(rng)
    a = _list_reply(items, persona, rng)
    turns = [(u, a)]
    # a second, different list request in the same chat (binding: answer the CURRENT ask)
    if rng.random() < 0.5:
        cat2 = rng.choice([c for c in _ASKS if c != cat])
        n2 = _n_items(rng)
        u2 = rng.choice(_ASKS[cat2]).format(n2)
        pool2 = list(_POOL_OF[cat2]); rng.shuffle(pool2)
        a2 = _list_reply(pool2[:n2], persona, rng)
        turns.append((u2, a2))
    # then a recall is fine too — but only when actually asked
    if rng.random() < 0.4:
        turns.append((rng.choice(["whats my name?", "do you remember my name?"]), None))  # patched by caller
    return turns

_STOP_ACK = {
    "sigma": ["Stopped. Say continue if something was running.",
              "Halted. Nothing is running now.",
              "Paused. What's next?"],
    "fake": ["Okay!! Stopped!!",
             "Stopping now!! What's next??",
             "Done!! Everything's on pause!!"],
}
_RUDE_ACK = {
    "sigma": ["Fair. Going quiet.",
              "Understood. I'll hold.",
              "Alright. I'm here when you want me.",
              "Noted. Silence it is."],
    "fake": ["Omg okay!! Quiet mode!!",
             "I'll hush!! Say when!!",
             "Sorry!! Zippering my mouth now!!"],
}

def fam_commands(persona, rng):
    """'stop' with nothing running, rude shutdowns, quiet-then-talk.
    The ONLY correct behavior: acknowledge, stay quiet, never echo memory."""
    u1 = rng.choice(["stop", "ok stop", "stop", "stop it", "stop talking", "shut up",
                     "shut the fuck up", "stfu", "be quiet", "quiet", "hush",
                     "shut your mouth", "stop pls"])
    rude = any(k in u1 for k in ("shut", "stfu", "quiet", "hush"))
    a1 = rng.choice(_RUDE_ACK[persona] if rude else _STOP_ACK[persona])
    turns = [(u1, a1)]
    if rng.random() < 0.6:
        u2 = rng.choice(["ok you can talk now", "alright im back", "hi", "you there?",
                         "ok sorry, continue"])
        a2 = ("Back. What do you need?" if persona == "sigma" else "I'm here!! Miss me??")
        turns.append((u2, a2))
    return turns

def generate_lists(n_target):
    out = []
    for _ in range(n_target):
        persona = "fake" if R.random() < 0.5 else "sigma"
        fam = fam_commands if R.random() < 0.42 else fam_list
        turns = fam(persona, R)
        # drop the None placeholder turns (recall patched elsewhere with real names)
        turns = [(u, a) for u, a in turns if a is not None]
        if not turns:
            continue
        out.append({"persona": persona, "desc": None, "block": None, "turns": turns, "kind": "lists"})
    return out

if __name__ == "__main__":
    for s in generate_lists(8):
        print("---", s["persona"])
        for u, a in s["turns"]:
            print("  U:", u)
            print("  A:", a)
