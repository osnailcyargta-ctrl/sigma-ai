"""v9 TICTACTOE BOARD PLAY — the model itself plays, no preset.

Training format (exactly what the browser will send):
  <user>board: x.o..x... you are o. your move.<ai><think>{real analysis}</think>{move}.

- Every REACHABLE position with o to move is enumerated from the empty board
  (x always moves first — same as the app).
- The move label comes from a from-scratch minimax: only game-theoretically
  optimal moves are labeled, ties broken randomly so the play doesn't look
  scripted.
- The <think> trace is generated from the REAL position features (win-in-1,
  block, fork, center/corner doctrine, endgame) — so the model's visible
  thinking is genuinely about the board it sees.
In the browser, digits of occupied cells are banned from sampling (legality
constraint) and the model still picks which empty cell to take: the move is
the model's own decision, learned from these weights."""
import random
from itertools import product

R = random.Random(3131)

LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(b):
    for a,c,d in LINES:
        if b[a] != "." and b[a] == b[c] == b[d]:
            return b[a]
    return "draw" if all(x != "." for x in b) else None

def minimax(b, is_o, depth=0):
    w = winner(b)
    if w == "o": return 10 - depth
    if w == "x": return depth - 10
    if w == "draw": return 0
    vals = []
    for i in range(9):
        if b[i] == ".":
            b[i] = "o" if is_o else "x"
            vals.append(minimax(b, not is_o, depth+1))
            b[i] = "."
    return max(vals) if is_o else min(vals)

def optimal_moves(b):
    """all moves for 'o' that keep the game-theoretic value"""
    best, bv = [], -99
    for i in range(9):
        if b[i] == ".":
            b[i] = "o"
            v = minimax(b, False, 1)
            b[i] = "."
            if v > bv: bv, best = v, [i]
            elif v == bv: best.append(i)
    return best

def _threat_cells(b, p):
    """cells that complete a line for player p (p has 2, one empty)"""
    out = []
    for a,c,d in LINES:
        tri = (b[a], b[c], b[d])
        if tri.count(p) == 2 and tri.count(".") == 1:
            out.append((a,c,d)[tri.index(".")])
    return out

def _fork_moves(b, p):
    """empty cells that create two simultaneous threats for p"""
    out = []
    for i in range(9):
        if b[i] != ".": continue
        b[i] = p
        if len(_threat_cells(b, p)) >= 2: out.append(i)
        b[i] = "."
    return out

def _line_name(a,c,d):
    rows = {0:"top row",1:"middle row",2:"bottom row"}
    cols = {0:"left column",1:"middle column",2:"right column"}
    if a//3 == c//3: return rows[a//3]
    if a%3 == c%3: return cols[a%3]
    return "the diagonal"

def analysis(b, mv, rng):
    """REAL reasoning text for the chosen move, computed from the position."""
    n_empt = b.count(".")
    # 1) winning move?
    b2 = b[:]; b2[mv] = "o"
    if winner(b2) == "o":
        ln = _line_name(*next((a,c,d) for a,c,d in LINES if b2[a]==b2[c]==b2[d]=="o"))
        return rng.choice([f"i win at {mv} — that closes {ln}. done.",
                           f"{mv} finishes {ln}. three for me. game over.",
                           f"they left {ln} open. i take {mv} and win."])
    # 2) must block?
    threats = _threat_cells(b, "x")
    if mv in threats:
        ln = _line_name(*next((a,c,d) for a,c,d in LINES
                              if (b[a],b[c],b[d]).count("x") == 2 and (b[a],b[c],b[d]).count(".") == 1 and mv in (a,c,d)))
        return rng.choice([f"they threaten {ln}. i block {mv}. no line for them.",
                           f"two in a row for them on {ln} — {mv} shuts it.",
                           f"block first. {mv} kills their {ln}."])
    # 3) creating a fork?
    b3 = b[:]; b3[mv] = "o"
    if len(_threat_cells(b3, "o")) >= 2:
        return rng.choice([f"{mv} opens two lines at once. a fork — they can't cover both.",
                           f"fork plan: {mv} gives me two winning squares next turn.",
                           f"i play {mv} and now two threats live. pick one, they lose the other."])
    # 4) preventing their fork?
    xf = _fork_moves(b, "x")
    if xf:
        return rng.choice([f"their fork would start at {xf[0] if mv in xf else 'one of these squares'}. "
                           f"i take {mv} and force them to defend instead.",
                           f"i see the double threat coming. {mv} breaks it before it exists."])
    # 5) doctrine: center / corners
    if mv == 4:
        return rng.choice(["center first. it touches four lines.",
                           "the middle square is worth the most. taking 4.",
                           "4 — strongest point on the board."])
    if mv in (0,2,6,8):
        return rng.choice([f"corner {mv}. corners control the diagonals.",
                           f"i value corners over sides. {mv} it is.",
                           f"{mv} keeps two lines open from a corner."])
    if n_empt <= 3:
        return rng.choice([f"only {n_empt} squares left. {mv} is the safest.",
                           f"endgame. {mv} can't hurt me.",
                           f"few squares left — {mv} holds the line."])
    return rng.choice([f"{mv} keeps my options open on two lines.",
                       "quiet move. development over tricks.",
                       f"nothing to punish yet. {mv} stays flexible."])

def board_str(b):
    return "".join(b)

def render(pos_b, mv, persona, rng):
    think = analysis(pos_b, mv, rng)
    move_tok = f"{mv}."
    b = board_str(pos_b)
    return {"persona": persona, "desc": None, "block": None,
            "turns": [(f"board: {b} you are o. your move.", f"<think>{think}</think>{move_tok}")],
            "kind": "ttt"}

def enumerate_positions():
    """DFS the real game tree; yield (board, o_to_move) for every reachable node."""
    seen, out = set(), []
    def dfs(b, o_turn):
        key = ("".join(b), o_turn)
        if key in seen: return
        seen.add(key)
        w = winner(b)
        if w is not None: return
        if o_turn: out.append(b[:])
        for i in range(9):
            if b[i] == ".":
                b[i] = "o" if o_turn else "x"
                dfs(b, not o_turn)
                b[i] = "."
    dfs(["."]*9, False)   # x moves first in the app
    return out

_cache = None
def all_positions():
    global _cache
    if _cache is None: _cache = enumerate_positions()
    return _cache

def generate_ttt_sessions(n_target):
    poss = all_positions()
    out = []
    # full coverage first: every position at least twice (different phrasing)
    for b in poss:
        mv = R.choice(optimal_moves(b[:]))
        persona = "sigma" if R.random() < 0.7 else "fake"
        out.append(render(b, mv, persona, R))
        mv2 = R.choice(optimal_moves(b[:]))
        out.append(render(b, mv2, persona, R))
    # extras random
    while len(out) < n_target:
        b = R.choice(poss)
        mv = R.choice(optimal_moves(b[:]))
        persona = "sigma" if R.random() < 0.7 else "fake"
        out.append(render(b, mv, persona, R))
    return out[:n_target]

if __name__ == "__main__":
    poss = all_positions()
    print(f"reachable o-to-move positions: {len(poss)}")
    b = [".",".",".",".","x",".",".",".","."]
    print("optimal for:", "".join(b), "->", optimal_moves(b[:]))
    b = ["x","x",".",".","o",".",".",".","."]
    print("optimal for:", "".join(b), "->", optimal_moves(b[:]))
    for s in generate_ttt_sessions(6):
        u, a = s["turns"][0]
        print("U:", u)
        print("A:", a)
