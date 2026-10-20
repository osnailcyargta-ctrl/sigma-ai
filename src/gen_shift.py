"""v11 POSITION-SHIFT augmentation — the ctx-2048 pos table (linear-
interpolated 256 -> 2048) leaves positions >256 under-trained for FREE-RUNNING
generation: teacher-forced NLL there is fine, but think+reply clusters drift
into letter soup. Fix: re-issue the PRECISION families (copy/memory/lists)
with a long filler prefix so their think+reply clusters sit at positions
300-1900 from token 0. Dense far-position training mass, cheap to build."""
import random
sys_path = None
import gen_long as gl

R = random.Random(555001)

def shift_prefix(n_turns):
    """filler turns that carry NO facts to remember — pure position padding"""
    turns = []
    fill = list(gl.FILLER_QA)
    R.shuffle(fill)
    for u, a in fill[:n_turns // 2]:
        turns.append((u, a))
    # a story arc pad (no names inside)
    story = R.choice(gl.STORIES)
    title, parts = story
    k = R.randint(2, len(parts))
    turns.append((f"tell me a story about {title}", parts[0]))
    for p in parts[1:k]:
        turns.append((R.choice(["go on", "continue", "and then?"]), p))
    while len(turns) < n_turns:
        u, a = R.choice(gl.FILLER_QA)
        turns.append((u, a))
    return turns[:n_turns]

def shift_sessions(base_sessions, n_out):
    """prefix base short sessions with filler so replies land far; the base
    session's OWN turns (facts + recall) stay verbatim after the prefix."""
    out = []
    for s in base_sessions:
        if len(out) >= n_out: break
        k = R.randint(5, 15)
        turns = shift_prefix(k) + list(s["turns"])
        out.append({"persona": s.get("persona") or "sigma", "turns": turns})
    return out
