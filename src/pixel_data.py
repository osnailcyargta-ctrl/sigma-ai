"""PIXEL GEN dataset — 32x32 pixel art, 16-color palette, 6 categories.
Procedural but HIGHLY randomized: every image is unique (the diffusion model
learns the family style, then invents its own from noise at sample time).
This file is the single source of truth for the PALETTE — pixel.js mirrors it.
Usage: python src/pixel_data.py  -> data/pixel_train.npy (N,32,32,3 uint8)
"""
import os, sys, numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

# 16-color pixel-art palette (RGB) — mirrored 1:1 in docs/pixel.js
PALETTE = [
    (  0,   0,   0),  # 0 outline black
    (255, 255, 255),  # 1 white
    (212, 212, 212),  # 2 light gray
    (122, 122, 122),  # 3 gray
    ( 46,  46,  56),  # 4 dark navy
    ( 27,  58, 107),  # 5 blue
    ( 63, 127, 204),  # 6 sky blue
    (126, 200, 227),  # 7 light blue
    (217,  79,  79),  # 8 red
    (242, 163,  60),  # 9 orange
    (247, 217,  76),  # 10 yellow
    (106, 176,  76),  # 11 green
    ( 46, 125,  79),  # 12 dark green
    (142,  90,  59),  # 13 brown
    (242, 201, 160),  # 14 sand/skin
    (155,  89, 182),  # 15 purple
]
P = len(PALETTE)
CLASSES = ["creature", "sunset", "spaceship", "flower", "castle", "gem"]
N_CLASSES = len(CLASSES)

def blank():
    img = np.full((32, 32), -1, dtype=np.int8)   # -1 = untouched background
    return img

def outline(img, col=0):
    """add a 1px dark outline around any non-background pixel blob"""
    bg = img == -1
    m = np.zeros((32, 32), dtype=bool)
    for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)):
        sh = np.roll(np.roll(img != -1, dy, 0), dx, 1)
        m |= sh
    img[m & bg] = col
    return img

def mirror_x(img):
    """symmetric sprites: left half mirrored to right"""
    img[:, 16:] = img[:, 15::-1]
    return img

def put(img, x, y, c):
    if 0 <= x < 32 and 0 <= y < 32: img[y, x] = c

def disc(img, cx, cy, r, c, fill=True):
    for y in range(32):
        for x in range(32):
            d2 = (x-cx)**2 + (y-cy)**2
            if d2 <= r*r and (fill or d2 > (r-1.4)**2): img[y, x] = c

def gen_creature(rng):
    img = blank()
    body = rng.choice([8, 9, 11, 6, 5, 15, 13, 3])
    dark = rng.choice([4, 0, 12, 5])
    w = rng.integers(5, 10)            # half-width of body
    h = rng.integers(7, 13)
    cx = rng.integers(9, 23)
    top = 30 - h
    for y in range(top, 30):
        # bulge profile: narrower at top and bottom
        t = (y - top) / max(1, h - 1)
        ww = int(w * (0.55 + 0.7 * np.sin(np.pi * min(1, max(0, t)))**0.8))
        for x in range(cx - ww, cx + ww + 1):
            if rng.random() < 0.94: put(img, x, y, body)
    # legs
    for lx in (cx - w + 1, cx + w - 1):
        for y in range(29, 31):
            put(img, lx + rng.integers(-1, 2), y, dark)
    # eyes (white + pupil)
    ey = top + h // 3
    for ex in (cx - max(2, w//3), cx + max(2, w//3)):
        put(img, ex, ey, 1); put(img, ex + rng.choice([-1, 0, 1]), ey, 0)
    # mouth
    my = top + int(h * 0.62)
    for x in range(cx - 2, cx + 3): put(img, x, my, dark)
    # belly patch / spots
    if rng.random() < 0.6:
        sc = rng.choice([1, 14, 7])
        disc(img, cx + rng.integers(-2, 3), top + h//2 + rng.integers(-1, 2),
             rng.integers(2, 4), sc)
    return outline(mirror_x(img))

def gen_sunset(rng):
    img = blank()
    horizon = rng.integers(16, 22)
    sky = [rng.choice([8, 9, 10]), rng.choice([9, 15, 8]), rng.choice([5, 4, 6])]
    bands = [0, horizon//3, 2*horizon//3]
    for y in range(horizon):
        c = sky[2] if y >= bands[2] else sky[1] if y >= bands[1] else sky[0]
        for x in range(32): img[y, x] = c
    # sun
    disc(img, rng.integers(8, 24), rng.integers(4, horizon - 4),
         rng.integers(3, 5), rng.choice([10, 1, 9]))
    # hills (2 layers)
    h1 = rng.integers(2, 5)
    for x in range(32):
        hh = h1 + int(2.2 * np.sin(x / rng.uniform(3, 6) + rng.uniform(0, 6)))
        for y in range(horizon - hh, horizon): img[y, x] = 12
    # water
    wcol = rng.choice([5, 6, 4])
    for y in range(horizon, 32):
        for x in range(32): img[y, x] = wcol
        if rng.random() < 0.7:   # glints
            for _ in range(rng.integers(1, 4)):
                put(img, rng.integers(0, 32), y, rng.choice([1, 7, 10]))
    return img

def gen_spaceship(rng):
    img = blank()
    hull = rng.choice([2, 3, 1, 4, 5])
    acc = rng.choice([8, 6, 15, 10])
    dark = rng.choice([4, 0])
    cy = rng.integers(12, 20)
    half = rng.integers(6, 11)
    # fuselage: lens shape
    for y in range(cy - 3, cy + 4):
        ww = int(half * (1 - ((y - cy) / 3.4) ** 2))
        for x in range(16 - ww, 16 + ww + 1): put(img, x, y, hull)
    # cockpit
    disc(img, 16, cy - 1, rng.integers(2, 3), acc)
    # wings (symmetric)
    wy = cy + rng.integers(-1, 2)
    for i in range(rng.integers(4, 8)):
        y = wy + i
        ww = half + 2 + i * rng.integers(0, 2)
        if y >= 31: break
        for x in (16 - ww, 16 + ww): put(img, x, y, acc)
    # engines
    for ex in (16 - half + 1, 16 + half - 1):
        put(img, ex, cy + 4, dark); put(img, ex, cy + 5, 9); put(img, ex, cy + 6, 10)
    return outline(mirror_x(img))

def gen_flower(rng):
    img = blank()
    cx = rng.integers(10, 22)
    cy = rng.integers(8, 13)
    pet = rng.choice([8, 15, 6, 9, 1])
    core = rng.choice([10, 9, 13])
    r = rng.integers(3, 5)
    # petals around center
    n = rng.integers(5, 9)
    for k in range(n):
        a = 2 * np.pi * k / n + rng.uniform(-0.15, 0.15)
        rr = r + rng.integers(0, 2)
        disc(img, int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a)),
             rng.integers(1, 3), pet)
    disc(img, cx, cy, 2, core)
    # stem + leaves
    stem = 12
    for y in range(cy + r, 31):
        put(img, cx + int(1.5 * np.sin(y / 5 + rng.uniform(0, 1))), y, stem)
    for s in (-1, 1):
        lx = cx + s * rng.integers(2, 4)
        ly = rng.integers(cy + 6, 26)
        for d in range(3):
            put(img, lx + s * d, ly + d - 1, 11); put(img, lx + s * (d + 1), ly + d, 11)
    return img

def gen_castle(rng):
    img = blank()
    wall = rng.choice([3, 2, 13, 4])
    dark = rng.choice([4, 0])
    base = 31
    wh = rng.integers(10, 16)      # wall height
    wt = rng.integers(6, 9)        # half width
    x0, x1 = 16 - wt, 16 + wt
    for y in range(base - wh, base):
        for x in range(x0, x1 + 1): put(img, x, y, wall)
    # battlements
    for x in range(x0, x1 + 1, 2): put(img, x, base - wh - 1, wall)
    # towers
    for tx in (x0 - 3, x1 + 3):
        th = wh + rng.integers(3, 7)
        for y in range(base - th, base):
            for x in range(tx - 2, tx + 3): put(img, x, y, wall)
        for x in range(tx - 3, tx + 4): put(img, x, base - th - 1, dark)
        put(img, tx, base - th, rng.choice([8, 9, 10]))   # flag
    # gate + windows
    gy = base - rng.integers(4, 6)
    for y in range(gy, base):
        for x in range(16 - 2, 16 + 3): put(img, x, y, dark)
    for wy in range(base - wh + 2, base - 4, 3):
        for wx in (x0 + 2, x1 - 2):
            put(img, wx, wy, rng.choice([10, 9]))
    return img

def gen_gem(rng):
    img = blank()
    col = rng.choice([6, 15, 8, 11, 10])
    hi = rng.choice([1, 7, 2])
    cy = rng.integers(13, 18)
    w = rng.integers(6, 10)
    # diamond: wide band + triangle top/bottom
    for y in range(cy - w, cy + w + 1):
        ww = w - abs(y - cy)
        for x in range(16 - ww, 16 + ww + 1): put(img, x, y, col)
    for y in range(cy - w, cy):
        ww = w - abs(y - cy)
        for x in range(16 - ww, 16 + ww + 1):
            if (x + y) % 2 == 0: put(img, x, y, hi)
    # facets
    for x in (16 - w + 1, 16 + w - 1): put(img, x, cy, hi)
    disc(img, 16, cy, 2, hi)
    return outline(mirror_x(img))

GENS = [gen_creature, gen_sunset, gen_spaceship, gen_flower, gen_castle, gen_gem]

def generate(n=9000, seed=777):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    pal = np.array(PALETTE, dtype=np.uint8)
    for i in range(n):
        c = i % N_CLASSES
        idx = GENS[c](rng)
        idx[idx < 0] = rng.choice([1, 2, 7]) if c == 1 else 1   # background = white-ish
        out[i] = pal[idx]
    return out

if __name__ == "__main__":
    x = generate()
    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "pixel_train.npy"), x)
    print("saved", x.shape, "palette", P, "classes", CLASSES)
    # quick sanity: unique images
    print("unique:", len(np.unique(x.reshape(x.shape[0], -1), axis=0)))
