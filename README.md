# SIGMA — a real AI built 100% from scratch

No pretraining. No datasets downloaded. No template bots. Every weight in this repo
was trained from zero on a corpus generated from zero, on a 2-core CPU.

## What this is

| Checkpoint | Params | Status |
|-----------|--------|--------|
| `2m`  | 2,027,930  | trained — eval ppl 1.76, game-intent acc ~100% |
| `5m`  | 5,162,240  | trained — eval ppl 1.55, game-intent acc ~100% |
| `10m` | 10,388,272 | trained — eval ppl 1.47, game-intent acc ~100% |
| `17m` | ~17M | queued |

- **Architecture**: decoder-only transformer (GPT-style) — learned token + positional
  embeddings, pre-LN blocks, causal multi-head self-attention, GELU MLPs, tied LM head.
  Implemented by hand in `src/model.py` on top of raw tensor ops.
- **Data**: `src/gen_chat.py`, `src/gen_memory.py`, `src/personas_gen.py` synthesize
  ~130k conversations (7.7M tokens): greetings, opinions, advice, math, jokes, facts,
  insults, love, memory recall, task interruption (stop → continue), game invites and
  game-invite *negatives*, two preset personalities (fake-smile yes-man, blunt sigma)
  and ~2,600 compositional custom personas (character × tone × quirk).
- **Tokenizer**: word-level, built from the corpus itself (`src/tokenizer.py`,
  ported 1:1 to JS in `docs/ai.js`).
- **Heads**: LM head + a game-intent classifier head (5 classes) sharing the trunk —
  the AI *learns* whether "tic tac toe is for kids" is different from
  "how about a quick tic tac toe duel". No keyword matching anywhere.
- **Memory**: 3 layers — (1) full chat context window, (2) long-term vector store in
  the browser (IndexedDB) using the model's own pooled embeddings + cosine retrieval,
  (3) learned `<mem>` markers the model emits to save important facts.

## Repo layout

```
src/            model, tokenizer, data generators, trainer, chat tester, exporter
data/           vocab.json, stats.json (corpus is regenerable via src/data_gen.py)
out/            torch checkpoints (.pt) + training meta
docs/           the web app (GitHub Pages) + models/*.onnx
```

## Train it yourself

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python src/data_gen.py
python src/train.py 2m 2400 9 3e-4     # chunked, resumable
python src/chat_test.py 2m             # real transcript, no mocks
python src/export_onnx.py 2m
```

## Web app (PWA)

`docs/` is served via GitHub Pages. Install it, and the Service Worker caches the
model — after the first load it works offline and never re-downloads weights.
