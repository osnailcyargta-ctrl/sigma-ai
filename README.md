# SIGMA — a real AI built 100% from scratch

No pretraining. No datasets downloaded. No template bots. Every weight in this repo
was trained from zero on a corpus generated from zero, on a 2-core CPU.

## What this is

| Checkpoint | Params | Status |
|-----------|--------|--------|
| `10m` | 10,388,272 | trained — eval ppl 1.47, game-intent acc ~100% |
| `17m` | 17,337,632 | trained — eval ppl 1.48, game-intent acc ~100% |
| `20m` | 20,088,768 | trained — think traces + interrupt discipline, intent acc 100% |
| `22m` | 22,276,040 | **flagship** — char-copy memory (open-vocabulary), think traces, interrupt discipline, eval ppl 1.19, intent acc 100% |

(2M and 5M checkpoints were retired from the repo at the 20M milestone.)

- **Architecture**: decoder-only transformer (GPT-style) — learned token + positional
  embeddings, pre-LN blocks, causal multi-head self-attention, GELU MLPs, tied LM head.
  Implemented by hand in `src/model.py` on top of raw tensor ops. Exported to ONNX
  with a **KV-cache graph** (prefill once, then one token per forward) so the browser
  streams tokens in real time; every export is verified token-identical against the
  full-forward PyTorch decode.
- **Thinking**: new checkpoints emit `<think>…</think>` private plans before every
  reply — trained on ~5M tokens of thought traces that restate the actual request,
  the exact resume point of paused lists/stories, game modes, and memory operations.
  The web app streams the plan live into a Z.ai-style expandable panel, then streams
  the answer token-by-token. Older checkpoints show real telemetry instead
  (intent probabilities, recalled memories, token counter).
- **Anti-fly discipline**: interrupt families teach the model that a paused story or
  list does NOT swallow unrelated messages — answer the new question first, then
  "continue" still resumes the original task exactly where it stopped.
- **Data**: `src/gen_chat.py`, `src/gen_memory.py`, `src/personas_gen.py`,
  `src/gen_think.py` synthesize ~128k conversations (~13M tokens): greetings,
  opinions, advice, math, jokes, facts, memory recall, task interruption
  (stop → continue), pivots with exact resume, game invites and invite *negatives*,
  two preset personalities (fake-smile yes-man, blunt sigma) and ~2,600 compositional
  custom personas (character × tone × quirk).
- **Tokenizer**: word-level, built from the corpus itself (`src/tokenizer.py`,
  ported 1:1 to JS in `docs/ai.js`). `vocab3.json` keeps every original id stable,
  appends `<think>`/`</think>` and — the v4 upgrade — **every missing single letter**.
  Any out-of-vocab word (a name, a gamertag, a city, a pet, anything you type) is
  now spelled letter-by-letter, so the model can literally READ and COPY arbitrary
  user text. No more canned-name hallucinations: say "my name is agusi" and it
  echoes, stores, and recalls **agusi** — not the most frequent name in its
  training pool. Old checkpoints stay compatible (they clamp the new ids).
- **Char-copy memory**: the 22m flagship is trained on 244 diverse names and
  open-vocabulary facts (pets, cities, bands, gamertags) where the assistant must
  copy the char-spelled value from context — echo, `<mem>` write, and recall are
  all exact copies. Verified on unseen names: agusi, riley, zerofrost — all
  perfect echo + recall.
- **Heads**: LM head + a game-intent classifier head (5 classes) sharing the trunk —
  the AI *learns* whether "tic tac toe is for kids" is different from
  "how about a quick tic tac toe duel". No keyword matching anywhere.
- **Memory**: 3 layers — (1) full chat context window, (2) long-term vector store in
  the browser (IndexedDB) using the model's own pooled embeddings + cosine retrieval
  (strict 0.45 threshold), (3) learned `<mem>` markers the model emits to save
  important facts — stored, retrieved, and quoted back later.

## Repo layout

```
src/            model, tokenizer, data generators, trainer, chat tester, exporter
data/           vocab.json, vocab2.json, stats.json (regenerable via src/data_gen.py)
out/            torch checkpoints (.pt) + training meta
docs/           the web app (GitHub Pages) + models/*.onnx
```

## Train it yourself

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python src/data_gen.py
python src/train.py 22m 4000 9 2e-4    # chunked, resumable
python src/chat_test.py 22m            # real transcript, no mocks
python src/export_onnx.py 22m          # KV-cache graph, numerically verified
```

## Web app (PWA)

`docs/` is served via GitHub Pages. Install it, and the Service Worker caches the
model — after the first load it works offline and never re-downloads weights.
