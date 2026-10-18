"""Word-level tokenizer built from scratch. Ported 1:1 to JavaScript for the browser."""
import json, re, os
from collections import Counter

SPECIALS = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<ai>", "<nl>",
    "<pers:fake>", "<pers:sigma>", "<pers:custom>",
    "<mem>", "</mem>", "<memory>", "</memory>",
]
PAT = re.compile(r"(<[^>]+>)|\s+|(\w+|[^\w\s])")

def tokenize_text(text):
    """Returns list of raw tokens (strings). Lowercases. Keeps special tokens intact."""
    text = text.lower()
    out = []
    for m in PAT.finditer(text):
        if m.group(1):  # special token like <mem>
            out.append(m.group(1))
        elif m.group(2):
            out.append(m.group(2))
        else:
            # whitespace between words -> explicit newline marker only if it was a newline
            pass
    # re-insert <nl> where original had newlines: handled by caller using <nl> in corpus text
    return out

class Tokenizer:
    def __init__(self, vocab=None):
        self.itos = list(vocab) if vocab else []
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, vocab_size=6000):
        cnt = Counter()
        for t in texts:
            cnt.update(tokenize_text(t))
        vocab = list(SPECIALS)
        for w, _ in cnt.most_common():
            if w in vocab:
                continue
            if len(vocab) >= vocab_size:
                break
            vocab.append(w)
        return cls(vocab)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in tokenize_text(text)]

    def decode(self, ids, skip_special=True, return_all=False):
        """decode(ids) -> clean text (mem/think spans hidden).
        decode(ids, return_all=True) -> (text, think, mems)."""
        drops = {"<pad>", "<bos>", "<user>", "<ai>",
                 "<memory>", "</memory>"} if skip_special else set()
        words = []
        in_mem = False
        in_think = False
        think_words = []
        for i in ids:
            w = self.itos[i] if 0 <= i < len(self.itos) else "<unk>"
            if w == "<mem>":
                in_mem = True
                continue
            if w == "</mem>":
                in_mem = False
                continue
            if w == "<think>":
                in_think = True
                continue
            if w == "</think>":
                in_think = False
                continue
            if in_mem:
                continue
            if in_think:
                if w not in drops and w not in SPECIALS:
                    think_words.append(w)
                continue
            if w in drops:
                continue
            words.append(w)
        def join(ws):
            out = ""
            for w in ws:
                if w == "<nl>":
                    out += "\n"
                elif w in SPECIALS:
                    continue
                elif w in {".", ",", "!", "?", ":", ";", "'", ")", "}", "%"} or w == "'s":
                    out += w
                else:
                    out += (" " if out and not out.endswith(("\n", "(", "{")) else "") + w
            return out.strip()
        text = join(words)
        if return_all:
            return text, join(think_words), []
        return text
