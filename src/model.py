"""GPT-style Transformer built 100% from scratch (no pretrained anything).
Decoder-only: token+positional embeddings, pre-LN blocks, causal self-attention,
GELU MLP, tied LM head, plus: intent head + mean-pooled embedding output
(used in-browser for semantic long-term memory retrieval).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    def __init__(self, vocab_size, d_model, n_layers, n_heads, block_size=256, n_intent=5):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.n_intent = n_intent

    def n_params(self):
        # exact count incl. embeddings, pos, blocks, final LN, tied head not double-counted
        p = self.vocab_size * self.d_model            # tok emb (tied with head)
        p += self.block_size * self.d_model           # pos emb
        per = 4 * self.d_model * self.d_model         # attn qkvo
        per += 2 * self.d_model * 4 * self.d_model    # mlp
        per += 2 * self.d_model                       # 2 layernorms
        p += per * self.n_layers
        p += 2 * self.d_model                         # final LN
        p += self.d_model * self.n_intent             # intent head
        return p

def configs(vocab=1446, block=256):
    """Sizes tuned to land close to the parameter targets with the real vocab."""
    return {
        "2m":  Config(vocab, 170, 5, 5, block),
        "5m":  Config(vocab, 256, 6, 8, block),
        "10m": Config(vocab, 368, 6, 8, block),
        "17m": Config(vocab, 416, 8, 8, block),
    }

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.n_heads = cfg.n_heads
        self.d = cfg.d_model
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self.intent_head = nn.Linear(cfg.d_model, cfg.n_intent, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None, intent_labels=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        pooled = x.mean(dim=1)                       # sentence embedding (memory retrieval)
        intent_logits = self.intent_head(pooled)

        out = {"logits": logits, "pooled": pooled, "intent_logits": intent_logits}
        if intent_labels is not None:
            out["intent_loss"] = F.cross_entropy(intent_logits, intent_labels)
        if targets is not None:
            lm = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                 targets[:, 1:].reshape(-1), reduction="none")
            lm = lm.clamp(min=0)
            lm[y_ignore := (targets[:, 1:] == -100).reshape(-1)] = 0.0
            if loss_mask is not None:
                m = loss_mask[:, 1:].reshape(-1).float()
                m = m * (~y_ignore).float()
                lm = (lm * m).sum() / m.sum().clamp(min=1)
            else:
                lm = lm[~y_ignore].mean()
            out["lm_loss"] = lm
        return out

    @torch.no_grad()
    def generate(self, idx, max_new=48, temperature=0.75, top_k=25, top_p=0.9,
                 stop_tokens=None, allow_nl=True):
        self.eval()
        stop = set(stop_tokens or [])
        for _ in range(max_new):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)["logits"][:, -1, :]
            logits[:, 3] = -1e9 if len(stop) else logits[:, 3]  # <eos> only if stopping
            for t in stop:
                logits[:, t] = -1e9
            if not allow_nl:
                logits[:, self.stoi_nl()] = -1e9
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(-1)
                drop = cum - sp > top_p
                sp[drop] = 0
                probs = torch.zeros_like(probs).scatter(1, si, sp)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            t = int(nxt)
            if t in stop:
                break
        return idx

    def stoi_nl(self):
        return 6  # "<nl>" index in SPECIALS
