/* resolve asset base from this script's own URL so the standalone surfaces
 * (shakespeagma/, studio/) reuse the same runtime + cached weights */
const SIGMA_BASE = (document.currentScript && document.currentScript.src
  ? document.currentScript.src.replace(/ai\.js(\?.*)?$/, "")
  : "./");

/* SIGMA AI runtime: word tokenizer (1:1 port of the Python tokenizer) +
 * ONNX inference with a KV cache (prefill once, then 1 token per step —
 * this is what makes real token-by-token streaming possible on CPU),
 * intent head, pooled embeddings for memory, and a two-phase generate()
 * (thinking phase -> answer phase) with abort + repetition penalty. */

const SPECIALS = ["<pad>","<unk>","<bos>","<eos>","<user>","<ai>","<nl>",
  "<pers:fake>","<pers:sigma>","<pers:custom>","<mem>","</mem>","<memory>","</memory>"];

const TOK_RE = /(<[^>]+>)|\s+|(\w+|[^\w\s])/g;

function rawTokens(text){
  text = text.toLowerCase();
  const out = [];
  for (const m of text.matchAll(TOK_RE)) {
    if (m[1]) out.push(m[1]);
    else if (m[2]) out.push(m[2]);
  }
  return out;
}

class Tok {
  constructor(itos){ this.itos = itos; this.stoi = {}; itos.forEach((w,i)=>this.stoi[w]=i); }
  static async load(){ const itos = await (await fetch(SIGMA_BASE + "models/vocab3.json")).json(); return new Tok(itos); }
  /* one raw token -> ids. OOV ascii words are spelled letter-by-letter so the
   * model can READ and COPY arbitrary user text (names, tags, anything). */
  wordIds(w){
    const i = this.stoi[w];
    if (i != null) return [i];
    const u = this.stoi["<unk>"];
    if (/^[a-z0-9]+$/.test(w) && w.length > 1 && w.length <= 24){
      const out = [];
      for (const c of w){ const ci = this.stoi[c]; if (ci == null) return [u]; out.push(ci); }
      return out;
    }
    return [u];
  }
  encode(text){ const out = []; for (const w of rawTokens(text)) out.push(...this.wordIds(w)); return out; }
  /* decode generated ids -> {text, mems, think}
   * - <mem>..</mem>     -> collected into mems (hidden from text)
   * - <think>..</think> -> collected into think (rendered in the thinking panel)
   * - consecutive single letter/digit tokens merge back into words (agusi)
   *   and contraction tails glue after an apostrophe (i'd, i'm, we'll) */
  decode(ids){
    const drop = new Set(["<pad>","<bos>","<user>","<ai>","<memory>","</memory>","<eos>","<unk>","<nl>"]);
    const hardGlue = new Set([".",",","!","?",":",";",")","}","%","'s"]);
    let out = "", inMem = false, inThink = false;
    const mems = []; let curMem = "", think = "", curThink = "";
    const isSingle = (w) => w.length === 1 && /[a-z0-9]/.test(w);
    const emit = (dst, w) => {
      // returns [newDst, prevSingle, tight]
      if (w === "'" || w === "(" || w === "{") return [dst + w, false, w === "'"];
      if (hardGlue.has(w)) return [dst + w, false, false];
      const single = isSingle(w);
      if (single && (this._prevSingle || this._tight)) return [dst + w, !!this._prevSingle, false];
      if (this._tight && w.length <= 3 && /[a-z0-9]+/.test(w)) return [dst + w, false, false];
      const sep = dst && !dst.endsWith("\n") && !dst.endsWith("(") && !dst.endsWith("{") ? " " : "";
      return [dst + sep + w, single, false];
    };
    this._prevSingle = false; this._tight = false;
    for (const i of ids){
      const w = this.itos[i]; if (w == null) continue;
      if (w === "<mem>"){ inMem = true; curMem = ""; this._prevSingle=false; this._tight=false; continue; }
      if (w === "</mem>"){ inMem = false; if (curMem.trim()) mems.push(curMem.trim()); this._prevSingle=false; this._tight=false; continue; }
      if (w === "<think>"){ inThink = true; curThink = ""; this._prevSingle=false; this._tight=false; continue; }
      if (w === "</think>"){ inThink = false; if (curThink.trim()) think = curThink.trim(); this._prevSingle=false; this._tight=false; continue; }
      if (w === "<nl>"){
        this._prevSingle=false; this._tight=false;
        if (inThink) curThink += "\n"; else if (!inMem) out += "\n";
        continue;
      }
      if (inThink){ if (!drop.has(w) && !SPECIALS.includes(w)){ [curThink, this._prevSingle, this._tight] = emit(curThink, w); } continue; }
      if (inMem){ if (!drop.has(w)){ [curMem, this._prevSingle, this._tight] = emit(curMem, w); } continue; }
      if (drop.has(w)) continue;
      if (SPECIALS.includes(w)) continue;
      [out, this._prevSingle, this._tight] = emit(out, w);
    }
    if (inThink && curThink.trim()) think = curThink.trim();   // stream-open think
    if (inMem && curMem.trim()) mems.push(curMem.trim());
    return { text: out.trim(), mems, think };
  }
}

/* ---------------- model session ---------------- */
const MDIMS = {   // [n_layers, n_heads, d_model, vocab_rows] per checkpoint
  "2m": [5, 5, 170, 1446], "5m": [6, 8, 256, 1446], "10m": [6, 8, 368, 1446],
  "17m": [8, 8, 416, 1446], "20m": [8, 8, 448, 1538], "22m": [8, 8, 472, 1588],
  "50m": [10, 10, 640, 1588], "v10": [10, 10, 640, 1589], "v11": [10, 10, 640, 1589],
};
/* positional-embedding size per checkpoint: older models hallucinate beyond
 * their trained context, so every feed is tail-clamped to this. v10 (ctx 512)
 * doubles the window the chat can pack; v11 grows it to 2048 — the word-forge
 * flagship trained on long conversations that actually span the window. */
const BLOCKS = { "2m":256, "5m":256, "10m":256, "17m":256, "20m":256, "22m":256, "50m":256, "v10":512, "v11":2048 };

class Sigma {
  constructor(sess, tk, size){
    this.sess = sess; this.tk = tk; this.size = size;
    this.cacheK = null; this.cacheV = null;      // ORT tensors [L,1,H,T,Dh]
    this.hasThink = tk.stoi["<think>"] != null;  // new checkpoints speak <think>
    const md = MDIMS[size] || [8, 8, 416, 1446];
    this.L = md[0]; this.H = md[1]; this.Dh = md[2] / md[1]; this.V = md[3];
    this.block = BLOCKS[size] || 256;
  }

  /* clamp ids to this checkpoint's embedding table (newer vocab words -> <unk>) */
  clamp(ids){ const u = this.tk.stoi["<unk>"]; return ids.map(i => i < this.V ? i : u); }

  /* does this checkpoint have a trained embedding row for a persona token?
   * (<pers:shakespeare> is vocab id 1588 — older tables end at 1587) */
  supportsPers(name){ const i = this.tk.stoi["<pers:" + name + ">"]; return i != null && i < this.V; }

  static async load(size, onProgress){
    const tk = await Tok.load();
    const buf = await fetchModelBuf(size, onProgress);
    const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return new Sigma(sess, tk, size);
  }

  reset(){ this.cacheK = null; this.cacheV = null; }

  /* run one forward with the live cache; returns the full output dict and
   * rotates the cache to the returned presents.
   * ids: full prompt (prefill, cache empty) or [tok] (decode step).
   * The graph REQUIRES past_k/past_v — prefill feeds zero-length ones. */
  async runStep(ids){
    const feeds = { idx: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]) };
    if (!this.cacheK){
      feeds.past_k = new ort.Tensor("float32", new Float32Array(0), [this.L, 1, this.H, 0, this.Dh]);
      feeds.past_v = new ort.Tensor("float32", new Float32Array(0), [this.L, 1, this.H, 0, this.Dh]);
    } else {
      feeds.past_k = this.cacheK; feeds.past_v = this.cacheV;
    }
    const out = await this.sess.run(feeds);
    this.cacheK = out.present_k; this.cacheV = out.present_v;
    return out;
  }

  static lastRow(logits){
    const [ , T, V ] = logits.dims;
    return logits.data.subarray((T-1)*V, T*V);
  }

  /* one softmax sampling step over a logits row. Repetition penalty NEVER
   * hits single letter/digit tokens: the model spells OOV names letter-by-letter
   * (joninibanana -> j o n i n i ...), and penalizing repeated letters is what
   * turned real names into soup like 'kanin' / 'ionnpi'. Words still get
   * penalized so normal replies don't loop.
   * LETTER-RUN GREEDY: while the model is mid-spelling (last 2 emitted tokens
   * are single letters), decoding goes near-greedy — copying is a precision
   * task, sampling noise per letter compounds over 10+ letters. A triple
   * repeat guard prevents greedy letter loops ('ccc...'). */
  sample(row, {temp, topK, topP, ban, repPen, repWindow, gen}){
    const V = row.length;
    const isSing = (v) => { const w = this.tk.itos[v]; return w != null && w.length === 1 && /[a-z0-9]/.test(w); };
    let effTemp = temp, effTopK = topK;
    if (gen && gen.length >= 2 && isSing(gen[gen.length-1]) && isSing(gen[gen.length-2])){
      effTemp = 0.02; effTopK = 3;
      if (gen[gen.length-1] === gen[gen.length-2]){          // no 3rd identical letter
        ban = new Set(ban); ban.add(gen[gen.length-1]);
      }
      // cycle guard: period-2 letter loops (ananana, sasasa) — if the last four
      // letters are X,Y,X,Y, the next X would extend the cycle: ban it.
      if (gen.length >= 4){
        const [a, b, c, d] = gen.slice(-4);
        if (a === c && b === d && a !== b){
          ban = new Set(ban); ban.add(gen[gen.length-2]);
        }
      }
    }
    /* letter-run decoding is pure argmax (top-1 of the sorted kept list):
     * sampling even within top-3 flips tail letters of long names
     * (agusiargya -> agusarge). Copy precision beats variety here. */
    const argmax = effTemp <= 0.05;
    const z = new Float32Array(V);
    let mx = -Infinity;
    for (let v = 0; v < V; v++){
      let x = row[v];
      if (ban.has(v)) x = -1e9;
      if (repPen && gen && gen.length){
        let hit = false;
        for (let i = Math.max(0, gen.length - repWindow); i < gen.length; i++)
          if (gen[i] === v){ hit = true; break; }
        if (hit){
          if (!isSing(v)) x = x > 0 ? x / repPen : x * repPen;
        }
      }
      z[v] = x; if (x > mx) mx = x;
    }
    const probs = new Float32Array(V);
    let sum = 0;
    for (let v = 0; v < V; v++){ probs[v] = Math.exp(z[v] - mx); sum += probs[v]; }
    for (let v = 0; v < V; v++) probs[v] /= sum;
    const idxs = Array.from({length: V}, (_, v) => v);
    idxs.sort((a,b) => probs[b] - probs[a]);
    const kept = idxs.slice(0, effTopK);
    let cum = 0; const kept2 = [];
    for (const v of kept){ kept2.push(v); cum += probs[v]; if (cum >= topP) break; }
    const tot = kept2.reduce((s,v) => s + probs[v], 0);
    let r = Math.random() * tot, pick = kept2[0];
    for (const v of kept2){ r -= probs[v]; if (r <= 0){ pick = v; break; } }
    return argmax ? kept2[0] : pick;
  }

  /* full two-phase reply:
   *  promptIds — token ids ending with <ai>
   *  onThink(txt, done)   — streamed thinking text (new checkpoints only)
   *  onToken(txt, n)      — streamed answer text
   *  shouldAbort()        — polled between steps; true -> return partial
   *  ban                  — iterable of token ids banned ONLY in the answer
   *                         phase (tic tac toe legality: occupied cell digits);
   *                         the thinking phase stays free to mention them.
   * Old checkpoints have no <think> token: the app fills the thinking panel
   * with real telemetry (intent probs, recalled memories) instead. */
  async generate(promptIds, {maxThink=34, maxNew=70, temp=0.68, topK=22, topP=0.9,
                             repPen=1.28, repWindow=14, ban=null, onThink, onToken, shouldAbort}={}){
    this.reset();
    promptIds = this.clamp(promptIds);
    /* never feed more positions than the checkpoint's positional table */
    if (promptIds.length > this.block) promptIds = promptIds.slice(-this.block);
    const tk = this.tk;
    const stopIds = new Set([tk.stoi["<user>"], tk.stoi["<eos>"]]);
    /* <mem>/</mem> are NOT banned: the model is trained to write memories and
     * decode() routes them into .mems for long-term storage (hidden from text). */
    const hardBan = new Set([tk.stoi["<pad>"], tk.stoi["<unk>"],
                             tk.stoi["<memory>"], tk.stoi["</memory>"]]);
    const thinkTok = tk.stoi["<think>"], endThink = tk.stoi["</think>"];

    let row = Sigma.lastRow((await this.runStep(promptIds)).logits);
    const gen = [];                       // every generated id (for rep penalty)
    let thinkIds = [], answerIds = [];

    /* -------- phase 1: thinking (models that speak <think>) -------- */
    if (this.hasThink){
      gen.push(thinkTok); thinkIds.push(thinkTok);          // force-open, like training
      row = Sigma.lastRow((await this.runStep([thinkTok])).logits);
      const thinkBan = new Set(hardBan); thinkBan.add(thinkTok);
      for (let n = 0; n < maxThink; n++){
        if (shouldAbort && shouldAbort()) break;
        const t2 = this.sample(row, {temp: temp*0.9, topK, topP, ban: thinkBan, repPen, repWindow, gen});
        gen.push(t2); thinkIds.push(t2);
        if (onThink) onThink(tk.decode(thinkIds).think, false);
        if (t2 === endThink || stopIds.has(t2)) break;
        row = Sigma.lastRow((await this.runStep([t2])).logits);
      }
      if (thinkIds[thinkIds.length-1] !== endThink){
        gen.push(endThink); thinkIds.push(endThink);
        row = Sigma.lastRow((await this.runStep([endThink])).logits);
      }
      if (onThink) onThink(tk.decode(thinkIds).think, true);
    }
    if (shouldAbort && shouldAbort())
      return { text: "", mems: [], think: tk.decode(thinkIds).think, aborted: true };

    /* -------- phase 2: answer -------- */
    const ansBan = new Set(hardBan);
    if (this.hasThink) ansBan.add(thinkTok);
    if (ban) for (const b of ban) ansBan.add(b);
    for (let n = 0; n < maxNew; n++){
      if (shouldAbort && shouldAbort()) break;
      const t = this.sample(row, {temp, topK, topP, ban: ansBan, repPen, repWindow, gen});
      gen.push(t); answerIds.push(t);
      if (onToken) onToken(tk.decode(answerIds).text, n+1);
      if (stopIds.has(t)) break;
      row = Sigma.lastRow((await this.runStep([t])).logits);
    }
    const res = tk.decode(answerIds);
    return { text: res.text, mems: res.mems, think: tk.decode(thinkIds).think, aborted: false };
  }

  /* one-shot scoring passes — run on their own cache, then restore the
   * chat generation cache so they never corrupt an in-flight reply. */
  async intent(userText, persona="sigma"){
    const keep = [this.cacheK, this.cacheV];
    this.reset();
    const ids = this.clamp(this.tk.encode(`<bos><pers:${persona}><user>${userText}`));
    const out = await this.runStep(ids);
    this.cacheK = keep[0]; this.cacheV = keep[1];
    const d = out.intent_logits.data;
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < d.length; i++) if (d[i] > bv){ bv = d[i]; bi = i; }
    const exps = Array.from(d, Math.exp), s = exps.reduce((a,b)=>a+b,0);
    return { label: bi, probs: exps.map(e => +(e/s).toFixed(2)) };
  }

  async embed(text){
    const keep = [this.cacheK, this.cacheV];
    this.reset();
    const ids = this.clamp(this.tk.encode(`<bos><user>${text}`));
    const out = await this.runStep(ids);
    const v = Array.from(out.pooled.data);
    this.cacheK = keep[0]; this.cacheV = keep[1];
    return v;
  }
}

async function fetchWithProgress(url, onProgress, tries = 3){
  const once = () => fetch(url, { cache: "default" }).then(res => {
    if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
    const total = +res.headers.get("content-length") || 0;
    if (!res.body || !total) return res.arrayBuffer();
    const reader = res.body.getReader(); const chunks = []; let got = 0;
    const pump = () => reader.read().then(({done, value}) => {
      if (done) return undefined;
      chunks.push(value); got += value.length;
      onProgress && onProgress(got / total);
      return pump();
    });
    return pump().then(() => {
      const buf = new Uint8Array(got); let off = 0;
      for (const c of chunks){ buf.set(c, off); off += c.length; }
      return buf.buffer;
    });
  });
  let err;
  for (let i = 0; i < tries; i++){
    try { return await once(); }
    catch (e){ err = e; await new Promise(r => setTimeout(r, 700)); }
  }
  throw err;
}

/* big checkpoints ship as byte-split shards (github 100MB/file limit):
 * fetch parts sequentially, concat into one buffer, progress is aggregate */
const SHARDS = { "v10": 3, "v11": 3, "50m": 3, "100m": 5, "px": 2 };
const FNAME = { "px": "pixel" };    // session key -> file base (export writes pixel.onnx.part*)
async function fetchModelBuf(size, onProgress){
  const f = FNAME[size] || size;
  const n = SHARDS[size] || 1;
  if (n === 1) return fetchWithProgress(SIGMA_BASE + `models/${f}.onnx`, onProgress);
  const parts = [];
  for (let i = 1; i <= n; i++){
    const base = i / n - (1 / n);                       // progress offset of this part
    parts.push(await fetchWithProgress(SIGMA_BASE + `models/${f}.onnx.part${i}`,
                 p => onProgress && onProgress(base + p / n)));
  }
  const tot = parts.reduce((s, b) => s + b.byteLength, 0);
  const buf = new Uint8Array(tot); let off = 0;
  for (const b of parts){ buf.set(new Uint8Array(b), off); off += b.byteLength; }
  return buf.buffer;
}

window.SIGMA = { Tok, Sigma, SPECIALS, fetchModelBuf };
