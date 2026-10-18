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
  static async load(){ const itos = await (await fetch("models/vocab2.json")).json(); return new Tok(itos); }
  encode(text){ const u = this.stoi["<unk>"]; return rawTokens(text).map(w => this.stoi[w] ?? u); }
  /* decode generated ids -> {text, mems, think}
   * - <mem>..</mem>     -> collected into mems (hidden from text)
   * - <think>..</think> -> collected into think (rendered in the thinking panel) */
  decode(ids){
    const drop = new Set(["<pad>","<bos>","<user>","<ai>","<memory>","</memory>","<eos>","<unk>","<nl>"]);
    const glue = new Set([".",",","!","?",":",";","'","%","'s",")","}"]);
    let out = "", inMem = false, inThink = false;
    const mems = []; let curMem = "", think = "", curThink = "";
    const push = (dst, w) => dst + (dst && !dst.endsWith(" ") ? " " : "") + w;
    for (const i of ids){
      const w = this.itos[i]; if (w == null) continue;
      if (w === "<mem>"){ inMem = true; curMem = ""; continue; }
      if (w === "</mem>"){ inMem = false; if (curMem.trim()) mems.push(curMem.trim()); continue; }
      if (w === "<think>"){ inThink = true; curThink = ""; continue; }
      if (w === "</think>"){ inThink = false; if (curThink.trim()) think = curThink.trim(); continue; }
      if (inThink){ if (!drop.has(w)) curThink = push(curThink, w); continue; }
      if (inMem){ if (!drop.has(w)) curMem = push(curMem, w); continue; }
      if (w === "<nl>"){ out += "\n"; continue; }
      if (drop.has(w)) continue;
      if (SPECIALS.includes(w)) continue;
      if (glue.has(w)) out += w;
      else out += (out && !out.endsWith("\n") && !out.endsWith(" ") ? " " : "") + w;
    }
    if (inThink && curThink.trim()) think = curThink.trim();   // stream-open think
    if (inMem && curMem.trim()) mems.push(curMem.trim());
    return { text: out.trim(), mems, think };
  }
}

/* ---------------- model session ---------------- */
const MDIMS = {   // [n_layers, n_heads, d_model, vocab_rows] per checkpoint
  "2m": [5, 5, 170, 1446], "5m": [6, 8, 256, 1446], "10m": [6, 8, 368, 1446],
  "17m": [8, 8, 416, 1446], "20m": [8, 8, 448, 1538], "22m": [8, 8, 472, 1540],
};

class Sigma {
  constructor(sess, tk, size){
    this.sess = sess; this.tk = tk; this.size = size;
    this.cacheK = null; this.cacheV = null;      // ORT tensors [L,1,H,T,Dh]
    this.hasThink = tk.stoi["<think>"] != null;  // new checkpoints speak <think>
    const md = MDIMS[size] || [8, 8, 416, 1446];
    this.L = md[0]; this.H = md[1]; this.Dh = md[2] / md[1]; this.V = md[3];
  }

  /* clamp ids to this checkpoint's embedding table (newer vocab words -> <unk>) */
  clamp(ids){ const u = this.tk.stoi["<unk>"]; return ids.map(i => i < this.V ? i : u); }

  static async load(size, onProgress){
    const tk = await Tok.load();
    const url = `models/${size}.onnx`;
    const buf = await fetchWithProgress(url, onProgress);
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

  /* one softmax sampling step over a logits row */
  sample(row, {temp, topK, topP, ban, repPen, repWindow, gen}){
    const V = row.length;
    const z = new Float32Array(V);
    let mx = -Infinity;
    for (let v = 0; v < V; v++){
      let x = row[v];
      if (ban.has(v)) x = -1e9;
      if (repPen && gen && gen.length){
        let hit = false;
        for (let i = Math.max(0, gen.length - repWindow); i < gen.length; i++)
          if (gen[i] === v){ hit = true; break; }
        if (hit) x = x > 0 ? x / repPen : x * repPen;
      }
      z[v] = x; if (x > mx) mx = x;
    }
    const probs = new Float32Array(V);
    let sum = 0;
    for (let v = 0; v < V; v++){ probs[v] = Math.exp(z[v] - mx); sum += probs[v]; }
    for (let v = 0; v < V; v++) probs[v] /= sum;
    const idxs = Array.from({length: V}, (_, v) => v);
    idxs.sort((a,b) => probs[b] - probs[a]);
    const kept = idxs.slice(0, topK);
    let cum = 0; const kept2 = [];
    for (const v of kept){ kept2.push(v); cum += probs[v]; if (cum >= topP) break; }
    const tot = kept2.reduce((s,v) => s + probs[v], 0);
    let r = Math.random() * tot, pick = kept2[0];
    for (const v of kept2){ r -= probs[v]; if (r <= 0){ pick = v; break; } }
    return temp <= 0.01 ? kept2[0] : pick;
  }

  /* full two-phase reply:
   *  promptIds — token ids ending with <ai>
   *  onThink(txt, done)   — streamed thinking text (new checkpoints only)
   *  onToken(txt, n)      — streamed answer text
   *  shouldAbort()        — polled between steps; true -> return partial
   * Old checkpoints have no <think> token: the app fills the thinking panel
   * with real telemetry (intent probs, recalled memories) instead. */
  async generate(promptIds, {maxThink=40, maxNew=70, temp=0.68, topK=22, topP=0.9,
                             repPen=1.28, repWindow=14, onThink, onToken, shouldAbort}={}){
    this.reset();
    promptIds = this.clamp(promptIds);
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

function fetchWithProgress(url, onProgress){
  return fetch(url).then(res => {
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
}

window.SIGMA = { Tok, Sigma, SPECIALS };
