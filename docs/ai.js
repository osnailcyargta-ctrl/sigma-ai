/* SIGMA AI runtime: word tokenizer (1:1 port of the Python tokenizer) +
 * ONNX inference (LM generation, intent head, pooled embeddings for memory). */

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
  static async load(){ const itos = await (await fetch("models/vocab.json")).json(); return new Tok(itos); }
  encode(text){ const u = this.stoi["<unk>"]; return rawTokens(text).map(w => this.stoi[w] ?? u); }
  decode(ids){
    const drop = new Set(["<pad>","<bos>","<user>","<ai>","<mem>","</mem>","<memory>","</memory>","<eos>","<unk>","<nl>"]);
    const glue = new Set([".",",","!","?",":",";","'","%","'s",")","}"]);
    let out = "";
    for (const i of ids){
      const w = this.itos[i]; if (w == null) continue;
      if (w === "<nl>"){ out += "\n"; continue; }
      if (drop.has(w)) continue;
      if (SPECIALS.includes(w)) continue;
      if (glue.has(w)) out += w;
      else out += (out && !out.endsWith("\n") && !out.endsWith(" ") ? " " : "") + w;
    }
    return out.trim();
  }
}

/* ---------------- model session ---------------- */
class Sigma {
  constructor(sess, tk, size){ this.sess = sess; this.tk = tk; this.size = size; }

  static async load(size, onProgress){
    const tk = await Tok.load();
    const url = `models/${size}.onnx`;
    const buf = await fetchWithProgress(url, onProgress);
    const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return new Sigma(sess, tk, size);
  }

  feed(ids){ return this.sess.run({ idx: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]) }); }

  /* full reply generation with streaming callback */
  async generate(promptText, {maxNew=44, temp=0.72, topK=25, topP=0.9, onToken}={}){
    const ids = this.tk.encode(promptText).slice(-250);
    const stop = new Set([this.tk.stoi["<user>"], this.tk.stoi["<eos>"]]);
    const nl = this.tk.stoi["<nl>"];
    const out = [];
    const rng = Math.random;
    for (let n = 0; n < maxNew; n++){
      const { logits } = await this.feed(ids);
      const [ , T, V ] = logits.dims;
      const row = logits.data.subarray((T-1)*V, T*V);
      // block specials the model should never say
      const banned = new Set([this.tk.stoi["<pad>"], this.tk.stoi["<bos>"], this.tk.stoi["<unk>"],
                              this.tk.stoi["<mem>"], this.tk.stoi["</mem>"],
                              this.tk.stoi["<memory>"], this.tk.stoi["</memory>"]]);
      const z = new Float32Array(V);
      let mx = -Infinity;
      for (let v = 0; v < V; v++){
        let x = row[v];
        if (banned.has(v)) x = -1e9;
        z[v] = x; if (x > mx) mx = x;
      }
      const probs = new Float32Array(V);
      let sum = 0;
      for (let v = 0; v < V; v++){ probs[v] = Math.exp(z[v] - mx); sum += probs[v]; }
      for (let v = 0; v < V; v++) probs[v] /= sum;
      // top-k
      const idxs = Array.from({length: V}, (_, v) => v);
      idxs.sort((a,b) => probs[b] - probs[a]);
      const kept = idxs.slice(0, topK);
      // top-p
      let cum = 0; const kept2 = [];
      for (const v of kept){ kept2.push(v); cum += probs[v]; if (cum >= topP) break; }
      const tot = kept2.reduce((s,v) => s + probs[v], 0);
      let r = rng() * tot, pick = kept2[0];
      for (const v of kept2){ r -= probs[v]; if (r <= 0){ pick = v; break; } }
      const t = Math.round(temp) === 0 ? kept2[0] : pick; // temp<=0 -> greedy
      ids.push(t); out.push(t);
      if (onToken) onToken(this.tk.decode(out));
      if (stop.has(t)) break;
    }
    return this.tk.decode(out).replace(/<mem>[\s\S]*?<\/mem>/g, "").trim();
  }

  async intent(userText, persona="sigma"){
    const ids = this.tk.encode(`<bos><pers:${persona}><user>${userText}`);
    const { intent_logits } = await this.feed(ids);
    const d = intent_logits.data;
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < d.length; i++) if (d[i] > bv){ bv = d[i]; bi = i; }
    const exps = Array.from(d, Math.exp), s = exps.reduce((a,b)=>a+b,0);
    return { label: bi, probs: exps.map(e => +(e/s).toFixed(2)) };
  }

  async embed(text){
    const ids = this.tk.encode(`<bos><user>${text}`);
    const { pooled } = await this.feed(ids);
    return Array.from(pooled.data);
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
