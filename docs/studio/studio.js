/* PIXEL STUDIO runtime — paint editor + the 32×32 pixel diffusion model.
 * The network denoises pure Gaussian noise into pixel art (DDPM-family,
 * v-prediction trunk + categorical 16-color palette head, trained from zero
 * in src/pixel_train.py / pixel_ft.py). Nothing is picked from a gallery:
 * every image is invented by the weights, seeded by (prompt, seed, temp).
 * Remix = real img2img: the canvas is noised to a mid schedule step, then
 * DDIM walks it back down — the net reinvents the details around your paint.
 * Self-contained: owns its sampler, palette, prompt steering and editor. */
(function(){
  if (window.ort) { ort.env.wasm.wasmPaths = "../"; ort.env.wasm.numThreads = 1; }
  const $ = s => document.querySelector(s);

  /* ---------------- palette (matches python training exactly) ---------------- */
  const PAL = [
    [0,0,0],[255,255,255],[212,212,212],[122,122,122],
    [46,46,56],[27,58,107],[63,127,204],[126,200,227],
    [217,79,79],[242,163,60],[247,217,76],[106,176,76],
    [46,125,79],[142,90,59],[242,201,160],[155,89,182]
  ];
  const CLASSES = ["creature", "sunset", "spaceship", "flower", "castle", "gem"];
  const STEER = {
    creature: ["creature","monster","beast","dog","cat","animal","blob","slime","ghost","frog","bird","alien","bug","bear","wolf","duck"],
    sunset:   ["sunset","sunrise","sky","dusk","dawn","mountain","landscape","sun","cloud","evening","horizon","valley","sea","ocean","field"],
    spaceship:["spaceship","ship","rocket","space","star","shuttle","ufo","craft","station","moon","galaxy","nebula"],
    flower:   ["flower","rose","tulip","bloom","plant","petal","garden","daisy","leaf","tree","forest","spring"],
    castle:   ["castle","tower","fortress","keep","palace","building","city","temple","house","ruin","wall","gate","dungeon"],
    gem:      ["gem","gemstone","crystal","diamond","jewel","ruby","emerald","stone","amber","treasure","shard"],
  };

  /* cosine alpha-bar schedule — same formula as pixel_train.py */
  const T = 300, SCHED = new Float32Array(T + 1);
  (function(){
    const s = 0.008; const f0 = Math.cos((0 + s) / (1 + s) * Math.PI / 2) ** 2;
    for (let i = 0; i <= T; i++){
      const f = Math.cos((i / T + s) / (1 + s) * Math.PI / 2) ** 2;
      SCHED[i] = Math.min(0.999, f / f0);
    }
  })();

  /* seeded rng so (prompt, seed) reproduces an image exactly */
  function mulberry32(a){
    return function(){
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const gaussFrom = (rng) => {
    let u = 0, v = 0;
    while (!u) u = rng(); while (!v) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const hashStr = (s) => { let h = 2166136261;
    for (let i = 0; i < s.length; i++){ h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0; };

  function steerClass(prompt, seed){
    const words = prompt.toLowerCase().split(/[^a-z]+/).filter(Boolean);
    for (const w of words)
      for (let c = 0; c < CLASSES.length; c++)
        if (STEER[CLASSES[c]].includes(w)) return c;
    return hashStr(prompt + "::" + seed) % CLASSES.length;   // unknown words: deterministic pick
  }

  /* ---------------- the diffusion sampler ---------------- */
  class Diffusion {
    constructor(sess){ this.sess = sess; }
    static async load(onProgress){
      const buf = await SIGMA.fetchModelBuf("px", onProgress);
      const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return new Diffusion(sess);
    }
    /* one step of the net: x,t,cls -> palette logits [1024,16] */
    net(x, t, cls){
      return this.sess.run({
        x: new ort.Tensor("float32", x, [1, 3072]),
        t: new ort.Tensor("float32", new Float32Array([t]), [1]),
        cls: new ort.Tensor("int64", BigInt64Array.from([BigInt(cls)]), [1]),
      });
    }
    /* paint a logits row-set into on-palette RGB in [-1,1]
     * temp<=0 -> argmax (crisp); else multinomial per pixel */
    paintX0(logits, temp, rng){
      const x0 = new Float32Array(3072);
      for (let p = 0; p < 1024; p++){
        let idx = 0;
        if (temp <= 0){
          let bv = -Infinity;
          for (let c = 0; c < 16; c++){ const v = logits[p*16+c]; if (v > bv){ bv = v; idx = c; } }
        } else {
          const probs = new Float32Array(16); let mx = -Infinity;
          for (let c = 0; c < 16; c++) mx = Math.max(mx, logits[p*16+c]);
          let sum = 0;
          for (let c = 0; c < 16; c++){ probs[c] = Math.exp((logits[p*16+c] - mx) / temp); sum += probs[c]; }
          let r = rng() * sum;
          for (let c = 0; c < 16; c++){ r -= probs[c]; if (r <= 0){ idx = c; break; } }
        }
        const col = PAL[idx];
        x0[p*3] = col[0] / 127.5 - 1; x0[p*3+1] = col[1] / 127.5 - 1; x0[p*3+2] = col[2] / 127.5 - 1;
      }
      return x0;
    }
    /* full DDIM walk. from (optional) = {x0: Float32Array 3072, strength 0..1}
     * -> img2img remix; else pure noise. onStep(i,n) between steps. */
    async generate(cls, {steps = 40, temp = 1.15, seed = 1, from = null, onStep} = {}){
      const rng = mulberry32(seed >>> 0);
      let t0, ts = [];
      if (from){
        t0 = Math.max(1, Math.round((T - 1) * from.strength));
      } else {
        t0 = T - 1;
      }
      for (let i = 0; i < steps; i++) ts.push(Math.round(t0 * (1 - i / (steps - 1))));
      let x = new Float32Array(3072);
      if (from){
        const s1 = Math.sqrt(SCHED[t0]), s2 = Math.sqrt(Math.max(1e-3, 1 - SCHED[t0]));
        for (let j = 0; j < 3072; j++) x[j] = s1 * from.x0[j] + s2 * gaussFrom(rng);
      } else {
        for (let j = 0; j < 3072; j++) x[j] = gaussFrom(rng);
      }
      let x0 = x;
      for (let i = 0; i < steps; i++){
        const t = ts[i], abT = SCHED[t], abPrev = i + 1 < steps ? SCHED[ts[i + 1]] : 1.0;
        const out = await this.net(x, t, cls);
        const last = i + 1 >= steps;
        x0 = this.paintX0(out.logits.data, last ? 0 : temp, rng);
        const s1 = Math.sqrt(abT), s2 = Math.sqrt(Math.max(1e-3, 1 - abT));
        const s3 = Math.sqrt(abPrev), s4 = Math.sqrt(Math.max(0, 1 - abPrev));
        const xn = new Float32Array(3072);
        for (let j = 0; j < 3072; j++){
          const eps = (x[j] - s1 * x0[j]) / s2;
          xn[j] = s3 * x0[j] + s4 * eps;
        }
        x = xn;
        if (onStep) onStep(i + 1, steps, x0);
      }
      /* final x0 -> RGBA bytes */
      const img = new Uint8ClampedArray(32 * 32 * 4);
      for (let p = 0; p < 1024; p++){
        img[p*4]   = Math.round((x0[p*3]   + 1) * 127.5);
        img[p*4+1] = Math.round((x0[p*3+1] + 1) * 127.5);
        img[p*4+2] = Math.round((x0[p*3+2] + 1) * 127.5);
        img[p*4+3] = 255;
      }
      return img;
    }
  }

  /* ================= the paint editor ================= */
  const board = $("#board"), gridCv = $("#grid");
  const bctx = board.getContext("2d", { willReadFrequently: true });
  gridCv.width = 512; gridCv.height = 512;
  (function drawGrid(){
    const g = gridCv.getContext("2d");
    g.clearRect(0, 0, 512, 512);
    g.strokeStyle = "rgba(255,255,255,.9)"; g.lineWidth = 1;
    for (let i = 0; i <= 32; i += 4){
      g.globalAlpha = i % 16 === 0 ? 0.9 : 0.4;
      g.beginPath(); g.moveTo(i * 16 + .5, 0); g.lineTo(i * 16 + .5, 512); g.stroke();
      g.beginPath(); g.moveTo(0, i * 16 + .5); g.lineTo(512, i * 16 + .5); g.stroke();
    }
  })();

  let px = new Uint8ClampedArray(32 * 32 * 4);       // ground truth pixels
  const undoStack = [], redoStack = [];
  let tool = "pencil", colorIdx = 1, painting = false, gridOn = true;

  function pushUndo(){
    undoStack.push(px.slice());
    if (undoStack.length > 60) undoStack.shift();
    redoStack.length = 0;
  }
  function render(){
    bctx.putImageData(new ImageData(px, 32, 32), 0, 0);
  }
  function clearCanvas(){
    pushUndo();
    for (let i = 0; i < 1024; i++){
      px[i*4] = PAL[0][0]; px[i*4+1] = PAL[0][1]; px[i*4+2] = PAL[0][2]; px[i*4+3] = 255;
    }
    render();
  }
  function loadImg(img){          // Uint8ClampedArray 32*32*4
    pushUndo();
    px = img.slice(); render();
  }

  function floodFill(sx, sy, hex){
    const at = (x, y) => (y * 32 + x) * 4;
    const key = (x, y) => `${px[at(x,y)]},${px[at(x,y)+1]},${px[at(x,y)+2]}`;
    const target = key(sx, sy);
    const [fr, fg, fb] = PAL[hex];
    if (target === `${fr},${fg},${fb}`) return;
    const q = [[sx, sy]];
    while (q.length){
      const [x, y] = q.pop();
      if (x < 0 || y < 0 || x > 31 || y > 31) continue;
      if (key(x, y) !== target) continue;
      const i = at(x, y);
      px[i] = fr; px[i+1] = fg; px[i+2] = fb; px[i+3] = 255;
      q.push([x+1,y],[x-1,y],[x,y+1],[x,y-1]);
    }
  }
  function nearestPal(r, g, b){
    let bi = 0, bd = Infinity;
    for (let i = 0; i < PAL.length; i++){
      const d = (PAL[i][0]-r)**2 + (PAL[i][1]-g)**2 + (PAL[i][2]-b)**2;
      if (d < bd){ bd = d; bi = i; }
    }
    return bi;
  }

  function cellFromEvent(e){
    const r = board.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / r.width * 32);
    const y = Math.floor((e.clientY - r.top) / r.height * 32);
    return [Math.max(0, Math.min(31, x)), Math.max(0, Math.min(31, y))];
  }
  function applyTool(x, y){
    const i = (y * 32 + x) * 4;
    if (tool === "pencil"){
      px[i] = PAL[colorIdx][0]; px[i+1] = PAL[colorIdx][1]; px[i+2] = PAL[colorIdx][2]; px[i+3] = 255;
    } else if (tool === "eraser"){
      px[i] = PAL[0][0]; px[i+1] = PAL[0][1]; px[i+2] = PAL[0][2]; px[i+3] = 255;
    } else if (tool === "picker"){
      colorIdx = nearestPal(px[i], px[i+1], px[i+2]);
      document.querySelectorAll(".sw").forEach((s, j) => s.classList.toggle("active", j === colorIdx));
      setTool("pencil");
      return;
    }
    render();
  }
  board.addEventListener("pointerdown", e => {
    e.preventDefault();
    const [x, y] = cellFromEvent(e);
    if (tool === "fill"){ pushUndo(); floodFill(x, y, colorIdx); render(); return; }
    pushUndo(); painting = true; board.setPointerCapture(e.pointerId);
    applyTool(x, y);
  });
  board.addEventListener("pointermove", e => {
    if (!painting) return;
    if (tool === "fill" || tool === "picker") return;
    const [x, y] = cellFromEvent(e); applyTool(x, y);
  });
  ["pointerup", "pointercancel"].forEach(ev => board.addEventListener(ev, () => painting = false));

  function setTool(t){
    tool = t;
    document.querySelectorAll(".tool[data-t]").forEach(b => b.classList.toggle("active", b.dataset.t === t));
    board.style.cursor = t === "picker" ? "copy" : "crosshair";
  }
  document.querySelectorAll(".tool[data-t]").forEach(b => b.onclick = () => setTool(b.dataset.t));

  /* palette swatches */
  (function(){
    const pal = $("#palette");
    PAL.forEach((c, i) => {
      const s = document.createElement("button");
      s.className = "sw" + (i === colorIdx ? " active" : "");
      s.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;
      s.title = `color ${i}`;
      s.onclick = () => { colorIdx = i; setTool("pencil");
        document.querySelectorAll(".sw").forEach((x, j) => x.classList.toggle("active", j === i)); };
      pal.appendChild(s);
    });
  })();

  $("#btnUndo").onclick = () => { if (undoStack.length){ redoStack.push(px.slice()); px = undoStack.pop(); render(); } };
  $("#btnRedo").onclick = () => { if (redoStack.length){ undoStack.push(px.slice()); px = redoStack.pop(); render(); } };
  $("#btnClear").onclick = clearCanvas;
  $("#btnGrid").onclick = () => { gridOn = !gridOn; gridCv.classList.toggle("on", gridOn);
    $("#btnGrid").classList.toggle("active", gridOn); };
  gridCv.classList.add("on");

  $("#btnPng").onclick = () => {
    const off = document.createElement("canvas"); off.width = 256; off.height = 256;
    const c = off.getContext("2d"); c.imageSmoothingEnabled = false;
    c.drawImage(board, 0, 0, 256, 256);
    const a = document.createElement("a");
    a.download = "sigma-pixel.png"; a.href = off.toDataURL("image/png"); a.click();
  };

  /* ================= forge wiring ================= */
  const state = { gen: null, busy: false };
  const status = $("#pxStatus"), btnForge = $("#btnForge"), btnVar = $("#btnVar"), btnRemix = $("#btnRemix");
  const gallery = $("#gallery");

  $("#btnDice").onclick = () => { $("#pxSeed").value = Math.floor(Math.random() * 100000); };
  $("#pxTemp").oninput = e => $("#pxTempV").textContent = (+e.target.value).toFixed(2);
  $("#pxStr").oninput = e => $("#pxStrV").textContent = (+e.target.value).toFixed(2);

  function canvasX0(){
    const x0 = new Float32Array(3072);
    for (let p = 0; p < 1024; p++){
      x0[p*3]   = px[p*4]   / 127.5 - 1;
      x0[p*3+1] = px[p*4+1] / 127.5 - 1;
      x0[p*3+2] = px[p*4+2] / 127.5 - 1;
    }
    return x0;
  }
  function pushGallery(img){
    const off = document.createElement("canvas"); off.width = 32; off.height = 32;
    off.getContext("2d").putImageData(new ImageData(img.slice(), 32, 32), 0, 0);
    const url = off.toDataURL("image/png");
    const empty = gallery.querySelector(".gal-empty"); if (empty) empty.remove();
    const b = document.createElement("button");
    b.className = "gal-item"; b.style.backgroundImage = `url(${url})`; b.title = "load into the canvas";
    b.onclick = () => loadImg(img);
    gallery.prepend(b);
    while (gallery.children.length > 12) gallery.lastChild.remove();
  }
  async function run(kind){
    if (!state.gen || state.busy) return;
    const prompt = $("#pxPrompt").value.trim() || "something strange";
    const seed = Math.max(0, +$("#pxSeed").value || 0) + (kind === "var" ? 1 : 0);
    $("#pxSeed").value = seed;
    const steps = +$("#pxSteps").value;
    const temp = +$("#pxTemp").value;
    const cls = steerClass(prompt, seed);
    state.busy = true;
    [btnForge, btnVar, btnRemix].forEach(b => b.disabled = true);
    const clsName = CLASSES[cls];
    try {
      if (kind === "remix"){
        status.textContent = `remixing — noising to ${Math.round(+$("#pxStr").value*100)}% then denoising…`;
        const img = await state.gen.generate(cls, { steps, temp, seed,
          from: { x0: canvasX0(), strength: +$("#pxStr").value },
          onStep: (i, n) => status.textContent = `remixing… step ${i}/${n} · steered: ${clsName}` });
        loadImg(img); pushGallery(img);
        status.textContent = `remixed — the net reinvented it around your paint (steered: ${clsName})`;
      } else {
        status.textContent = `denoising pure noise · ${steps} steps…`;
        const img = await state.gen.generate(cls, { steps, temp, seed,
          onStep: (i, n) => status.textContent = `denoising… step ${i}/${n} · steered: ${clsName}` });
        loadImg(img); pushGallery(img);
        status.textContent = `forged from noise — seed ${seed}, steered: ${clsName}`;
      }
    } catch (err){
      console.error(err);
      status.textContent = "forge hiccup: " + err.message;
    }
    state.busy = false;
    [btnForge, btnVar, btnRemix].forEach(b => b.disabled = false);
  }
  btnForge.onclick = () => run("forge");
  btnVar.onclick = () => run("var");
  btnRemix.onclick = () => run("remix");

  /* boot */
  clearCanvas(); render();
  const stateEl = $("#state");
  (async () => {
    stateEl.textContent = "loading diffusion weights…";
    try {
      state.gen = await Diffusion.load(p => {
        stateEl.textContent = `loading weights… ${Math.round(p*100)}%`;
        status.textContent = `loading diffusion weights… ${Math.round(p*100)}%`;
      });
      btnForge.disabled = false; btnVar.disabled = false; btnRemix.disabled = false;
      status.textContent = "ready — describe it, forge it, paint on it, remix it";
      stateEl.textContent = "pixel studio ready";
    } catch (err){
      console.error(err);
      status.textContent = "weight load failed — check connection once, then it's cached";
      stateEl.textContent = "load failed";
    }
  })();
})();
