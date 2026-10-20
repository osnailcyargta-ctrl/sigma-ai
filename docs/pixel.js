/* SIGMA Pixel Studio — the browser side of the 32x32 pixel diffusion model.
 * The network is a real generative model (DDPM-family, v-prediction trunk +
 * categorical 16-color palette head trained with cross-entropy): it starts
 * from pure Gaussian noise and denoises over N DDIM steps. Every image is
 * invented by the weights — nothing is picked from a template gallery.
 * Sampler mirrors src/pixel_ft.py: multinomial palette draws at intermediate
 * steps, argmax on the final step for crisp pixels. */
(function(){
  const PALETTE = [
    [0,0,0],[255,255,255],[212,212,212],[122,122,122],
    [46,46,56],[27,58,107],[63,127,204],[126,200,227],
    [217,79,79],[242,163,60],[247,217,76],[106,176,76],
    [46,125,79],[142,90,59],[242,201,160],[155,89,182]
  ].map(c => c.map(v => v / 127.5 - 1));          // to [-1,1], matches python

  const CLASSES = ["creature", "sunset", "spaceship", "flower", "castle", "gem"];

  /* cosine alpha-bar schedule — same formula as pixel_train.py */
  const T = 300, SCHED = new Float32Array(T + 1);
  (function(){
    const s = 0.008; let f0 = Math.cos((0 + s) / (1 + s) * Math.PI / 2) ** 2;
    for (let i = 0; i <= T; i++){
      const f = Math.cos((i / T + s) / (1 + s) * Math.PI / 2) ** 2;
      SCHED[i] = Math.min(0.999, f / f0);
    }
  })();

  const gauss = () => {
    let u = 0, v = 0;
    while (!u) u = Math.random(); while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  class PixelGen {
    constructor(sess){ this.sess = sess; }
    static async load(onProgress){
      const buf = await SIGMA.fetchModelBuf("px", onProgress);
      const sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return new PixelGen(sess);
    }
    /* one DDIM pass; onStep(step, total) fires between steps so the UI can breathe */
    async generate(clsIdx, {steps = 40, temp = 1.1, onStep} = {}){
      const x = new Float32Array(3072);
      for (let i = 0; i < 3072; i++) x[i] = gauss();
      const ts = [];
      for (let i = 0; i < steps; i++) ts.push(Math.round((T - 1) * (1 - i / (steps - 1))));
      let x0 = new Float32Array(3072);
      for (let i = 0; i < steps; i++){
        const t = ts[i], abT = SCHED[t], abPrev = i + 1 < steps ? SCHED[ts[i + 1]] : 1.0;
        const out = await this.sess.run({
          x: new ort.Tensor("float32", x, [1, 3072]),
          t: new ort.Tensor("float32", new Float32Array([t]), [1]),
          cls: new ort.Tensor("int64", BigInt64Array.from([BigInt(clsIdx)]), [1]),
        });
        const logits = out.logits.data;               // [1024,16]
        x0 = new Float32Array(3072);
        const last = i + 1 >= steps || temp <= 0;
        for (let p = 0; p < 1024; p++){
          let idx = 0;
          if (last){
            let bv = -Infinity;
            for (let c = 0; c < 16; c++){ const v = logits[p*16+c]; if (v > bv){ bv = v; idx = c; } }
          } else {
            /* softmax at temp, sample */
            const probs = new Float32Array(16); let mx = -Infinity;
            for (let c = 0; c < 16; c++) mx = Math.max(mx, logits[p*16+c]);
            let sum = 0;
            for (let c = 0; c < 16; c++){ probs[c] = Math.exp((logits[p*16+c] - mx) / temp); sum += probs[c]; }
            let r = Math.random() * sum;
            for (let c = 0; c < 16; c++){ r -= probs[c]; if (r <= 0){ idx = c; break; } }
          }
          const col = PALETTE[idx];
          x0[p*3] = col[0]; x0[p*3+1] = col[1]; x0[p*3+2] = col[2];
        }
        /* DDIM update: eps = (x - sqrt(ab) x0)/sqrt(1-ab); x = sqrt(ab') x0 + sqrt(1-ab') eps */
        const s1 = Math.sqrt(abT), s2 = Math.sqrt(Math.max(1e-3, 1 - abT));
        const s3 = Math.sqrt(abPrev), s4 = Math.sqrt(Math.max(0, 1 - abPrev));
        const xn = new Float32Array(3072);
        for (let j = 0; j < 3072; j++){
          const eps = (x[j] - s1 * x0[j]) / s2;
          xn[j] = s3 * x0[j] + s4 * eps;
        }
        x.set(xn);
        if (onStep) onStep(i + 1, steps, x0);
      }
      /* final x0 is on-palette RGB in [-1,1] -> RGBA bytes */
      const img = new Uint8ClampedArray(32 * 32 * 4);
      for (let p = 0; p < 1024; p++){
        img[p*4]   = Math.round((x0[p*3] + 1) * 127.5);
        img[p*4+1] = Math.round((x0[p*3+1] + 1) * 127.5);
        img[p*4+2] = Math.round((x0[p*3+2] + 1) * 127.5);
        img[p*4+3] = 255;
      }
      return img;
    }
  }

  window.SIGMA_PIXEL = { PixelGen, CLASSES };
})();
