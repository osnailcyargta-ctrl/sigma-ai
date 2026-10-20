/* SIGMA app: chat UI + personality + 3-layer memory + game intent wiring +
 * Z.ai-style live thinking panel + send cooldown / stop control. */
(function(){
  if (window.ort) { ort.env.wasm.wasmPaths = "./"; ort.env.wasm.numThreads = 1; }
  const $ = s => document.querySelector(s);
  const chat = $("#chat"), input = $("#input"), sendBtn = $("#btnSend");
  const chipModel = $("#chipModel"), chipPers = $("#chipPers"), loadState = $("#loadState");

  const BLOCK = 256;                     // legacy cap; live cap comes from model.block
  const state = {
    size: localStorage.getItem("sigma.size") || "50m",
    pers: localStorage.getItem("sigma.pers") || "sigma",
    custom: localStorage.getItem("sigma.custom") || "",
    mode: localStorage.getItem("sigma.mode") || "chat",   // chat | shakes (Claude-style switch)
    history: [],          // [{u, a}] working memory (context window)
    sessionMems: [],      // the model's OWN <mem> writes this session — always replayed
    model: null, tk: null, loading: false,
    generating: false, abort: false, lastSend: 0,
  };

  /* ---------------- mode switch (Sigma Chat <-> Shakespeare-GMA) ---------------- */
  const MODE_META = {
    chat:   { label: "Sigma Chat",     ico: "Σ", pers: null,
              ph: "Message SIGMA — or invite it to a game..." },
    shakes: { label: "Shakespeare-GMA", ico: "✒", pers: "shakespeare",
              ph: "Speak, friend — verse & wisdom await..." },
  };
  function setMode(m){
    state.mode = m; localStorage.setItem("sigma.mode", m);
    const meta = MODE_META[m];
    document.body.classList.toggle("mode-shakes", m === "shakes");
    $("#modeName").textContent = meta.label;
    $("#modeIco").textContent = meta.ico;
    $("#input").placeholder = meta.ph;
    document.querySelectorAll(".mode-item").forEach(x => x.classList.toggle("active", x.dataset.mode === m));
    chipPers.textContent = m === "shakes" ? "shakespeare-gma" : state.pers;
    $("#modeMenu").classList.remove("open");
  }
  /* the persona actually fed to the network for the current mode. Shakespeare
   * uses the trained <pers:shakespeare> token on checkpoints that have it; on
   * older tables it degrades to <pers:custom> + a style hint (still real
   * conditioning, no tag matching). */
  function modePersona(){
    const meta = MODE_META[state.mode];
    if (!meta.pers) return { tok: state.pers, hint: "" };
    if (state.model && state.model.supportsPers("shakespeare"))
      return { tok: "shakespeare", hint: "" };
    return { tok: "custom",
             hint: "you speak in grand verse like shakespeare and ask wise questions like socrates<nl>" };
  }

  /* ---------------- long-term memory (IndexedDB vector store) ---------------- */
  const MEM = (() => {
    let db;
    function open(){
      return new Promise((res, rej) => {
        const rq = indexedDB.open("sigma-mem", 1);
        rq.onupgradeneeded = () => rq.result.createObjectStore("turns", { keyPath: "id", autoIncrement: true });
        rq.onsuccess = () => { db = rq.result; res(); };
        rq.onerror = rej;
      });
    }
    const tx = () => db.transaction("turns", "readwrite").objectStore("turns");
    async function all(){ return new Promise(res => { const rq = tx().getAll(); rq.onsuccess = () => res(rq.result || []); }); }
    async function add(entry){ return new Promise(res => { const rq = tx().add(entry); rq.onsuccess = () => res(); }); }
    async function clear(){ return new Promise(res => { const rq = tx().clear(); rq.onsuccess = () => res(); }); }
    return { open, all, add, clear };
  })();

  function cosine(a, b){
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++){ dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
  }

  async function retrieve(queryEmb, k = 3){
    const rows = await MEM.all();
    if (!rows.length) return [];
    return rows.map(r => ({ ...r, sim: cosine(queryEmb, r.emb) }))
               .sort((a, b) => b.sim - a.sim).slice(0, k)
               .filter(r => r.sim > 0.35);      // wide enough to catch name recalls
  }

  async function refreshMemList(){
    const rows = await MEM.all();
    $("#memCount").textContent = rows.length;
    const list = $("#memList");
    list.innerHTML = rows.length ? "" : `<div class="mem-empty">memory is empty — tell it something about you</div>`;
    rows.slice(-40).reverse().forEach(r => {
      const d = document.createElement("div");
      d.className = "mem-item"; d.textContent = r.text;
      list.appendChild(d);
    });
  }

  /* ---------------- UI helpers ---------------- */
  function bubble(role, text){
    const m = document.createElement("div");
    m.className = "msg " + (role === "user" ? "user" : "ai");
    m.innerHTML = `<div class="avatar">${role === "user" ? "U" : "Σ"}</div><div class="bubble"></div>`;
    m.querySelector(".bubble").textContent = text;
    chat.appendChild(m); chat.scrollTop = chat.scrollHeight;
    return m;
  }

  /* AI message with a Z.ai-style live thinking panel above the answer bubble */
  function aiMessage(){
    const m = document.createElement("div");
    m.className = "msg ai";
    m.innerHTML =
      `<div class="avatar">Σ</div>` +
      `<div class="msg-col">` +
      `  <div class="think open"><button class="think-head" type="button">` +
      `    <span class="think-star">✦</span><span class="think-label">Thinking</span>` +
      `    <span class="chev">▾</span></button>` +
      `  <div class="think-body"></div></div>` +
      `  <div class="bubble"></div>` +
      `</div>`;
    const panel = m.querySelector(".think");
    m.querySelector(".think-head").onclick = () => panel.classList.toggle("open");
    chat.appendChild(m); chat.scrollTop = chat.scrollHeight;
    return {
      msg: m,
      bubble: m.querySelector(".bubble"),
      thinkBody: m.querySelector(".think-body"),
      thinkLabel: m.querySelector(".think-label"),
      thinkPanel: panel,
      closeThink(sec){ panel.classList.remove("open");
        this.thinkLabel.textContent = sec != null ? `Thought for ${sec}s` : "Thought"; },
    };
  }

  function greet(){
    chat.innerHTML = "";
    bubble("ai", "SIGMA online. A transformer trained from zero — every weight, every word of data. " +
      "I keep real memory of our chats, and I play tic tac toe, rock paper scissors and dice. Try me.");
  }

  /* ---------------- send button state machine ---------------- */
  function setBtn(mode){
    sendBtn.classList.remove("stop");
    if (mode === "idle"){ sendBtn.disabled = false; sendBtn.textContent = "➤"; sendBtn.title = "send"; }
    else if (mode === "cooldown"){ sendBtn.disabled = true; sendBtn.textContent = "➤"; }
    else if (mode === "stop"){ sendBtn.disabled = false; sendBtn.classList.add("stop");
      sendBtn.textContent = "■"; sendBtn.title = "stop generating"; }
  }

  /* ---------------- model loading ---------------- */
  async function loadModel(size){
    if (state.loading) return;
    state.loading = true; setBtn("cooldown");
    loadState.textContent = `loading ${size} weights…`;
    document.querySelectorAll("#modelSeg button").forEach(b => b.classList.toggle("active", b.dataset.m === size));
    try {
      let pctEl = loadState;
      const m = await SIGMA.Sigma.load(size, p => { pctEl.textContent = `loading ${size}… ${Math.round(p*100)}%`; });
      state.model = m; state.tk = m.tk; state.size = size;
      localStorage.setItem("sigma.size", size);
      chipModel.textContent = size.toUpperCase();
      const P = {"10m":"10,388,272 params · ppl 1.47","17m":"17,337,632 params · ppl 1.48",
                 "20m":"20,088,768 params · think + interrupts","22m":"22,276,040 params · char-copy memory",
                 "50m":"50,362,240 params · ppl 1.15 · open-vocab copy · flagship",
                 "v10":"50.5M params · ctx 512 · shakespeare-gma · flagship"};
      $("#modelInfo").textContent = P[size] || "";
      GAMES_setRuntime(m, state.pers);          // games can open before any send
      setMode(state.mode);                      // refresh persona chip for the mode
      loadState.textContent = `${size} ready`;
      setBtn("idle");
    } catch (err) {
      console.error(err);
      loadState.textContent = "load failed — check connection once, then it's cached";
    }
    state.loading = false;
  }

  /* ---------------- prompt assembly (the 3-layer memory) ----------------
   * Greedy newest-first packing: persona + <memory> block + current turn are
   * ALWAYS kept; older turns are added newest-first, only as COMPLETE turns.
   * The old slice(-keep) could cut a char-spelled name in half mid-sequence
   * ('joninibanana' -> '...nibanana') — that alone destroyed recall.
   * The <memory> block carries: (a) the model's OWN <mem> writes from this
   * session — always, because recall must never depend on a similarity
   * threshold — plus (b) long-term rows that semantically match the query.
   * Replaying the model's own written memories is exactly the trained
   * <memory>-block format, so this is in-distribution, not tag matching. */
  async function buildPromptIds(userText, memRows, model){
    const tk = model.tk;
    const keep = model.block - (model.hasThink ? 34 : 2) - 74;
    const mp = modePersona();
    const head = `<bos><pers:${mp.tok}>` +
      (mp.hint || (mp.tok === "custom" ? (state.custom.trim() || "you are helpful, honest and a little witty") + "<nl>" : ""));
    const sessRows = state.sessionMems.slice(-6)
      .map(t => "• " + t);
    const seen = new Set(sessRows.map(r => r.slice(2).toLowerCase()));
    const retrRows = (memRows || [])
      .filter(r => !seen.has(r.text.toLowerCase()))
      .map(r => "• " + r.text);
    const allRows = retrRows.concat(sessRows).slice(-8);
    const mblock = allRows.length ? "<memory>" + allRows.join(" ") + "</memory>" : "";
    const cur = `<user>${userText}<ai>`;
    let room = keep - tk.encode(head + mblock + cur).length;
    const parts = [];
    for (let i = state.history.length - 1; i >= 0; i--){
      const s = `<user>${state.history[i].u}<ai>${state.history[i].a}`;
      const c = tk.encode(s).length;
      if (c > room) break;
      parts.unshift(s); room -= c;
    }
    return tk.encode(head + mblock + parts.join("") + cur);
  }

  /* ---------------- send flow ---------------- */
  /* memory integrity: the model writes <mem> facts itself, but the USER's own
   * words are the ground truth for the VALUES inside them. If the model's
   * write garbles a name ('agusiagrya'), transcribe the verbatim string the
   * user actually typed. The model still decides WHAT to store and WHEN to
   * recall — this only guarantees the stored bytes match the user's text. */
  function verbatimFromUser(text, memText){
    const mem = memText.toLowerCase();
    const nm = mem.match(/user's name is ([a-z0-9]+)/);
    if (nm){
      const pat = text.toLowerCase().match(/\b(?:my names?\s+(?:is|are)|my name\s+is|i'm|im|call me|name's)\s+([a-z0-9]+)/);
      if (pat && pat[1] !== nm[1]) return memText.replace(new RegExp(nm[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), pat[1]);
    }
    const dg = mem.match(/user has a (dog|cat) called ([a-z0-9]+)/);
    if (dg){
      const pat = text.toLowerCase().match(new RegExp("(?:" + dg[1] + "|pet|puppy|kitten)(?:'s name)? (?:is|is called|called|named)?\\s*([a-z0-9]+)"));
      if (pat && pat[1] !== dg[2]) return memText.replace(new RegExp(dg[2].replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), pat[1]);
    }
    return memText;
  }

  async function send(){
    const text = input.value.trim();
    if (!text || state.loading || !state.model || state.generating) return;
    if (Date.now() - state.lastSend < 600) return;      // click cooldown
    state.lastSend = Date.now();
    input.value = ""; input.style.height = "auto";
    bubble("user", text);
    state.generating = true; state.abort = false;
    setBtn("stop");                                     // click again = stop
    loadState.textContent = "thinking…";

    const ui = aiMessage();
    const tel = [];
    const addTel = line => { tel.push(line);
      if (!state.model.hasThink) ui.thinkBody.textContent = tel.join("\n"); };
    const thinkStart = Date.now();

    try {
      /* 1) GAME INTENT — semantic analysis by the network's intent head */
      const it = await state.model.intent(text, modePersona().tok);
      const [, TTT, RPS, DICE, ASK] = [0,1,2,3,4];
      const names = ["chat", "tic tac toe", "rock paper scissors", "dice", "games menu"];
      addTel(`intent: ${names[it.label]} (${Math.round(Math.max(...it.probs)*100)}%)`);

      /* 2) retrieval from long-term memory (embeddings from the same model) */
      const qemb = await state.model.embed(text);
      const memRows = await retrieve(qemb, 2);
      addTel(memRows.length
        ? `recalled ${memRows.length} memor${memRows.length>1?"ies":"y"}: ${memRows.map(r=>r.text).join(" | ")}`
        : "no long-term memory matches this");

      /* 3) generate — thinking streams into the panel, answer streams into the bubble.
       * Math-shaped turns decode near-greedy: the model memorized exact sums,
       * sampling noise would only corrupt recall (still the model doing the math). */
      const isMath = /\b\d+\s*[+x\-]\s*\d+/.test(text) || /^count to \d+/.test(text.trim());
      const promptIds = await buildPromptIds(text, memRows, state.model);
      let answer = "", stopped = false, tokCount = 0;
      const gen = await state.model.generate(promptIds, {
        maxThink: 34, maxNew: 72, temp: isMath ? 0.12 : 0.65, topK: isMath ? 4 : 20, topP: 0.9,
        shouldAbort: () => state.abort,
        onThink: (txt, done) => {
          if (state.model.hasThink) ui.thinkBody.textContent = txt;
          if (done){ ui.closeThink(Math.max(1, Math.round((Date.now()-thinkStart)/1000))); }
          chat.scrollTop = chat.scrollHeight;
        },
        onToken: (t, n) => {
          tokCount = n;
          if (!state.model.hasThink) addTel(`composing reply… ${n} tokens`);
          answer = t; ui.bubble.textContent = t;
          chat.scrollTop = chat.scrollHeight;
        },
      });
      if (!state.model.hasThink) ui.closeThink();
      answer = gen.text || answer;
      stopped = gen.aborted;
      ui.bubble.textContent = answer + (stopped && answer ? " ⏹" : "");
      if (!answer) ui.bubble.textContent = stopped ? "(stopped)" : "…";

      /* 4) learn: the model itself decided what to save via its <mem> markers */
      for (let mtext of gen.mems){
        mtext = verbatimFromUser(text, mtext);
        state.sessionMems.push(mtext);
        if (state.sessionMems.length > 20) state.sessionMems.shift();
        await MEM.add({ text: mtext, emb: await state.model.embed(mtext), t: Date.now() });
      }
      await refreshMemList();

      state.history.push({ u: text, a: answer });
      if (state.history.length > 12) state.history.shift();

      /* 5) open the game if the network decided it's an invitation */
      if (!stopped){
        GAMES_setRuntime(state.model, state.pers);
        if (it.label === TTT) GAMES.tictactoe("You're on — board's below. Your move first.");
        else if (it.label === RPS) GAMES.rps("Pick your throw below.");
        else if (it.label === DICE) GAMES.dice("Two dice, three rounds. Highest total takes each one.");
      }
      loadState.textContent = `${state.size} ready`;
    } catch (err){
      console.error(err);
      ui.bubble.textContent = "Neural hiccup: " + err.message;
      loadState.textContent = `${state.size} ready`;
    }
    state.generating = false; state.abort = false;
    setBtn("cooldown");
    setTimeout(() => { if (!state.generating) setBtn("idle"); }, 900);   // anti double-fire cooldown
    input.focus();
  }

  /* ---------------- wiring ---------------- */
  sendBtn.addEventListener("pointerdown", e => e.preventDefault());  // never steal focus (Enter double-fire bug)
  sendBtn.onclick = () => {
    if (state.generating){
      // accidental double-click must NOT kill generation: stop arms after 1.2s
      if (Date.now() - state.lastSend < 1200) return;
      state.abort = true; loadState.textContent = "stopping…"; return;
    }
    send();
  };
  input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); send(); } });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 130) + "px"; });

  document.querySelectorAll("#modelSeg button").forEach(b => b.onclick = () => { if (!state.generating) loadModel(b.dataset.m); });

  document.querySelectorAll(".mode-item").forEach(x => x.onclick = () => setMode(x.dataset.mode));
  $("#btnMode").onclick = (e) => { e.stopPropagation(); $("#modeMenu").classList.toggle("open"); };
  document.addEventListener("click", (e) => {
    if (!$("#modeWrap").contains(e.target)) $("#modeMenu").classList.remove("open");
  });

  function setPers(p){
    state.pers = p; localStorage.setItem("sigma.pers", p);
    document.querySelectorAll(".pers-item").forEach(x => x.classList.toggle("active", x.dataset.p === p));
    $("#customPrompt").style.display = p === "custom" ? "block" : "none";
    chipPers.textContent = p === "custom" ? "custom" : p;
    if (state.mode === "shakes") setMode("chat");    // persona pick leaves shakes mode
  }
  document.querySelectorAll(".pers-item").forEach(x => x.onclick = () => setPers(x.dataset.p));
  $("#customPrompt").addEventListener("input", e => { state.custom = e.target.value; localStorage.setItem("sigma.custom", state.custom); });

  $("#btnClearMem").onclick = async () => { await MEM.clear(); refreshMemList(); };

  /* ---------------- pixel art studio (32x32 diffusion, real generation) ---------------- */
  const PX = { gen: null, loading: false, busy: false };
  async function pixelStudio(){
    GAMES.closeAll();
    const dock = document.getElementById("gameDock");
    const p = document.createElement("div");
    p.className = "game pixel-game";
    p.innerHTML = `
      <div class="game-head"><div class="game-title">Pixel Studio</div>
        <div class="game-note">32×32 diffusion · denoises pure noise · every image unique</div>
        <button class="game-x" title="close">✕</button></div>
      <div class="px-row">
        <div class="px-classes">${SIGMA_PIXEL.CLASSES.map((c,i) =>
          `<button class="px-cls${i===0?" active":""}" data-c="${i}">${c}</button>`).join("")}</div>
        <button class="rps-btn" id="pxGo" style="flex:0 0 auto;padding:10px 16px">✨ Generate</button>
      </div>
      <div class="px-canvas-wrap"><canvas id="pxCanvas" width="256" height="256"></canvas>
        <div id="pxStatus" class="px-status">weights not loaded yet</div></div>
      <div class="px-actions"><a id="pxDl" class="px-dl" download="sigma-pixel.png" style="display:none">⤓ Download PNG</a></div>`;
    p.querySelector(".game-x").onclick = () => dock.innerHTML = "";
    dock.appendChild(p);
    dock.scrollIntoView({behavior:"smooth", block:"end"});
    const status = p.querySelector("#pxStatus"), canvas = p.querySelector("#pxCanvas");
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    let cls = 0;
    p.querySelectorAll(".px-cls").forEach(b => b.onclick = () => {
      cls = +b.dataset.c;
      p.querySelectorAll(".px-cls").forEach(x => x.classList.toggle("active", x === b));
    });
    /* lazy-load weights on first open */
    if (!PX.gen && !PX.loading){
      PX.loading = true;
      status.textContent = "loading diffusion weights…";
      try {
        PX.gen = await SIGMA_PIXEL.PixelGen.load(pr => {
          status.textContent = `loading diffusion weights… ${Math.round(pr*100)}%`;
        });
        status.textContent = "ready — pick a subject and generate";
      } catch (e){ console.error(e); status.textContent = "load failed — check connection, it caches after"; }
      PX.loading = false;
    }
    p.querySelector("#pxGo").onclick = async () => {
      if (!PX.gen || PX.busy) return;
      PX.busy = true;
      const btn = p.querySelector("#pxGo"); btn.disabled = true;
      try {
        const img = await PX.gen.generate(cls, { steps: 40, temp: 1.1, onStep: (i, n) => {
          status.textContent = `denoising… step ${i}/${n}`;
        }});
        const off = new OffscreenCanvas(32, 32);
        off.getContext("2d").putImageData(new ImageData(img, 32, 32), 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(off, 0, 0, 256, 256);
        status.textContent = `done — a ${SIGMA_PIXEL.CLASSES[cls]} invented from noise`;
        const dl = p.querySelector("#pxDl");
        dl.href = canvas.toDataURL("image/png"); dl.style.display = "inline-block";
      } catch (e){ console.error(e); status.textContent = "generation hiccup: " + e.message; }
      btn.disabled = false; PX.busy = false;
    };
  }
  $("#btnPixel").onclick = pixelStudio;
  $("#btnNew").onclick = () => {
    if (state.generating) state.abort = true;
    state.history = [];
    state.sessionMems = [];
    if (state.model) state.model.reset();
    greet(); GAMES.closeAll();
  };
  $("#btnMenu").onclick = () => $("#sidebar").classList.toggle("open");
  document.addEventListener("click", e => {
    if (window.innerWidth <= 820 && !$("#sidebar").contains(e.target) && e.target.id !== "btnMenu")
      $("#sidebar").classList.remove("open");
  });

  /* PWA install */
  let deferredPrompt = null;
  window.addEventListener("beforeinstallprompt", e => { e.preventDefault(); deferredPrompt = e; $("#btnInstall").style.display = "block"; });
  $("#btnInstall").onclick = async () => { if (deferredPrompt){ deferredPrompt.prompt(); deferredPrompt = null; $("#btnInstall").style.display = "none"; } };
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js");

  /* boot */
  $("#customPrompt").value = state.custom;
  setPersInit();
  setMode(state.mode);
  greet();
  MEM.open().then(refreshMemList);
  loadModel(state.size);

  function setPersInit(){
    const p = state.pers;
    document.querySelectorAll(".pers-item").forEach(x => x.classList.toggle("active", x.dataset.p === p));
    $("#customPrompt").style.display = p === "custom" ? "block" : "none";
    chipPers.textContent = p === "custom" ? "custom" : p;
  }
})();
