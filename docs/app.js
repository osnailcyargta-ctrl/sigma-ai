/* SIGMA app: chat UI + personality + 3-layer memory + game intent wiring +
 * Z.ai-style live thinking panel + send cooldown / stop control. */
(function(){
  if (window.ort) { ort.env.wasm.wasmPaths = "./"; ort.env.wasm.numThreads = 1; }
  const $ = s => document.querySelector(s);
  const chat = $("#chat"), input = $("#input"), sendBtn = $("#btnSend");
  const chipModel = $("#chipModel"), chipPers = $("#chipPers"), loadState = $("#loadState");

  const BLOCK = 256;                     // model context window (positional emb)
  const state = {
    size: localStorage.getItem("sigma.size") || "50m",
    pers: localStorage.getItem("sigma.pers") || "sigma",
    custom: localStorage.getItem("sigma.custom") || "",
    history: [],          // [{u, a}] working memory (context window)
    sessionMems: [],      // the model's OWN <mem> writes this session — always replayed
    model: null, tk: null, loading: false,
    generating: false, abort: false, lastSend: 0,
  };

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
                 "50m":"50,362,240 params · ppl 1.15 · open-vocab copy · flagship"};
      $("#modelInfo").textContent = P[size] || "";
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
    const keep = BLOCK - (model.hasThink ? 34 : 2) - 74;
    const head = `<bos><pers:${state.pers}>` +
      (state.pers === "custom" ? (state.custom.trim() || "you are helpful, honest and a little witty") + "<nl>" : "");
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
      const it = await state.model.intent(text, state.pers);
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
      for (const mtext of gen.mems){
        state.sessionMems.push(mtext);
        if (state.sessionMems.length > 20) state.sessionMems.shift();
        await MEM.add({ text: mtext, emb: await state.model.embed(mtext), t: Date.now() });
      }
      await refreshMemList();

      state.history.push({ u: text, a: answer });
      if (state.history.length > 12) state.history.shift();

      /* 5) open the game if the network decided it's an invitation */
      if (!stopped){
        if (it.label === TTT) GAMES.tictactoe("You're on — board's below. Your move first.");
        else if (it.label === RPS) GAMES.rps("Pick your throw below.");
        else if (it.label === DICE) GAMES.dice("Dice are hot. Two per throw, highest total wins the round.");
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

  function setPers(p){
    state.pers = p; localStorage.setItem("sigma.pers", p);
    document.querySelectorAll(".pers-item").forEach(x => x.classList.toggle("active", x.dataset.p === p));
    $("#customPrompt").style.display = p === "custom" ? "block" : "none";
    chipPers.textContent = p === "custom" ? "custom" : p;
  }
  document.querySelectorAll(".pers-item").forEach(x => x.onclick = () => setPers(x.dataset.p));
  $("#customPrompt").addEventListener("input", e => { state.custom = e.target.value; localStorage.setItem("sigma.custom", state.custom); });

  $("#btnClearMem").onclick = async () => { await MEM.clear(); refreshMemList(); };
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
  setPers(state.pers);
  greet();
  MEM.open().then(refreshMemList);
  loadModel(state.size);
})();
