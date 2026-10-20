/* SHAKESPEAGMA runtime — the same trained transformer as Sigma Chat, driven in
 * word-forge mode on its own page (no sidebar). Uses the trained
 * <pers:shakespeare> token when the checkpoint has it (v10 does); older tables
 * degrade to <pers:custom> + a style hint. The net's <think> phase streams
 * into the collapsible "the quill thinks" panel.
 * The quick-quill buttons map to the persona's TRAINED ask families (greeting,
 * praise, insult, socratic questions, bard lists, in-context style switch);
 * topic→verse-on-demand lands with the next training wave. */
(function(){
  if (window.ort) { ort.env.wasm.wasmPaths = "../"; ort.env.wasm.numThreads = 1; }
  const $ = s => document.querySelector(s);
  const state = {
    len: +(localStorage.getItem("sgm.len") || 90),
    model: null, forging: false,
  };

  function buildPromptIds(subject){
    const m = state.model;
    let head;
    if (m.supportsPers("shakespeare")) head = `<bos><pers:shakespeare>`;
    else head = `<bos><pers:custom>you speak in grand verse like shakespeare and ask wise questions like socrates<nl>`;
    return m.tk.encode(head + `<user>${subject}<ai>`);
  }

  /* ---------- page plumbing ---------- */
  const works = $("#works"), topic = $("#topic"), btnForge = $("#btnForge"), stateEl = $("#state");

  function newWork(subject){
    const note = $("#emptyNote"); if (note) note.remove();
    const w = document.createElement("article");
    w.className = "work";
    w.innerHTML = `
      <div class="work-tag">shakespeagma</div>
      <div class="quill open"><button class="quill-head" type="button">
        <span>the quill thinks</span><span class="chev">▾</span></button>
        <div class="quill-body"></div></div>
      <div class="work-subject-line">— “${subject}”</div>
      <div class="work-body verse"></div>
      <div class="work-foot">
        <button class="act-again">forge again</button>
        <button class="act-copy">copy</button>
      </div>`;
    const q = w.querySelector(".quill");
    w.querySelector(".quill-head").onclick = () => q.classList.toggle("open");
    works.appendChild(w);
    works.scrollTop = works.scrollHeight;
    return w;
  }

  async function forge(subject){
    if (state.forging || !state.model || !subject) return;
    state.forging = true; btnForge.disabled = true;
    const w = newWork(subject);
    const body = w.querySelector(".work-body");
    const qbody = w.querySelector(".quill-body");
    const q = w.querySelector(".quill");
    const cur = document.createElement("span"); cur.className = "cursor";
    body.after(cur);
    stateEl.textContent = "forging…";
    const t0 = Date.now();
    try {
      const promptIds = buildPromptIds(subject);
      await state.model.generate(promptIds, {
        maxThink: 40, maxNew: state.len, temp: 0.8, topK: 30, topP: 0.92,
        onThink: (txt, done) => {
          qbody.textContent = txt || "…";
          works.scrollTop = works.scrollHeight;
          if (done){
            if (txt && txt.trim()) q.classList.remove("open");
            q.querySelector("span").textContent =
              `the quill thinks · ${Math.max(1, Math.round((Date.now()-t0)/1000))}s`;
          }
        },
        onToken: t => {
          body.textContent = t || "…";
          works.scrollTop = works.scrollHeight;
        },
      });
      if (!body.textContent.trim()) body.textContent = "(the muse was silent — forge again)";
    } catch (err){
      console.error(err);
      body.textContent = "the quill snapped: " + err.message;
    }
    cur.remove();
    w.querySelector(".act-copy").onclick = () => navigator.clipboard && navigator.clipboard.writeText(body.textContent);
    w.querySelector(".act-again").onclick = () => forge(subject);
    state.forging = false; btnForge.disabled = !topic.value.trim();
    stateEl.textContent = "shakespeagma ready";
    topic.focus();
  }

  /* ---------- wiring ---------- */
  $("#stylePick").querySelectorAll(".style-btn").forEach(b => b.onclick = () => {
    topic.value = b.dataset.q;
    topic.dispatchEvent(new Event("input"));
    forge(topic.value.trim());
  });
  $("#lenPick").querySelectorAll(".len-btn").forEach(b => b.onclick = () => {
    state.len = +b.dataset.l; localStorage.setItem("sgm.len", state.len);
    $("#lenPick").querySelectorAll(".len-btn").forEach(x => x.classList.toggle("active", x === b));
  });
  topic.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); forge(topic.value.trim()); } });
  topic.addEventListener("input", () => { btnForge.disabled = state.forging || !topic.value.trim(); });
  btnForge.onclick = () => forge(topic.value.trim());

  /* ---------- model boot (same brain as chat) ---------- */
  (async () => {
    stateEl.textContent = "loading weights…";
    try {
      const size = ["v10","50m","22m"].includes(localStorage.getItem("sigma.size"))
        ? localStorage.getItem("sigma.size") : "v10";
      const m = await SIGMA.Sigma.load(size, p => { stateEl.textContent = `loading ${size}… ${Math.round(p*100)}%`; });
      state.model = m;
      btnForge.disabled = !topic.value.trim();
      stateEl.textContent = `shakespeagma ready · ${size}`;
    } catch (err){
      console.error(err);
      stateEl.textContent = "weight load failed — retry once online";
    }
  })();
})();
