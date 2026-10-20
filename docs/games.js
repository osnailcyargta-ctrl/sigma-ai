/* Game suite v9: tic tac toe PLAYED BY THE MODEL (no preset, no minimax —
 * the network reads the board text and picks the cell itself, every move),
 * rock paper scissors, dice (locked: 2 dice, 3 rounds).
 * Each game is a self-contained panel docked under the chat. */
(function(){
  const dock = document.getElementById("gameDock");
  const el = (h) => { const d = document.createElement("div"); d.innerHTML = h.trim(); return d.firstChild; };

  /* runtime injected by app.js: { model, pers } */
  const RT = { model: null, pers: "sigma" };
  window.GAMES_setRuntime = (model, pers) => { RT.model = model; RT.pers = pers || "sigma"; };

  function closeAll(){ dock.innerHTML = ""; }
  function panel(title, note, bodyHtml){
    closeAll();
    const p = el(`<div class="game">
      <div class="game-head"><div class="game-title">${title}</div>
        <div class="game-note">${note}</div><button class="game-x" title="close">✕</button></div>
      ${bodyHtml}</div>`);
    p.querySelector(".game-x").onclick = closeAll;
    dock.appendChild(p);
    dock.scrollIntoView({behavior:"smooth", block:"end"});
    return p;
  }
  function say(p, text){
    let l = p.querySelector(".game-line");
    if (!l){ l = el(`<div class="game-line"></div>`); p.appendChild(l); }
    l.textContent = text;
  }

  /* ---------------- TIC TAC TOE — the model plays ---------------- */
  /* The AI is ⭘ (o). On its turn we send the board as text to the network:
   *   <user>board: x.o..x... you are o. your move.<ai>
   * and the model answers "<think>analysis</think>5." — trained on every
   * reachable position with minimax-optimal labels. Digits of occupied cells
   * are banned from sampling (legality), but WHICH empty cell to take is the
   * model's own decision from its own weights. No lookup table anywhere. */
  function tictactoe(aiLine){
    const p = panel("Tic Tac Toe", "you are ✕ and move first · the model reads the board and picks every ⭘ itself", `
      <div class="ttt-grid">${Array.from({length:9},(_,i)=>`<button class="ttt-cell" data-i="${i}"></button>`).join("")}</div>
      <div class="score" id="tttScore">score — you 0 · ai 0 · draw 0</div>`);
    say(p, aiLine || "Board's up. Your move first.");
    const cells = [...p.querySelectorAll(".ttt-cell")];
    let board = Array(9).fill(null), over = false, aiThinking = false;
    let sc = {you:0, ai:0, d:0};
    const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    const winner = b => { for (const [a,b2,c] of LINES) if (b[a] && b[a]===b[b2] && b[a]===b[c]) return b[a];
                          return b.every(x=>x) ? "draw" : null; };
    const bstr = b => b.map(c => c === "x" ? "x" : c === "o" ? "o" : ".").join("");
    function draw(){
      cells.forEach((c,i)=>{ c.textContent = board[i]==="x" ? "✕" : board[i]==="o" ? "⭘" : "";
                             c.disabled = !!board[i] || over || aiThinking; });
    }
    function score(){ p.querySelector("#tttScore").textContent = `score — you ${sc.you} · ai ${sc.ai} · draw ${sc.d}`; }
    function end(w){
      over = true;
      if (w === "x"){ sc.you++; say(p, "You win this one. Rematch? Tap any cell."); }
      else if (w === "o"){ sc.ai++; say(p, "Three in a row — the model takes it. Again?"); }
      else { sc.d++; say(p, "Draw. Classic. Go again?"); }
      score(); draw();
      setTimeout(()=>{ board = Array(9).fill(null); over = false; draw(); }, 900);
    }
    async function aiMove(){
      const model = RT.model;
      if (!model){ // model not loaded yet — can't play
        say(p, "Model is still loading — your move is on the board, waiting for my weights.");
        return;
      }
      aiThinking = true; draw();
      say(p, "Σ reading the board…");
      const boardText = bstr(board);
      const prompt = `<bos><pers:${RT.pers}><user>board: ${boardText} you are o. your move.<ai>`;
      const ban = new Set();
      for (let i = 0; i < 9; i++)
        if (board[i]) { const id = model.tk.stoi[String(i)]; if (id != null) ban.add(id); }
      let moveId = -1, thought = "";
      for (let attempt = 0; attempt < 2 && moveId < 0; attempt++){
        const res = await model.generate(model.tk.encode(prompt), {
          maxThink: 30, maxNew: 6, temp: 0.12, topK: 4, topP: 0.9,
          ban: attempt === 0 ? ban : new Set(),
          onThink: (txt) => { thought = txt; say(p, "Σ: " + txt); },
        });
        const m = (res.text || "").match(/[0-8]/);
        if (m){ const v = +m[0]; if (!board[v]) moveId = v; }
      }
      aiThinking = false;
      if (moveId < 0){ // last resort: uniform empty cell (practically never happens)
        const empt = board.map((c,i)=>c?-1:i).filter(i=>i>=0);
        moveId = empt[Math.floor(Math.random()*empt.length)];
      }
      board[moveId] = "o"; draw();
      say(p, thought ? `Σ: ${thought}` : "");
      const w = winner(board);
      if (w) end(w);
    }
    cells.forEach(c => c.onclick = () => {
      const i = +c.dataset.i;
      if (board[i] || over || aiThinking) return;
      board[i] = "x"; draw();
      let w = winner(board);
      if (w) return end(w);
      aiMove().then(draw);
    });
    draw();
  }

  /* ---------------- ROCK PAPER SCISSORS ---------------- */
  function rps(aiLine){
    const p = panel("Rock Paper Scissors", "first feeling wins — pick your throw", `
      <div class="rps-row">
        <button class="rps-btn" data-t="rock">✊</button>
        <button class="rps-btn" data-t="paper">✋</button>
        <button class="rps-btn" data-t="scissors">✌️</button>
      </div>
      <div class="score" id="rpsScore">score — you 0 · ai 0</div>`);
    say(p, aiLine || "Pick your throw.");
    let sc = {you:0, ai:0}, hist = [];
    const beats = {rock:"scissors", paper:"rock", scissors:"paper"};
    p.querySelectorAll(".rps-btn").forEach(b => b.onclick = () => {
      const you = b.dataset.t;
      // AI: counter the user's most frequent recent throw (pattern-aware, fair RNG on ties)
      const freq = {rock:0, paper:0, scissors:0};
      hist.slice(-6).forEach(h => freq[h]++);
      const guess = Object.keys(freq).sort((a,b2)=>freq[b2]-freq[a])[0];
      const ai = (Math.random() < 0.65 && hist.length >= 3) ? beats[guess] : ["rock","paper","scissors"][Math.floor(Math.random()*3)];
      hist.push(you);
      let msg = `you ${you} vs ai ${ai} — `;
      if (you === ai){ msg += "tie."; }
      else if (beats[you] === ai){ sc.you++; msg += "you win the round!"; }
      else { sc.ai++; msg += "AI takes the round."; }
      p.querySelector("#rpsScore").textContent = `score — you ${sc.you} · ai ${sc.ai}`;
      say(p, msg);
    });
  }

  /* ---------------- DICE — locked: 2 dice, 3 rounds ---------------- */
  function dice(aiLine){
    const p = panel("Dice Duel", "2 dice per throw · 3 rounds · highest total takes the round", `
      <div class="dice-row">
        <button class="rps-btn" id="dRoll" style="font-size:16px;flex:0 0 auto;padding:10px 18px">🎲 Roll</button>
        <div class="die" id="d0">·</div><div class="die" id="d1">·</div>
        <div class="die" id="d2">·</div><div class="die" id="d3">·</div>
      </div>
      <div class="score" id="dScore">round 1 of 3 — you 0 · ai 0</div>
      <div class="score" id="dFinal" style="min-height:0"></div>`);
    say(p, aiLine || "Two dice, three rounds. Roll when ready.");
    let round = 0, sc = {you:0, ai:0}, done = false;
    const pips = ["·","⚀","⚁","⚂","⚃","⚄","⚅"];
    const show = (vals) => vals.forEach((v,i)=>{ const d = p.querySelector("#d"+i); d.textContent = pips[v]; d.classList.remove("roll"); void d.offsetWidth; d.classList.add("roll"); });
    p.querySelector("#dRoll").onclick = () => {
      if (done) return;
      const mine = [1+Math.floor(Math.random()*6), 1+Math.floor(Math.random()*6)];
      const theirs = [1+Math.floor(Math.random()*6), 1+Math.floor(Math.random()*6)];
      show([mine[0], mine[1], theirs[0], theirs[1]]);
      const ms = mine[0]+mine[1], ts = theirs[0]+theirs[1];
      let msg = `round ${round+1}: you ${mine.join("+")}=${ms} vs ai ${theirs.join("+")}=${ts} — `;
      if (ms > ts){ sc.you++; msg += "round is yours!"; }
      else if (ts > ms){ sc.ai++; msg += "round to the model."; }
      else { msg += "dead tie, no score."; }
      round++;
      p.querySelector("#dScore").textContent =
        round < 3 ? `round ${round+1} of 3 — you ${sc.you} · ai ${sc.ai}` : `final — you ${sc.you} · ai ${sc.ai}`;
      if (round >= 3){
        done = true;
        msg += sc.you > sc.ai ? " GAME OVER: you win 3 rounds of dice! GG!"
             : sc.ai > sc.you ? " GAME OVER: the model wins this one. Rematch?"
             : " GAME OVER: all square. Dead even.";
        p.querySelector("#dRoll").textContent = "🎲 Rematch";
        p.querySelector("#dRoll").onclick = () => { closeAll(); dice("Fresh board. Two dice, three rounds. Roll."); };
      }
      say(p, msg);
    };
  }

  window.GAMES = { tictactoe, rps, dice, closeAll };
})();
