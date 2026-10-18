/* Game suite: tic tac toe (vs minimax AI), rock paper scissors, dice (2 or 3).
 * Each game is a self-contained panel docked under the chat. */
(function(){
  const dock = document.getElementById("gameDock");
  const el = (h) => { const d = document.createElement("div"); d.innerHTML = h.trim(); return d.firstChild; };

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

  /* ---------------- TIC TAC TOE (minimax, plays perfect) ---------------- */
  function tictactoe(aiLine){
    const p = panel("Tic Tac Toe", "you are ✕ · the model plays a perfect minimax ⭘", `
      <div class="ttt-grid">${Array.from({length:9},(_,i)=>`<button class="ttt-cell" data-i="${i}"></button>`).join("")}</div>
      <div class="score" id="tttScore">score — you 0 · ai 0 · draw 0</div>`);
    say(p, aiLine || "Board's up. Your move first.");
    const cells = [...p.querySelectorAll(".ttt-cell")];
    let board = Array(9).fill(null), over = false;
    let sc = {you:0, ai:0, d:0};
    const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    const winner = b => { for (const [a,b2,c] of LINES) if (b[a] && b[a]===b[b2] && b[a]===b[c]) return b[a];
                          return b.every(x=>x) ? "draw" : null; };
    function draw(){
      cells.forEach((c,i)=>{ c.textContent = board[i]==="x" ? "✕" : board[i]==="o" ? "⭘" : ""; c.disabled = !!board[i] || over; });
    }
    function score(){ p.querySelector("#tttScore").textContent = `score — you ${sc.you} · ai ${sc.ai} · draw ${sc.d}`; }
    function end(w){
      over = true;
      if (w === "x"){ sc.you++; say(p, "You win this one. Rematch? Tap any cell."); }
      else if (w === "o"){ sc.ai++; say(p, "Three in a row — AI takes it. Again?"); }
      else { sc.d++; say(p, "Draw. Classic. Go again?"); }
      score(); draw();
      setTimeout(()=>{ board = Array(9).fill(null); over = false; draw(); }, 900);
    }
    function best(b){ // minimax, o to move
      let bw, bs = -Infinity;
      for (let i=0;i<9;i++) if (!b[i]){
        b[i]="o"; const s = mini(b, false); b[i]=null;
        if (s > bs){ bs = s; bw = i; }
      }
      return bw;
      function mini(bd, isO){
        const w = winner(bd);
        if (w === "o") return 1; if (w === "x") return -1; if (w === "draw") return 0;
        let best = isO ? -Infinity : Infinity;
        for (let i=0;i<9;i++) if (!bd[i]){
          bd[i] = isO ? "o" : "x";
          const s = mini(bd, !isO);
          bd[i] = null;
          best = isO ? Math.max(best,s) : Math.min(best,s);
        }
        return best;
      }
    }
    cells.forEach(c => c.onclick = () => {
      const i = +c.dataset.i;
      if (board[i] || over) return;
      board[i] = "x"; draw();
      let w = winner(board);
      if (w) return end(w);
      const m = best(board.slice());
      board[m] = "o"; draw();
      w = winner(board);
      if (w) end(w);
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

  /* ---------------- DICE (2 or 3 dice per throw) ---------------- */
  function dice(aiLine){
    const p = panel("Dice Duel", "each throw rolls 2 dice (or 3) · highest total wins the round · most round-wins takes the game", `
      <div class="dice-row">
        <button class="rps-btn" id="dRoll" style="font-size:16px;flex:0 0 auto;padding:10px 18px">🎲 Roll</button>
        <button class="rps-btn" id="dMode" style="font-size:13px;flex:0 0 auto;padding:10px 14px">dice: 2</button>
        <div class="die" id="d0">·</div><div class="die" id="d1">·</div>
        <div class="die" id="d2" style="display:none">·</div>
        <div class="die" id="d3" style="display:none">·</div><div class="die" id="d4" style="display:none">·</div><div class="die" id="d5" style="display:none">·</div>
      </div>
      <div class="score" id="dScore">rounds — you 0 · ai 0 · round 0</div>`);
    say(p, aiLine || "Roll when ready.");
    let n = 2, sc = {you:0, ai:0}, round = 0;
    const pips = ["·","⚀","⚁","⚂","⚃","⚄","⚅"];
    const show = (ids) => ids.forEach((v,i)=>{ const d = p.querySelector("#d"+i); d.textContent = pips[v]; d.classList.remove("roll"); void d.offsetWidth; d.classList.add("roll"); });
    p.querySelector("#dMode").onclick = (e) => {
      n = n === 2 ? 3 : 2;
      e.target.textContent = "dice: " + n;
      [0,1,2,3,4,5].forEach(i => p.querySelector("#d"+i).style.display = (n===2 ? (i<2?"":"none") : (i>=2&&i<5?"":"none")));
      say(p, `Switched to ${n} dice per throw. Same rules: most round-wins takes it.`);
    };
    p.querySelector("#dRoll").onclick = () => {
      const mine = Array.from({length:n}, ()=>1+Math.floor(Math.random()*6));
      const theirs = Array.from({length:n}, ()=>1+Math.floor(Math.random()*6));
      show(n===2 ? [mine[0],mine[1],theirs[0],theirs[1]] : [mine[0],mine[1],mine[2],theirs[0],theirs[1],theirs[2]]);
      round++;
      const ms = mine.reduce((a,b)=>a+b), ts = theirs.reduce((a,b)=>a+b);
      let msg = `round ${round}: you ${mine.join("+")}=${ms} vs ai ${theirs.join("+")}=${ts} — `;
      if (ms > ts){ sc.you++; msg += "round is yours!"; }
      else if (ts > ms){ sc.ai++; msg += "round to the AI."; }
      else { msg += "dead tie, no score."; }
      p.querySelector("#dScore").textContent = `rounds — you ${sc.you} · ai ${sc.ai} · round ${round}`;
      if (sc.you !== sc.ai && (sc.you >= 5 || sc.ai >= 5 || round >= 10)){
        msg += sc.you > sc.ai ? " GAME: you win the whole thing! GG!" : " GAME: AI wins this one. Rematch?";
      }
      say(p, msg);
    };
  }

  window.GAMES = { tictactoe, rps, dice, closeAll };
})();
