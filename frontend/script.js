// If the frontend is served BY the FastAPI backend itself (single-server
// setup), leave this as "" -- same origin, no CORS needed. If you're
// running the SPLIT deployment (frontend on GitHub Pages/Vercel, backend
// on Render), set this to your backend's full deployed URL instead, e.g.
// "https://your-app-name.onrender.com".
const API_BASE = "";

// Render's free tier can take 30-60 seconds to wake up from a cold
// start. Any status-setting function can call this to escalate the
// message if a request is taking a while, so a slow response reads as
// "waking up" rather than "broken".
function armWakeupWarning(initialDelayMs = 4000) {
  return setTimeout(() => {
    setStatus("UNIT WAS ASLEEP. WAKING UP... THIS CAN TAKE UP TO A MINUTE.");
  }, initialDelayMs);
}

const statusLine = document.getElementById("statusLine");
const board = document.getElementById("board");
const cells = Array.from(document.querySelectorAll(".cell"));
const newGameBtn = document.getElementById("newGameBtn");
const markSelect = document.getElementById("markSelect");
const lossOverlay = document.getElementById("lossOverlay");
const retryBtn = document.getElementById("retryBtn");

let gameId = null;
let humanMark = "X";
let agentMark = "O";

function setStatus(text) {
  statusLine.innerHTML = `&gt; ${text}<span class="cursor">_</span>`;
}

function renderBoard(cellValues) {
  cellValues.forEach((value, i) => {
    cells[i].textContent = value.trim() === "" ? "" : value;
  });
}

function setCellsEnabled(enabled) {
  cells.forEach((cell) => {
    // A cell should only ever be clickable if the game is active AND
    // the cell is still empty -- occupied cells stay disabled even
    // when the board is otherwise "live".
    const isEmpty = cell.textContent.trim() === "";
    cell.disabled = !(enabled && isEmpty);
  });
}

async function startNewGame() {
  humanMark = markSelect.value;
  setStatus("CONNECTING TO NEURAL DEFENSE UNIT...");
  const wakeupTimer = armWakeupWarning();

  try {
    const resp = await fetch(`${API_BASE}/game/new`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ human_mark: humanMark }),
    });
    clearTimeout(wakeupTimer);
    if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
    const data = await resp.json();

    gameId = data.game_id;
    agentMark = data.agent_mark;
    renderBoard(data.board);
    lossOverlay.classList.remove("active");

    if (data.status === "in_progress") {
      setCellsEnabled(true);
      setStatus(`GAME ACTIVE. YOU ARE '${humanMark}'. AWAITING YOUR MOVE.`);
    } else {
      // Extremely unlikely on turn 0, but handled for completeness.
      handleGameOver(data.status);
    }
  } catch (err) {
    clearTimeout(wakeupTimer);
    setStatus(`CONNECTION ERROR: ${err.message}. IS THE BACKEND RUNNING?`);
  }
}

async function playMove(position) {
  setCellsEnabled(false);
  setStatus("TRANSMITTING MOVE...");
  const wakeupTimer = armWakeupWarning();

  try {
    const resp = await fetch(`${API_BASE}/game/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: gameId, position }),
    });
    clearTimeout(wakeupTimer);
    if (!resp.ok) {
      const errBody = await resp.json();
      setStatus(`REJECTED: ${errBody.detail}`);
      setCellsEnabled(true);
      return;
    }
    const data = await resp.json();
    renderBoard(data.board);

    if (data.status === "in_progress") {
      setCellsEnabled(true);
      setStatus("AWAITING YOUR MOVE.");
    } else {
      handleGameOver(data.status);
    }
  } catch (err) {
    clearTimeout(wakeupTimer);
    setStatus(`CONNECTION ERROR: ${err.message}`);
    setCellsEnabled(true);
  }
}

function handleGameOver(status) {
  setCellsEnabled(false);
  if (status === "agent_win") {
    setStatus("GAME OVER: DEFEAT.");
    lossOverlay.classList.add("active");
  } else if (status === "human_win") {
    // Should never actually happen against a fully verified agent --
    // but if you retrain with fewer episodes and it's not yet perfect,
    // this will fire, which is itself a useful debugging signal.
    setStatus("GAME OVER: YOU WIN. (Unexpected -- check training/verification.)");
  } else if (status === "draw") {
    setStatus("GAME OVER: DRAW. NEURAL DEFENSE UNIT HOLDS.");
  }
}

cells.forEach((cell) => {
  cell.addEventListener("click", () => {
    const pos = Number(cell.dataset.pos);
    playMove(pos);
  });
});

newGameBtn.addEventListener("click", startNewGame);
retryBtn.addEventListener("click", () => {
  lossOverlay.classList.remove("active");
  startNewGame();
});

// Board starts empty and disabled until the player clicks "New Game".
setCellsEnabled(false);