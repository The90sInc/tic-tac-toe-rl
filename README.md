# Tic-Tac-Toe RL

A browser-playable tic-tac-toe game backed by a FastAPI service and a Q-learning agent. The agent is trained through self-play, and the API serves both the game page and its endpoints from one process.

## Project Structure

```text
tic-tac-toe_rl/
|-- backend/
|   |-- requirements.txt
|   `-- app/
|       |-- api/
|       |   |-- main.py          # FastAPI routes and static frontend hosting
|       |   |-- schemas.py       # Request and response validation
|       |   `-- session_store.py # In-memory game sessions
|       |-- game/
|       |   |-- engine.py        # Board rules and legal moves
|       |   `-- symmetry.py      # Canonicalizes rotated/reflected boards
|       |-- models/
|       |   `-- q_table.pkl      # Trained Q-values loaded at server startup
|       `-- rl/
|           |-- agent.py        # Q-learning policy and persistence
|           |-- train.py        # Self-play training
|           |-- evaluate.py     # Checks the agent against optimal play
|           `-- minimax.py      # Minimax reference strategy
`-- frontend/
    |-- index.html              # Game page
    |-- script.js               # UI behavior and API requests
    `-- style.css               # Terminal-inspired presentation
```

## How It Works

- The game engine represents the nine cells as an immutable tuple and enforces tic-tac-toe rules.
- The Q-learning agent learns from self-play. Board symmetry is canonicalized so rotations and reflections share Q-values.
- The trained table in `backend/app/models/q_table.pkl` is loaded once when the API starts. Training is not performed on each launch.
- The frontend sends requests to the same FastAPI origin. `POST /game/new` creates a session; `POST /game/move` records a human move and returns the agent's reply; `GET /game/{game_id}` retrieves the current board.
- Game sessions are held in process memory. Restarting the service clears active games, and multiple server instances are not supported without replacing the session store with shared storage.
- FastAPI also serves the frontend at `/`. Interactive API documentation is available at `/docs`.

The evaluator explores games against minimax-optimal opponent moves. It is a useful check of the learned policy; it is separate from the game server.

## Run Locally

Run these commands from the repository root. Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
uvicorn app.api.main:app --app-dir backend --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

On macOS or Linux, create and activate the virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn app.api.main:app --app-dir backend --reload
```

To retrain the agent or run the evaluation, from the repository root:

```bash
python backend/app/rl/train.py
python backend/app/rl/evaluate.py
```

Training overwrites `backend/app/models/q_table.pkl`. Keep that model file in the repository when deploying; the server requires it at startup.

## Deploy to Render

Deploy as a **Web Service** from the repository. Use the repository root as the service's Root Directory (leave the Root Directory field blank). The app serves `frontend/` from outside `backend/`, so setting the service root to `backend` will make the frontend unavailable.

Use these settings:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Root Directory | Leave blank (repository root) |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `uvicorn app.api.main:app --app-dir backend --host 0.0.0.0 --port $PORT` |

Render provides the `PORT` environment variable. No separate frontend service or database is needed for a single instance. Keep the deployed model file at `backend/app/models/q_table.pkl`. Because game sessions are in memory, run one service instance with the default single Uvicorn worker; active games are lost when the service restarts. Render's free web services may sleep when idle, so the first request after inactivity can take longer while the service starts.

After deployment, open the service URL for the game or append `/docs` to inspect the API.