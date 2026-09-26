"""
FastAPI app that serves the trained Q-learning agent as a playable game.

Two endpoints carry the whole game:
  POST /game/new   -- start a game, choosing which mark the human plays
  POST /game/move  -- human plays a move, agent replies, in one round trip

Combining "human moves" and "agent replies" into a single endpoint
(rather than the frontend calling one endpoint for the human's move and
a separate one to ask for the agent's reply) means the frontend only
ever needs to make one request per human turn.
"""
import sys
from pathlib import Path

# Locate the models directory relative to this file (main.py)
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to 'backend/app'
MODEL_PATH = BASE_DIR / "models" / "q_table.pkl"
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.game.engine import new_board, apply_move, valid_moves, winner, is_draw, other_player
from app.rl.agent import QLearningAgent
from app.api.schemas import NewGameRequest, NewGameResponse, MoveRequest, MoveResponse
from app.api.session_store import game_store

agent = QLearningAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained Q-table ONCE, when the server starts -- not on
    every request. Unpickling and holding it in memory is cheap; doing
    it per-request would be wasteful and would also risk race
    conditions if the file were being rewritten mid-request.
    """
    agent.load(MODEL_PATH)
    yield


app = FastAPI(title="Tic-Tac-Toe RL", lifespan=lifespan)

# In development, allow any origin so the frontend (served separately,
# possibly on a different port) can call this API from the browser.
# In production, replace "*" with the actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _determine_status(board: tuple, human_mark: str, agent_mark: str) -> str:
    """Shared win/draw/in-progress check, used by both endpoints so the logic lives in one place."""
    w = winner(board)
    if w == human_mark:
        return "human_win"
    elif w == agent_mark:
        return "agent_win"
    elif is_draw(board):
        return "draw"
    return "in_progress"


@app.post("/game/new", response_model=NewGameResponse)
def new_game(request: NewGameRequest):
    human_mark = request.human_mark
    agent_mark = other_player(human_mark)
    board = new_board()
    status = "in_progress"

    # If the human chose to play second, the agent's opening move
    # happens immediately, before the game is even handed back to the client.
    if agent_mark == "X":
        action = agent.choose_action(board, valid_moves(board), training=False)
        board = apply_move(board, action, agent_mark)
        status = _determine_status(board, human_mark, agent_mark)

    game_id = game_store.create(board, human_mark, agent_mark)
    return NewGameResponse(
        game_id=game_id,
        board=list(board),
        human_mark=human_mark,
        agent_mark=agent_mark,
        status=status,
    )


@app.post("/game/move", response_model=MoveResponse)
def make_move(request: MoveRequest):
    session = game_store.get(request.game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail=f"Game already finished: {session.status}")
    if session.board[request.position] != " ":
        raise HTTPException(status_code=400, detail="That cell is already occupied")

    # 1. Apply the human's move.
    board = apply_move(session.board, request.position, session.human_mark)
    status = _determine_status(board, session.human_mark, session.agent_mark)
    agent_move = None

    # 2. If the human's move didn't end the game, the agent replies --
    #    training=False means it always plays its single best-known move.
    if status == "in_progress":
        agent_move = agent.choose_action(board, valid_moves(board), training=False)
        board = apply_move(board, agent_move, session.agent_mark)
        status = _determine_status(board, session.human_mark, session.agent_mark)

    game_store.update(request.game_id, board, status)

    return MoveResponse(board=list(board), status=status, agent_move=agent_move)


@app.get("/game/{game_id}", response_model=MoveResponse)
def get_game(game_id: str):
    """Fetch the current state of a game -- useful if the frontend reloads mid-game."""
    session = game_store.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return MoveResponse(board=list(session.board), status=session.status, agent_move=None)


"""
Serve the frontend's static files (index.html, style.css, script.js) from
this same server, so one command gives you the whole app on one port.
IMPORTANT: this must be registered LAST. 
Starlette (which FastAPI is built on) matches routes in the order they were added, and a mount at
"/" would otherwise swallow every request -- including /game/new --
before it ever reached our actual API routes above. 
Registering it after all the @app.get/@app.post routes means FastAPI checks those
specific routes first, and only falls back to serving a static file
when nothing else matched.
"""

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")