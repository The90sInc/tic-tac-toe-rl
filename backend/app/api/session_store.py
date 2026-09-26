"""
In-memory storage for in-progress games.

Each HTTP request is handled independently -- the server has no memory
of "the last request" by default. So when a human plays a move, we need
some way to know what board that game was on before this request. We
solve that here with the simplest possible approach: a dict keyed by a
random game ID, held in the server process's memory.

This is intentionally NOT how you'd do it at scale: if you ever run
more than one server process (for load balancing), each process would
have its own separate dict, and a game could "disappear" if a later
request lands on a different process. The fix at that point is to swap
this dict for Redis (same get/set/delete interface, shared across
processes) -- but for a single-instance deployment, this is simpler,
faster, and has one less moving part to run.
"""

import uuid
from typing import Optional

class GameSession:
    def __init__(self, board: tuple, human_mark: str, agent_mark: str, status: str = "in_progress"):
        self.board = board
        self.human_mark = human_mark
        self.agent_mark = agent_mark
        self.status = status  # "in_progress" | "human_win" | "agent_win" | "draw"


class GameSessionStore:
    def __init__(self):
        self._games: dict[str, GameSession] = {}

    def create(self, board: tuple, human_mark: str, agent_mark: str) -> str:
        game_id = str(uuid.uuid4())
        self._games[game_id] = GameSession(board, human_mark, agent_mark)
        return game_id

    def get(self, game_id: str) -> Optional[GameSession]:
        return self._games.get(game_id)

    def update(self, game_id: str, board: tuple, status: str) -> None:
        session = self._games[game_id]
        session.board = board
        session.status = status

    def delete(self, game_id: str) -> None:
        self._games.pop(game_id, None)


# A single shared instance used by the whole app -- see main.py.
game_store = GameSessionStore()