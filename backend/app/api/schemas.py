"""
Request/response shapes for the API.

Pydantic models double as documentation here: FastAPI uses them to
auto-generate the OpenAPI schema (visible at /docs), and to validate
incoming requests before our route code ever runs -- e.g. a request
missing `position`, or with `position` as a string, is rejected with a
clear 422 error automatically, before we'd have to check for it by hand.
"""

from typing import Optional
from pydantic import BaseModel, Field


class NewGameRequest(BaseModel):
    """
    Which mark the HUMAN wants to play. Defaults to "X" (human moves first). 
    If the human picks "O", the agent moves first and its
    opening move is included in the response.
    """
    human_mark: str = Field(default="X", pattern="^[XO]$")


class NewGameResponse(BaseModel):
    game_id: str
    board: list[str]          # 9 cells, ' ' | 'X' | 'O' -- JSON has no tuples, so a list
    human_mark: str
    agent_mark: str
    status: str                # "in_progress" | "human_win" | "agent_win" | "draw"


class MoveRequest(BaseModel):
    game_id: str
    position: int = Field(ge=0, le=8)  # FastAPI rejects out-of-range positions before our code runs


class MoveResponse(BaseModel):
    board: list[str]
    status: str
    agent_move: Optional[int] = None  # which cell the agent just played, if it moved