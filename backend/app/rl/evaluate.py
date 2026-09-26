"""
Exhaustive unbeatability verification.

Runs a random sample of games in isolation and it's easy to miss a
weakness in a rarely-visited branch. So instead we recursively explore
EVERY optimal move minimax could make, and EVERY move tied for best in
the agent's own Q-table, at every play -- and confirm that on every
single resulting path, the agent never loses. This is exhaustive over
the "optimal play" subtree, not a sample, so if it reports zero losses,
that's a proof, not a statistic.
"""

import sys
from pathlib import Path

# Add the 'backend' folder (2 levels up from train.py) to Python's import path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.game.engine import (
    new_board, valid_moves, apply_move, winner, is_draw, other_player,
)
from app.rl.agent import QLearningAgent
from app.rl.minimax import minimax_best_moves


def agent_best_moves(agent: QLearningAgent, board: tuple, valid: list[int]) -> list[int]:
    """All actions tied for the agent's highest Q-value from this board."""
    q_values = [agent.get_q(board, a) for a in valid]
    best_q = max(q_values)
    return [a for a, q in zip(valid, q_values) if abs(q - best_q) < 1e-9]


def exhaustive_verify(
    agent: QLearningAgent,
    agent_mark: str,
    board: tuple = None,
    current_player: str = "X",
    stats: dict = None,
) -> dict:
    """
    Recursively explore the full "both sides play optimally" subtree.
    The agent's moves branch over every Q-value tie; the opponent's
    moves branch over every minimax-optimal choice. Returns counts of
    how many resulting games were wins/draws/losses for the agent, and
    the actual board paths for any losses found (should be empty).
    """
    if board is None:
        board = new_board()
    if stats is None:
        stats = {"win": 0, "draw": 0, "loss": 0, "loss_examples": []}

    w = winner(board)
    if w == agent_mark:
        stats["win"] += 1
        return stats
    elif w is not None:
        stats["loss"] += 1
        if len(stats["loss_examples"]) < 3:
            stats["loss_examples"].append(board)
        return stats
    elif is_draw(board):
        stats["draw"] += 1
        return stats

    valid = valid_moves(board)
    if current_player == agent_mark:
        candidates = agent_best_moves(agent, board, valid)
    else:
        candidates = minimax_best_moves(board, current_player)

    for action in candidates:
        next_board = apply_move(board, action, current_player)
        exhaustive_verify(agent, agent_mark, next_board, other_player(current_player), stats)

    return stats


def exhaustive_verify_vs_any_move(
    agent: QLearningAgent,
    agent_mark: str,
    board: tuple = None,
    current_player: str = "X",
    stats: dict = None,
) -> dict:
    """
    Like exhaustive_verify, but the opponent branches over EVERY legal
    move, not just minimax-optimal ones. This proves the agent can't be
    beaten by ANY opponent behavior -- a careless human included -- not
    just a perfect one. The full tic-tac-toe game tree is small enough
    (a few hundred thousand paths at most) that this is still fast.
    """
    if board is None:
        board = new_board()
    if stats is None:
        stats = {"win": 0, "draw": 0, "loss": 0, "loss_examples": []}

    w = winner(board)
    if w == agent_mark:
        stats["win"] += 1
        return stats
    elif w is not None:
        stats["loss"] += 1
        if len(stats["loss_examples"]) < 3:
            stats["loss_examples"].append(board)
        return stats
    elif is_draw(board):
        stats["draw"] += 1
        return stats

    valid = valid_moves(board)
    candidates = agent_best_moves(agent, board, valid) if current_player == agent_mark else valid

    for action in candidates:
        next_board = apply_move(board, action, current_player)
        exhaustive_verify_vs_any_move(agent, agent_mark, next_board, other_player(current_player), stats)

    return stats


def run_full_verification(agent: QLearningAgent) -> None:
    for agent_mark, first_mover in [("X", "agent moves first"), ("O", "minimax moves first")]:
        stats = exhaustive_verify(agent, agent_mark)
        total = stats["win"] + stats["draw"] + stats["loss"]
        print(f"\nAgent plays {agent_mark} ({first_mover}) -- {total} optimal-play paths explored")
        print(f"  wins:   {stats['win']}")
        print(f"  draws:  {stats['draw']}")
        print(f"  LOSSES: {stats['loss']}")
        if stats["loss_examples"]:
            from app.game.engine import render
            print("  Example losing final position(s):")
            for board in stats["loss_examples"]:
                print(render(board))
                print()


if __name__ == "__main__":
    agent = QLearningAgent()
    agent.load("app/models/q_table.pkl")
    run_full_verification(agent)