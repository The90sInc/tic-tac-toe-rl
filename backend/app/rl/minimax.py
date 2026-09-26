"""
Minimax solver for tic-tac-toe.

This is NOT learned -- it's a brute-force search over the entire game
tree, so it is optimal by construction. It exists purely as a ground
truth to test the trained Q-learning agent against. Tic-tac-toe's game
tree is tiny (at most ~5,478 reachable board states), so minimax with
simple memoization is instant -- no alpha-beta pruning needed.

minimax_value(board, player) returns the outcome of `board` from
`player`'s perspective, assuming both sides play perfectly from here on:
    +1 = this player wins
     0 = draw
    -1 = this player loses
"""

from functools import lru_cache

from app.game.engine import valid_moves, apply_move, winner, is_draw, other_player


@lru_cache(maxsize=None)
def minimax_value(board: tuple, player: str) -> int:
    w = winner(board)
    if w == player:
        return 1
    elif w is not None:
        return -1
    elif is_draw(board):
        return 0

    # This player's best outcome is the negation of the best outcome
    # the OPPONENT can force from any resulting position (zero-sum).
    best = -2
    for action in valid_moves(board):
        next_board = apply_move(board, action, player)
        value = -minimax_value(next_board, other_player(player))
        best = max(best, value)
    return best


def minimax_best_moves(board: tuple, player: str) -> list[int]:
    """All moves that are equally optimal for `player` from `board` (often more than one)."""
    best_value = minimax_value(board, player)
    best_moves = []
    for action in valid_moves(board):
        next_board = apply_move(board, action, player)
        value = -minimax_value(next_board, other_player(player))
        if value == best_value:
            best_moves.append(action)
    return best_moves