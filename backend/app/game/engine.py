"""
Core tic-tac-toe rules, with no knowledge of RL or the web.
 
Board representation
---------------------
A board is a tuple of 9 characters, each ' ' (empty), 'X', or 'O'.
Positions are indexed like a phone keypad:
 
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
 
We use a tuple (not a list) because it must be hashable: the RL agent's
Q-table will use board states as dictionary keys.
"""

EMPTY = " "
PLAYER_X = "X"
PLAYER_O = "O"

#Every possible way to get three in a row, as index triples.
WIN_LINES = [
    (0, 1, 2),  # top row
    (3, 4, 5),  # middle row
    (6, 7, 8),  # bottom row
    (0, 3, 6),  # left column
    (1, 4, 7),  # middle column
    (2, 5, 8),  # right column
    (0, 4, 8),  # diagonal top-left to bottom-right
    (2, 4, 6),  # diagonal top-right to bottom-left
]

def new_board() -> tuple:
    """Return a fresh, empty board."""
    return tuple(EMPTY for _ in range(9))

def valid_moves(board: tuple) -> list[int]:
    """Return a list of empty cell indices where a move can be played."""
    return [i for i, cell in enumerate(board) if cell == EMPTY]

def apply_move(board: tuple, position: int, player: str) -> tuple:
    """
    Return a NEW board with `player`'s mark placed at `position`.
    Does not mutate the input board (boards are immutable tuples).
    Raises ValueError if the move is illegal.
    """
    if player not in (PLAYER_X, PLAYER_O):
        raise ValueError(f"Invalid player: {player!r}")
    if position not in range(9):
        raise ValueError(f"Invalid position: {position!r}")
    if board[position] != EMPTY:
        raise ValueError(f"Position {position} is already occupied by {board[position]!r}")

    board_list = list(board)
    board_list[position] = player
    return tuple(board_list)

def winner(board: tuple) -> str | None:
    """Return "X" or "O" if that player has three in a row, else None."""
    for a,b,c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None

def is_draw(board: tuple) -> bool:
    """ Return True if the board is full and there is no winner."""
    return EMPTY not in board and winner(board) is None

def is_game_over(board: tuple) -> bool:
    """Return True if the game is over (win or draw)."""
    return winner(board) is not None or is_draw(board)

def other_player(player: str) -> str:
    """Return the Opponent's mark"""
    return PLAYER_O if player == PLAYER_X else PLAYER_X

def render(board: tuple) -> str:
    """Human-readable string representation of the board."""
    rows = []
    for r in range(3):
        cells = board[r*3:(r+1)*3]
        rows.append(" | ".join(cells))
    return "\n---------\n".join(rows)