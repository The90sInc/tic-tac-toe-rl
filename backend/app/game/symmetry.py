"""
Tic-tac-toe has 8 spatial symmetries (4 rotations x 2, including
reflections) that all preserve the rules of the game -- a strategy
that's correct for one board is correct for all 8 of its
rotated/reflected equivalents. Without accounting for this, a Q-table
has to independently (re)learn each equivalent position from scratch,
which wastes a huge fraction of training on redundant states.

canonical_form(board) maps any board to ONE consistent representative
among its 8 symmetric equivalents, plus the permutation used to get
there -- so callers can translate action indices consistently between
the real board and its canonical form.
"""

from functools import lru_cache


def _build_permutation(transform) -> tuple:
    """Given a (row, col) -> (row, col) transform, build the equivalent index permutation."""
    perm = [0] * 9
    for i in range(9):
        r, c = divmod(i, 3)
        nr, nc = transform(r, c)
        perm[i] = nr * 3 + nc
    return tuple(perm)


# The 8 symmetries of a square: identity, 3 rotations, and 4 reflections.
_TRANSFORMS = [
    lambda r, c: (r, c),         # identity
    lambda r, c: (c, 2 - r),     # rotate 90 clockwise
    lambda r, c: (2 - r, 2 - c), # rotate 180
    lambda r, c: (2 - c, r),     # rotate 270 clockwise
    lambda r, c: (r, 2 - c),     # flip horizontal
    lambda r, c: (2 - r, c),     # flip vertical
    lambda r, c: (c, r),         # transpose (main diagonal)
    lambda r, c: (2 - c, 2 - r), # anti-diagonal
]

SYMMETRY_PERMUTATIONS = tuple(_build_permutation(transform) for transform in _TRANSFORMS)


def apply_permutation(board: tuple, perm: tuple) -> tuple:
    """Return a new board where the piece at index i moves to index perm[i]."""
    new_board = [None] * 9
    for i in range(9):
        new_board[perm[i]] = board[i]
    return tuple(new_board)


def invert_permutation(perm: tuple) -> tuple:
    inv = [0] * 9
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


@lru_cache(maxsize=None)
def canonical_form(board: tuple) -> tuple:
    """
    Return (canonical_board, perm), where perm maps REAL board indices
    to CANONICAL board indices: apply_permutation(board, perm) == canonical_board.
    The canonical board is whichever of the 8 symmetric variants sorts
    lowest -- an arbitrary but consistent choice, so all 8 equivalent
    boards always map to the exact same canonical form.
    """
    best_board = None
    best_perm = None
    for perm in SYMMETRY_PERMUTATIONS:
        transformed = apply_permutation(board, perm)
        if best_board is None or transformed < best_board:
            best_board = transformed
            best_perm = perm
    return best_board, best_perm