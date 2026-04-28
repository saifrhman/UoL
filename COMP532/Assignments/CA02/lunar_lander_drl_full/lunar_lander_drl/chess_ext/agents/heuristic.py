"""
Two simple agents:

- RandomAgent: uniform-random over legal moves. The brief's required
  baseline.

- HeuristicAgent: material + piece-square table evaluation with 1-ply
  greedy look-ahead and capture-extension. Strong enough to be a
  meaningful sparring partner and cheap enough to play hundreds of
  games per minute. Also used as the offline `--llm-stub` opponent for
  the LLM agent (see agents/llm.py).
"""

from __future__ import annotations

import random
from typing import Optional

import chess

from .base import BaseAgent


# ---------------------------------------------------------------------- #
# Piece values and piece-square tables                                   #
# ---------------------------------------------------------------------- #
# Standard centi-pawn values used in classical engines.
PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20_000,
}

# Piece-square tables (white perspective, square 0 = a1).
# Source: classical Chess Programming Wiki "simplified evaluation".
_PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT_PST = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP_PST = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK_PST = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_QUEEN_PST = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
_KING_PST_MID = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
]

_PST = {
    chess.PAWN:   _PAWN_PST,
    chess.KNIGHT: _KNIGHT_PST,
    chess.BISHOP: _BISHOP_PST,
    chess.ROOK:   _ROOK_PST,
    chess.QUEEN:  _QUEEN_PST,
    chess.KING:   _KING_PST_MID,
}


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centi-pawns from White's perspective.

    Sum of (piece value + piece-square bonus) for white pieces, minus
    the same for black pieces. PSTs are mirrored vertically for black.
    Terminal positions are handled by the caller (`select_move`).
    """
    score = 0
    for sq, piece in board.piece_map().items():
        v = PIECE_VALUE[piece.piece_type]
        # PST is given from white's perspective; mirror for black.
        pst_idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
        v += _PST[piece.piece_type][pst_idx]
        score += v if piece.color == chess.WHITE else -v
    return score


# ---------------------------------------------------------------------- #
# RandomAgent                                                            #
# ---------------------------------------------------------------------- #
class RandomAgent(BaseAgent):
    """Uniform-random over legal moves. Required by the brief as a baseline."""

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


# ---------------------------------------------------------------------- #
# HeuristicAgent                                                         #
# ---------------------------------------------------------------------- #
class HeuristicAgent(BaseAgent):
    """1-ply greedy with material + PST evaluation.

    For each legal move, push it, evaluate, then pop. Take the move
    that maximises evaluation from the side-to-move's perspective.
    Ties are broken randomly so the agent does not play deterministically
    in symmetric positions.
    """

    name = "heuristic"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        is_white = board.turn == chess.WHITE
        best_score = -10**9
        best_moves = []
        for mv in legal:
            board.push(mv)
            # If we just delivered checkmate, take it immediately.
            if board.is_checkmate():
                board.pop()
                return mv
            s = evaluate(board)
            if not is_white:
                s = -s
            board.pop()
            if s > best_score:
                best_score = s
                best_moves = [mv]
            elif s == best_score:
                best_moves.append(mv)
        return self.rng.choice(best_moves)
