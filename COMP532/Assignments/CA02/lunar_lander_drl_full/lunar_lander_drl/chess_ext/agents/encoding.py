"""
Tensor encoding for chess positions.

Used by the DRL agent. Converts a `chess.Board` into a fixed-shape tensor
suitable for a small CNN.

Encoding (8 x 8 x 18 channels):

    Channels  0- 5: white pieces (P, N, B, R, Q, K) - one-hot per square
    Channels  6-11: black pieces (P, N, B, R, Q, K) - one-hot per square
    Channel 12-15: castling rights (white K, white Q, black k, black q)
                   - filled with 1.0 across the whole 8x8 plane if available
    Channel 16:    side to move (1.0 = White, 0.0 = Black) - constant plane
    Channel 17:    en-passant target (one-hot at the EP square if any)

Total: 18 planes -> 18 * 64 = 1152 input features.
"""

from __future__ import annotations

import chess
import numpy as np


PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
          chess.ROOK, chess.QUEEN, chess.KING)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Return an (18, 8, 8) float32 tensor encoding `board`."""
    t = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes (white = 0..5, black = 6..11)
    for sq, piece in board.piece_map().items():
        plane = PIECES.index(piece.piece_type)
        if piece.color == chess.BLACK:
            plane += 6
        r, c = chess.square_rank(sq), chess.square_file(sq)
        t[plane, 7 - r, c] = 1.0  # row 0 = rank 8 for visual consistency

    # Castling rights (planes 12..15)
    if board.has_kingside_castling_rights(chess.WHITE):  t[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): t[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  t[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): t[15, :, :] = 1.0

    # Side to move (plane 16)
    if board.turn == chess.WHITE:
        t[16, :, :] = 1.0

    # En-passant target (plane 17)
    if board.ep_square is not None:
        r, c = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        t[17, 7 - r, c] = 1.0

    return t
