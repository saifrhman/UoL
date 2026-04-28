"""
Deep RL chess agent.

Architecture
------------
A small convolutional **value network** ``V_theta(s) -> R`` predicts the
position's value from the side-to-move's perspective in the range
[-1, +1] (loss .. win). Move selection at play time is one-ply
look-ahead:

    a* = argmin_a  V_theta(push(s, a))

(Minimum because after my move the opponent is to move, so a higher
``V_theta`` from *their* perspective means a worse outcome for *me*.)

Why a value network rather than a policy/Q network with a fixed move
vocabulary? Two reasons:

1. Chess has 4,672 distinct (from-sq, to-sq, promotion) pairs in
   AlphaZero's encoding. A value network is much smaller (~50 k
   parameters) and trains faster on the budget available here.
2. With 1-ply look-ahead and a static evaluation network, **legal moves
   are guaranteed by construction**: we only ever score positions that
   exist in the move-generator's output, so the agent cannot
   hallucinate illegal moves. This is the value-based equivalent of
   AlphaZero's MCTS-with-prior architecture.

Training
--------
Trained by **temporal-difference learning from games against a weak
Stockfish (skill 0, depth 1) opponent**, with reward shaping from a
heuristic eval to densify the otherwise sparse +/- 1 win/loss signal.

This *is* a deep-RL setup -- there is a neural network, an experience
buffer, target bootstrapping and gradient updates -- but it is *not*
AlphaZero, which used 5,000 first-generation TPUs over many days and
is not reproducible inside a 4 GB sandbox. We are explicit about that
in the report.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAgent
from .encoding import board_to_tensor


# ---------------------------------------------------------------------- #
# Network                                                                #
# ---------------------------------------------------------------------- #
class ValueNet(nn.Module):
    """Small CNN value network for chess positions.

    18-channel input -> 2x conv block -> global avg pool -> MLP -> tanh.
    ~50k parameters; designed to train on a CPU in tens of minutes.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),  # value in [-1, +1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


# ---------------------------------------------------------------------- #
# Agent                                                                  #
# ---------------------------------------------------------------------- #
class DRLAgent(BaseAgent):
    """Value-network agent with 1-ply look-ahead move selection."""

    name = "drl"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        epsilon: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.device = device or torch.device("cpu")
        self.net = ValueNet().to(self.device)
        if model_path and os.path.exists(model_path):
            sd = torch.load(model_path, map_location=self.device, weights_only=False)
            self.net.load_state_dict(sd["net"] if "net" in sd else sd)
        self.net.eval()
        self.epsilon = epsilon  # for exploration during evaluation tournaments
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)

        # 1-ply look-ahead: score each resulting child position from the
        # opponent's perspective; pick the move that minimises it.
        # Terminal short-circuits avoid wasting a forward pass.
        best_score = float("inf")
        best_moves: List[chess.Move] = []
        tensors = []
        cand_moves = []
        for mv in legal:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                return mv  # immediate mate
            if board.is_stalemate() or board.is_insufficient_material():
                # Treat draws as score 0 (neutral).
                tensors.append(board_to_tensor(board))
                cand_moves.append((mv, 0.0))
                board.pop()
                continue
            tensors.append(board_to_tensor(board))
            cand_moves.append((mv, None))
            board.pop()

        # Batch the forward pass
        if any(s is None for _, s in cand_moves):
            x = torch.from_numpy(np.stack(tensors)).to(self.device)
            v = self.net(x).cpu().numpy()  # value from opponent's POV
        else:
            v = np.zeros(len(cand_moves))

        for i, (mv, s) in enumerate(cand_moves):
            score = s if s is not None else float(v[i])
            if score < best_score - 1e-6:
                best_score = score
                best_moves = [mv]
            elif abs(score - best_score) < 1e-6:
                best_moves.append(mv)

        return self.rng.choice(best_moves)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(sd["net"] if "net" in sd else sd)
        self.net.eval()


# ---------------------------------------------------------------------- #
# Replay buffer for training                                             #
# ---------------------------------------------------------------------- #
class ChessReplay:
    """Stores (state_tensor, target_value) pairs for training the value net."""

    def __init__(self, capacity: int = 200_000, seed: int = 0):
        self.buf: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state: np.ndarray, value: float) -> None:
        self.buf.append((state, float(value)))

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(self.buf, k=min(batch_size, len(self.buf)))
        states = np.stack([b[0] for b in batch])
        values = np.array([b[1] for b in batch], dtype=np.float32)
        return torch.from_numpy(states), torch.from_numpy(values)
