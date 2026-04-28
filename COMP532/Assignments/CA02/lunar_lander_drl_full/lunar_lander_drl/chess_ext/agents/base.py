"""
Common base class for chess agents.

All agents share a tiny interface so the tournament harness can pit any
combination against each other and a UCI shim can wrap any of them.
"""

from __future__ import annotations

import abc

import chess


class BaseAgent(abc.ABC):
    """Abstract base class for a chess move-selecting agent."""

    name: str = "base"

    @abc.abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move:
        """Return one *legal* move for the side to move on `board`."""

    # Optional hooks ---------------------------------------------------- #
    def reset(self) -> None:
        """Called at the start of each new game. Override if stateful."""

    def close(self) -> None:
        """Called once at shutdown. Override to release subprocess handles."""
