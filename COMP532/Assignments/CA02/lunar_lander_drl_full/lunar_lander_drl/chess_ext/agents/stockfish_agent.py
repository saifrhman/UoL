"""
Stockfish wrapper agent.

Used as:

- a *strong opponent* during DRL training (skill 0, depth 1) - kept very
  weak so the DRL agent can actually win some games and bootstrap.
- a *reference oracle* in the tournament so we have an upper bound for
  the playing strength of all our agents.
"""

from __future__ import annotations

from typing import Optional

import chess
import chess.engine

from .base import BaseAgent


class StockfishAgent(BaseAgent):
    """Wrap a UCI-compatible Stockfish binary as a BaseAgent.

    Parameters
    ----------
    path : str
        Path to the stockfish executable. Default '/usr/games/stockfish'
        (Ubuntu apt install).
    skill : int
        Stockfish 'Skill Level' option (0 = weakest, 20 = strongest).
    depth : int | None
        If set, search to this fixed depth. Otherwise use ``time_limit``.
    time_limit : float
        Per-move time limit in seconds.
    """

    name = "stockfish"

    def __init__(
        self,
        path: str = "/usr/games/stockfish",
        skill: int = 0,
        depth: Optional[int] = 1,
        time_limit: float = 0.05,
    ):
        self.path = path
        self.skill = skill
        self.depth = depth
        self.time_limit = time_limit
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self.name = f"stockfish-skill{skill}"

    def _ensure(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Skill Level": self.skill})
        return self._engine

    def select_move(self, board: chess.Board) -> chess.Move:
        eng = self._ensure()
        if self.depth is not None:
            limit = chess.engine.Limit(depth=self.depth)
        else:
            limit = chess.engine.Limit(time=self.time_limit)
        result = eng.play(board, limit)
        return result.move

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except chess.engine.EngineTerminatedError:
                pass
            self._engine = None

    def __del__(self) -> None:
        self.close()
