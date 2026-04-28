"""Chess agents for the COMP532 optional extension."""

from .base import BaseAgent
from .heuristic import HeuristicAgent, RandomAgent, evaluate
from .llm import LLMAgent
from .drl import DRLAgent, ValueNet, ChessReplay
from .stockfish_agent import StockfishAgent
from .encoding import board_to_tensor

__all__ = [
    "BaseAgent",
    "HeuristicAgent",
    "RandomAgent",
    "LLMAgent",
    "DRLAgent",
    "ValueNet",
    "ChessReplay",
    "StockfishAgent",
    "board_to_tensor",
    "evaluate",
]
