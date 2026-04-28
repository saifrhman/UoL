"""
Dueling Double DQN agent for LunarLander (COMP532 CA-2).

Submodules
----------
network        -> DuelingQNetwork (PyTorch)
replay_buffer  -> ReplayBuffer (uniform)
agent          -> DuelingDoubleDQNAgent + AgentConfig
train          -> training loop (1000 episodes by default)
evaluate       -> greedy evaluation and GIF recording
plotting       -> reward and loss figures
"""

from .agent import AgentConfig, DuelingDoubleDQNAgent
from .network import DuelingQNetwork
from .replay_buffer import ReplayBuffer
from .train import make_env, train
from .evaluate import evaluate, record_gif
from .plotting import plot_losses, plot_rewards

__all__ = [
    "AgentConfig",
    "DuelingDoubleDQNAgent",
    "DuelingQNetwork",
    "ReplayBuffer",
    "make_env",
    "train",
    "evaluate",
    "record_gif",
    "plot_rewards",
    "plot_losses",
]
