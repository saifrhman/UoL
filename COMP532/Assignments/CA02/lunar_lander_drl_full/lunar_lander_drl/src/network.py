"""
Dueling Q-Network for Deep Q-Learning.

The network outputs Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)). This decoupling
of state-value V(s) and advantage A(s,a) follows Wang et al. (2016, ICML)
and improves policy evaluation when many actions have similar values, which
is true in LunarLander where most "hover" steps have low-impact actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """
    Dueling MLP for discrete-action Q-learning.

    Architecture:
        feature trunk:   state_dim -> hidden -> hidden  (ReLU)
        value head:      hidden -> hidden -> 1
        advantage head:  hidden -> hidden -> action_dim
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

    The mean-subtraction (rather than max) is used because it is more
    stable in practice, as recommended in the original Dueling DQN paper.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim

        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # Value stream V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

        # Advantage stream A(s, a)
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He-initialisation for ReLU layers; small final layer std for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, action_dim)."""
        z = self.trunk(state)
        v = self.value_head(z)                     # (B, 1)
        a = self.advantage_head(z)                 # (B, action_dim)
        # Identifiability: subtract mean advantage so V uniquely identifies value.
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
