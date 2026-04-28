"""
Dueling Double DQN (D3QN) agent.

Combines two stabilising ideas on top of vanilla DQN:

1. Double DQN (van Hasselt et al., AAAI 2016) - decouples action selection
   from action evaluation in the bootstrap target:
       a*  = argmax_a Q_online(s', a)
       y   = r + gamma * (1 - done) * Q_target(s', a*)
   This reduces the well-known maximisation bias of standard DQN.

2. Dueling architecture (Wang et al., ICML 2016) - separates V(s) and
   A(s, a). See network.py for details.

Exploration uses linearly-decayed epsilon-greedy.
The target network is synchronised by Polyak (soft) averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .network import DuelingQNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    """All hyperparameters in one place. Defaults tuned for LunarLander."""

    state_dim: int = 8
    action_dim: int = 4
    hidden: int = 128

    gamma: float = 0.99            # discount factor
    lr: float = 5e-4               # Adam learning rate
    batch_size: int = 64
    buffer_capacity: int = 100_000
    min_buffer_for_train: int = 1_000   # warm-up steps before training

    # Soft target update: target <- tau * online + (1 - tau) * target
    tau: float = 1e-3
    update_every: int = 4          # one gradient step every k env steps

    # Epsilon-greedy schedule
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995   # multiplicative per episode

    # Optimisation
    grad_clip_norm: float = 10.0
    seed: int = 42


class DuelingDoubleDQNAgent:
    """Dueling Double DQN agent for discrete-action environments."""

    def __init__(self, cfg: AgentConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Online and target networks
        self.online_net = DuelingQNetwork(
            cfg.state_dim, cfg.action_dim, cfg.hidden
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            cfg.state_dim, cfg.action_dim, cfg.hidden
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Target net is never trained directly.
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=cfg.lr
        )

        self.replay = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            state_dim=cfg.state_dim,
            device=self.device,
            seed=cfg.seed,
        )

        self.epsilon = cfg.epsilon_start
        self.train_step = 0

    # ------------------------------------------------------------------ #
    # Action selection                                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action; greedy=True forces argmax (for evaluation)."""
        if (not greedy) and (np.random.random() < self.epsilon):
            return int(np.random.randint(self.cfg.action_dim))

        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.online_net(s)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Learning step                                                      #
    # ------------------------------------------------------------------ #
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Push transition; perform one gradient step every `update_every` env steps."""
        self.replay.push(state, action, reward, next_state, done)
        self.train_step += 1

        if (
            len(self.replay) < self.cfg.min_buffer_for_train
            or self.train_step % self.cfg.update_every != 0
        ):
            return None

        loss = self._learn()
        return loss

    def _learn(self) -> float:
        """One Q-learning update. Returns scalar loss for logging."""
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)

        # Current Q(s, a)
        q_pred = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double-DQN target: select with online, evaluate with target.
        with torch.no_grad():
            next_actions = self.online_net(s2).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            target = r + self.cfg.gamma * (1.0 - d) * next_q

        # Smooth-L1 (Huber) is more robust to occasional reward outliers
        # (e.g. -100 crash vs +100 land).
        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.cfg.grad_clip_norm
        )
        self.optimizer.step()

        # Polyak (soft) target update
        with torch.no_grad():
            for p_online, p_target in zip(
                self.online_net.parameters(), self.target_net.parameters()
            ):
                p_target.data.mul_(1.0 - self.cfg.tau)
                p_target.data.add_(self.cfg.tau * p_online.data)

        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Bookkeeping                                                        #
    # ------------------------------------------------------------------ #
    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay, called once per episode."""
        self.epsilon = max(
            self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self.train_step,
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.cfg.epsilon_end)
        self.train_step = ckpt.get("train_step", 0)
