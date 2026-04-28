"""
Plotting utilities for training curves.

Matplotlib only - no seaborn dependency, deliberately. The two required
figures are produced by `plot_rewards()` and `plot_losses()`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def _ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average for visual smoothing."""
    out = np.zeros_like(x, dtype=np.float64)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def plot_rewards(
    rewards: np.ndarray,
    mean100: np.ndarray | None = None,
    out_path: str = "plots/rewards.png",
    solved_score: float = 200.0,
    title: str = "Training rewards on LunarLander",
) -> str:
    """Episode reward curve with 100-episode rolling mean."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rewards = np.asarray(rewards)
    x = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, rewards, alpha=0.30, color="#1f77b4", label="Episode reward")

    if mean100 is None:
        # Compute a rolling mean if not supplied.
        if len(rewards) >= 100:
            cs = np.cumsum(np.insert(rewards, 0, 0))
            mean100 = (cs[100:] - cs[:-100]) / 100.0
            ax.plot(
                np.arange(100, len(rewards) + 1),
                mean100,
                color="#1f77b4",
                lw=2.0,
                label="100-episode mean",
            )
    else:
        mean100 = np.asarray(mean100)
        ax.plot(x, mean100, color="#1f77b4", lw=2.0, label="100-episode mean")

    ax.axhline(
        solved_score,
        ls="--",
        color="#2ca02c",
        lw=1.2,
        label=f"Solved threshold (={int(solved_score)})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_losses(
    losses: np.ndarray,
    out_path: str = "plots/loss.png",
    title: str = "Training loss vs. gradient step",
) -> str:
    """Training loss curve over gradient updates (smoothed for readability)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    losses = np.asarray(losses)
    x = np.arange(1, len(losses) + 1)
    smoothed = _ema(losses, alpha=0.01)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, losses, alpha=0.20, color="#d62728", label="Per-step loss")
    ax.plot(x, smoothed, color="#d62728", lw=2.0, label="EMA-smoothed")
    ax.set_yscale("log")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Huber loss (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
