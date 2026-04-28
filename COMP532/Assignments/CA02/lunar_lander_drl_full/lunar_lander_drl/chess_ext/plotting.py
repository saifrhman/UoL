"""Plotting utilities for the chess extension."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(summary_path: str, out_path: str = "chess_ext/plots/training.png") -> str:
    """Plot win-rate vs Random and Heuristic over training games."""
    with open(summary_path) as f:
        summary = json.load(f)
    log = summary["eval_log"]

    games = [e["game"] for e in log]
    wr_r  = [e["wr_random"] for e in log]
    wr_h  = [e["wr_heuristic"] for e in log]
    comb  = [e["combined"] for e in log]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(games, wr_r,  marker="o", lw=2, color="#1f77b4", label="vs Random")
    ax.plot(games, wr_h,  marker="s", lw=2, color="#d62728", label="vs Heuristic")
    ax.plot(games, comb,  marker="^", lw=2, color="#2ca02c", label="Combined", alpha=0.7)
    ax.axhline(0.5, ls=":", color="grey", lw=1, label="Coin-flip baseline")
    ax.set_xlabel("Training games played")
    ax.set_ylabel("Win-rate (wins + 0.5 draws)")
    ax.set_title("DRL chess agent: evaluation win-rate vs baselines")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_curve(losses_path: str, out_path: str = "chess_ext/plots/loss.png") -> str:
    losses = np.load(losses_path)
    if len(losses) == 0:
        return ""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # EMA smoothing for readability
    alpha = 0.01
    ema = np.zeros_like(losses, dtype=np.float64)
    ema[0] = losses[0]
    for i in range(1, len(losses)):
        ema[i] = alpha * losses[i] + (1 - alpha) * ema[i - 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(losses, alpha=0.25, color="#9467bd", label="Per-step loss")
    ax.plot(ema,    color="#9467bd", lw=2, label="EMA-smoothed")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("MSE loss on value targets")
    ax.set_title("Chess value-network training loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_tournament_matrix(
    tournament_path: str,
    out_path: str = "chess_ext/plots/tournament_matrix.png",
) -> str:
    """Render the head-to-head win-rate matrix as a heatmap."""
    with open(tournament_path) as f:
        data = json.load(f)
    matrix = data["matrix"]

    names = list(matrix.keys())
    n = len(names)
    M = np.full((n, n), np.nan, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a == b: continue
            M[i, j] = matrix[a][b]["winrate"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Head-to-head win-rate (row vs column)")
    ax.set_xlabel("Opponent (column)")
    ax.set_ylabel("Player (row)")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="grey")
            else:
                d = matrix[names[i]][names[j]]
                ax.text(j, i, f"{d['wins']}-{d['losses']}-{d['draws']}\n{d['winrate']:.0%}",
                        ha="center", va="center", fontsize=8,
                        color=("white" if M[i, j] < 0.4 or M[i, j] > 0.7 else "black"))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Win-rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
