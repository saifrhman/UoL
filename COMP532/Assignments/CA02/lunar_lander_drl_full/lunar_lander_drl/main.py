"""
Entry point: train -> plot -> evaluate -> record GIF.

Usage
-----
    python main.py --episodes 1000

To train fewer episodes for a quick smoke test:

    python main.py --episodes 50

After training, the best checkpoint by 100-episode mean reward is saved
to `models/d3qn_lunarlander.pt`, and `models/d3qn_lunarlander_final.pt`
contains the very last weights.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow `python main.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    AgentConfig,
    evaluate,
    plot_losses,
    plot_rewards,
    record_gif,
    train,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Dueling Double DQN on LunarLander")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--save-dir", type=str, default="results")
    p.add_argument("--model-path", type=str, default="models/d3qn_lunarlander.pt")
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the post-training greedy evaluation.",
    )
    p.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF recording (useful in headless CI).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = AgentConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        buffer_capacity=args.buffer_capacity,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    summary = train(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        cfg=cfg,
        save_dir=args.save_dir,
        model_path=args.model_path,
        seed=args.seed,
    )

    rewards = np.asarray(summary["rewards"])
    losses = np.asarray(summary["losses"])
    mean100 = np.asarray(summary["mean100"])

    plot_rewards(rewards, mean100, out_path="plots/rewards.png")
    plot_losses(losses, out_path="plots/loss.png")
    print("Saved plots to plots/rewards.png and plots/loss.png")

    if not args.skip_eval:
        mean, std, returns = evaluate(args.model_path, n_episodes=30, cfg=cfg)
        print(
            f"\nGreedy evaluation over 30 episodes: "
            f"mean = {mean:.2f}, std = {std:.2f}"
        )

    if not args.skip_gif:
        gif_path = record_gif(args.model_path, out_path="videos/agent_demo.gif")
        print(f"Saved demo GIF to {gif_path}")


if __name__ == "__main__":
    main()
