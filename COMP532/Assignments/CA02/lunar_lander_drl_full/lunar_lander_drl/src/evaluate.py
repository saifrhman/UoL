"""
Evaluation and visualisation utilities for the trained DQN agent.

- evaluate(): runs N greedy episodes and reports mean/std reward.
- record_gif(): saves a GIF showing a greedy roll-out for the report.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

from .agent import AgentConfig, DuelingDoubleDQNAgent
from .train import make_env


def evaluate(
    model_path: str,
    n_episodes: int = 30,
    seed: int = 1234,
    cfg: AgentConfig | None = None,
) -> Tuple[float, float, List[float]]:
    """Run `n_episodes` greedy roll-outs. Returns (mean, std, all_returns)."""
    cfg = cfg or AgentConfig(seed=seed)
    env = make_env()
    agent = DuelingDoubleDQNAgent(cfg)
    agent.load(model_path)
    agent.online_net.eval()

    returns: List[float] = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_r = 0.0
        done = False
        while not done:
            a = agent.select_action(state, greedy=True)
            state, r, terminated, truncated, _ = env.step(a)
            ep_r += r
            done = terminated or truncated
        returns.append(ep_r)

    env.close()
    return float(np.mean(returns)), float(np.std(returns)), returns


def record_gif(
    model_path: str,
    out_path: str = "videos/agent_demo.gif",
    n_episodes: int = 3,
    seed: int = 0,
    fps: int = 30,
    cfg: AgentConfig | None = None,
    frame_stride: int = 2,
) -> str:
    """Save a GIF of greedy roll-outs.

    Frames are streamed to disk via `imageio.get_writer` rather than
    accumulated in a list, to keep memory bounded. `frame_stride` keeps
    every k-th frame to further reduce file size and memory pressure.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cfg = cfg or AgentConfig()
    env = make_env(render_mode="rgb_array")
    agent = DuelingDoubleDQNAgent(cfg)
    agent.load(model_path)
    agent.online_net.eval()

    # GIF writer with effective fps after stride
    writer = imageio.get_writer(out_path, mode="I", fps=max(1, fps // frame_stride), loop=0)
    try:
        for ep in range(n_episodes):
            state, _ = env.reset(seed=seed + ep)
            done = False
            t = 0
            last_frame = None
            while not done:
                if t % frame_stride == 0:
                    last_frame = env.render()
                    writer.append_data(np.asarray(last_frame))
                a = agent.select_action(state, greedy=True)
                state, _, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                t += 1
            # Hold the final frame between episodes for visual clarity.
            if last_frame is not None:
                for _ in range(5):
                    writer.append_data(np.asarray(last_frame))
    finally:
        writer.close()
        env.close()
    return out_path
