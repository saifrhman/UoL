"""
Training loop for LunarLander with the Dueling Double DQN agent.

LunarLander-v3 is the current (Gymnasium >= 1.0) name for the box2d
LunarLander environment formerly known as LunarLander-v2 in older
OpenAI Gym releases. It is identical in dynamics, observations and
reward shaping; only the version tag was bumped.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict
from typing import Dict, List

import gymnasium as gym
import numpy as np

from .agent import AgentConfig, DuelingDoubleDQNAgent


# Backwards-compatible env name lookup. Older Gym uses LunarLander-v2,
# Gymnasium >= 1.0 ships LunarLander-v3.
def make_env(env_id: str | None = None, **kwargs) -> gym.Env:
    candidates = (
        [env_id]
        if env_id is not None
        else ["LunarLander-v3", "LunarLander-v2"]
    )
    last_err: Exception | None = None
    for cid in candidates:
        try:
            return gym.make(cid, **kwargs)
        except Exception as e:  # pragma: no cover
            last_err = e
    raise RuntimeError(f"Could not create LunarLander env: {last_err}")


def train(
    n_episodes: int = 1000,
    max_steps: int = 1000,
    cfg: AgentConfig | None = None,
    save_dir: str = "results",
    model_path: str = "models/d3qn_lunarlander.pt",
    log_every: int = 20,
    solved_score: float = 200.0,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Train the agent for `n_episodes`.

    Returns a dict with per-episode rewards, per-update losses and the
    rolling mean reward used to decide when the task is solved.

    The agent is *not* stopped when the environment is "solved" (>= 200
    over the last 100 episodes); we keep training so the learning curve
    covers the full 0-1000 episode range required by the brief.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    cfg = cfg or AgentConfig(seed=seed)
    env = make_env()
    eval_seed = seed

    agent = DuelingDoubleDQNAgent(cfg)

    rewards: List[float] = []
    losses: List[float] = []        # one entry per gradient step
    mean100: List[float] = []
    window = deque(maxlen=100)
    best_mean = -np.inf
    solved_episode: int | None = None

    t_start = time.time()
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset(seed=eval_seed + ep)
        ep_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, greedy=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
            state = next_state
            ep_reward += reward
            if done:
                break

        agent.decay_epsilon()
        rewards.append(ep_reward)
        window.append(ep_reward)
        m100 = float(np.mean(window))
        mean100.append(m100)

        # Save best snapshot so far (by 100-ep rolling mean) once we have
        # enough episodes to make the moving average meaningful.
        if len(window) >= 50 and m100 > best_mean:
            best_mean = m100
            agent.save(model_path)

        if solved_episode is None and len(window) == 100 and m100 >= solved_score:
            solved_episode = ep

        if ep % log_every == 0 or ep == 1:
            print(
                f"Ep {ep:4d}/{n_episodes} | "
                f"R = {ep_reward:7.2f} | "
                f"mean100 = {m100:7.2f} | "
                f"eps = {agent.epsilon:.3f} | "
                f"buf = {len(agent.replay):6d} | "
                f"loss = {(losses[-1] if losses else float('nan')):.4f}",
                flush=True,
            )

        # Periodic intermediate snapshot so we have results even if the
        # process is interrupted (e.g. by a session timeout).
        if ep % 50 == 0:
            np.save(os.path.join(save_dir, "rewards.npy"), np.asarray(rewards))
            np.save(os.path.join(save_dir, "losses.npy"), np.asarray(losses))
            np.save(os.path.join(save_dir, "mean100.npy"), np.asarray(mean100))
            partial = {
                "rewards": rewards,
                "losses": losses,
                "mean100": mean100,
                "solved_episode": solved_episode,
                "best_mean100": float(best_mean if np.isfinite(best_mean) else -np.inf),
                "n_episodes": ep,
                "duration_seconds": time.time() - t_start,
                "config": asdict(cfg),
            }
            with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
                json.dump(partial, f, indent=2)
            agent.save(os.path.splitext(model_path)[0] + "_latest.pt")

    # Final checkpoint: always overwrite the last network so `evaluate.py`
    # can use the most recent weights even if no new best occurred.
    final_path = os.path.splitext(model_path)[0] + "_final.pt"
    agent.save(final_path)

    duration = time.time() - t_start
    summary = {
        "rewards": rewards,
        "losses": losses,
        "mean100": mean100,
        "solved_episode": solved_episode,
        "best_mean100": float(best_mean if np.isfinite(best_mean) else -np.inf),
        "n_episodes": n_episodes,
        "duration_seconds": duration,
        "config": asdict(cfg),
    }
    with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(save_dir, "rewards.npy"), np.asarray(rewards))
    np.save(os.path.join(save_dir, "losses.npy"), np.asarray(losses))
    np.save(os.path.join(save_dir, "mean100.npy"), np.asarray(mean100))

    print(
        f"\nTraining finished in {duration/60:.1f} min. "
        f"Best 100-ep mean reward: {best_mean:.2f}. "
        f"Solved at episode: {solved_episode}."
    )
    env.close()
    return summary
