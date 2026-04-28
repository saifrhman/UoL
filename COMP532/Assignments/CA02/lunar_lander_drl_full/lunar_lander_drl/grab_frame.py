"""Extract a single representative frame from a greedy roll-out (for the report)."""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import AgentConfig, DuelingDoubleDQNAgent
from src.train import make_env


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/d3qn_lunarlander.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "plots/demo_frame.png"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cfg = AgentConfig()
    env = make_env(render_mode="rgb_array")
    agent = DuelingDoubleDQNAgent(cfg)
    agent.load(model_path, map_location="cpu")
    agent.online_net.eval()

    state, _ = env.reset(seed=seed)
    best_frame = None
    step = 0
    while step < 1000:
        a = agent.select_action(state, greedy=True)
        state, r, terminated, truncated, _ = env.step(a)
        # Capture the frame just before touchdown for a visually informative still.
        if state[6] or state[7]:  # leg contact flags
            best_frame = env.render()
            break
        step += 1
        if terminated or truncated:
            best_frame = env.render()
            break
    if best_frame is None:
        best_frame = env.render()

    imageio.imwrite(out_path, np.array(best_frame))
    print(f"Wrote {out_path}")
    env.close()


if __name__ == "__main__":
    main()
