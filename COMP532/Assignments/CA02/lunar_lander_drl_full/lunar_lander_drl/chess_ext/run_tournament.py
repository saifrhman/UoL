"""
Run the full ablation tournament for the COMP532 chess extension.

Round-robin matches between five agents:
    - Random      (uniform random over legal moves; required baseline)
    - LLM-stub    (LLM-pattern agent, offline simulator)
    - Heuristic   (material + PST, 1-ply greedy)
    - DRL         (trained value-network agent, 1-ply look-ahead)
    - Stockfish-0 (reference; weak Stockfish at skill 0, depth 1)

Each pairing plays N games with alternating colours. Outputs:
    chess_ext/results/tournament.json    -- full per-pairing results
    chess_ext/results/winrate_table.csv  -- compact win-rate table for the report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_ext.agents import (
    RandomAgent, HeuristicAgent, LLMAgent, DRLAgent, StockfishAgent,
)
from chess_ext.tournament import play_match, save_results


def make_agents(drl_model_path: str, seed: int = 123):
    """Build a fresh set of all five agents."""
    return [
        RandomAgent(seed=seed),
        LLMAgent(backend="stub", model="stub-v1", illegal_rate=0.05, seed=seed),
        HeuristicAgent(seed=seed),
        DRLAgent(model_path=drl_model_path, seed=seed),
        StockfishAgent(skill=0, depth=1),
    ]


def round_robin(
    agents,
    n_games: int,
    max_plies: int,
    save_dir: str,
    pgn_dir: str,
):
    """Play every (i, j) with i < j and store results."""
    os.makedirs(save_dir, exist_ok=True)

    pairings = []
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            pairings.append((i, j))

    matrix: Dict[str, Dict[str, dict]] = {}
    for a in agents:
        matrix[a.name] = {b.name: {} for b in agents if a.name != b.name}

    for (i, j) in pairings:
        a, b = agents[i], agents[j]
        print(f"\n========== {a.name}  vs  {b.name}  ({n_games} games) ==========")
        sub_pgn = os.path.join(pgn_dir, f"{a.name}_vs_{b.name}")
        res = play_match(
            a, b, n_games=n_games, max_plies=max_plies,
            verbose=True, pgn_dir=sub_pgn,
        )
        matrix[a.name][b.name] = {
            "wins": res.a_wins, "losses": res.b_wins, "draws": res.draws,
            "winrate": res.a_winrate,
        }
        matrix[b.name][a.name] = {
            "wins": res.b_wins, "losses": res.a_wins, "draws": res.draws,
            "winrate": res.b_winrate,
        }
        save_results(res, os.path.join(save_dir, f"match_{a.name}_vs_{b.name}.json"))

    # Aggregate per-agent
    per_agent: Dict[str, dict] = {}
    for a_name, opps in matrix.items():
        total_w = sum(d["wins"]   for d in opps.values())
        total_l = sum(d["losses"] for d in opps.values())
        total_d = sum(d["draws"]  for d in opps.values())
        total_g = total_w + total_l + total_d
        per_agent[a_name] = {
            "wins": total_w, "losses": total_l, "draws": total_d,
            "games": total_g,
            "winrate": (total_w + 0.5 * total_d) / max(1, total_g),
        }

    with open(os.path.join(save_dir, "tournament.json"), "w") as f:
        json.dump({"matrix": matrix, "per_agent": per_agent,
                   "n_games_per_pair": n_games}, f, indent=2)

    # CSV table for the report
    csv_path = os.path.join(save_dir, "winrate_table.csv")
    names = [a.name for a in agents]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *names, "Total"])
        for a_name in names:
            row = [a_name]
            for b_name in names:
                if a_name == b_name:
                    row.append("--")
                else:
                    d = matrix[a_name][b_name]
                    row.append(f"{d['wins']}-{d['losses']}-{d['draws']} ({d['winrate']:.0%})")
            row.append(f"{per_agent[a_name]['winrate']:.0%}")
            w.writerow(row)

    return matrix, per_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=10,
                    help="Games per pairing (alternating colours).")
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--drl-model", default="chess_ext/models/drl_value.pt")
    ap.add_argument("--save-dir",  default="chess_ext/results")
    ap.add_argument("--pgn-dir",   default="chess_ext/games")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    agents = make_agents(args.drl_model, seed=args.seed)
    try:
        matrix, per_agent = round_robin(
            agents, args.n_games, args.max_plies, args.save_dir, args.pgn_dir,
        )
        print("\n========== Per-agent totals ==========")
        for name, d in sorted(per_agent.items(), key=lambda x: -x[1]["winrate"]):
            print(f"  {name:30s}  W{d['wins']:3d} L{d['losses']:3d} D{d['draws']:3d}"
                  f"  ({d['winrate']:.1%} over {d['games']} games)")
    finally:
        for a in agents:
            try: a.close()
            except Exception: pass


if __name__ == "__main__":
    main()
