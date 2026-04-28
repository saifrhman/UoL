"""
Training loop for the chess value network.

Approach
--------
Each "iteration" plays one game between the current value-network agent
and a weak Stockfish opponent (skill 0, depth 1). At every ply we
record the position tensor and the side to move so that, once the game
ends, we can back-propagate the terminal outcome (+1 win, -1 loss, 0
draw) to every state visited by the side that ended up with that
result. From each side's perspective:

    target_white = +1 if white won, -1 if black won, 0 if draw
    target_black = -target_white

This is essentially TD(1) (Monte-Carlo) value-function regression -
the simplest thing that works for chess given the budget. We also
blend in a heuristic shaping term:

    target = (1 - alpha) * outcome + alpha * tanh(eval / 600)

so the value net gets a useful gradient even from drawn or truncated
training games. ``alpha`` decays as training progresses.

The agent is periodically evaluated against Random and Heuristic
baselines and the best snapshot by combined win-rate is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_ext.agents import (  # noqa: E402
    DRLAgent, ChessReplay,
    RandomAgent, HeuristicAgent, StockfishAgent,
    board_to_tensor, evaluate as heur_eval,
)


# ---------------------------------------------------------------------- #
# Self-play game generator                                               #
# ---------------------------------------------------------------------- #
def play_training_game(
    drl: DRLAgent,
    opponent,
    drl_is_white: bool,
    max_plies: int,
    epsilon: float,
) -> Tuple[List[Tuple[np.ndarray, bool]], float, chess.Board]:
    """Play one game; return trajectory, white's terminal score in [-1,+1], final board."""
    board = chess.Board()
    trajectory: List[Tuple[np.ndarray, bool]] = []
    drl.epsilon = epsilon

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        # Record state before each move (from side-to-move's perspective).
        trajectory.append((board_to_tensor(board), board.turn == chess.WHITE))
        is_drl_turn = (board.turn == chess.WHITE) == drl_is_white
        agent = drl if is_drl_turn else opponent
        try:
            mv = agent.select_move(board)
        except Exception:
            break
        if mv not in board.legal_moves:
            break
        board.push(mv)

    drl.epsilon = 0.0
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is True:
            white_score = 1.0
        elif outcome.winner is False:
            white_score = -1.0
        else:
            white_score = 0.0
    else:
        # Truncated: fall back to heuristic eval as the pseudo-terminal value.
        white_score = math.tanh(heur_eval(board) / 800.0)

    return trajectory, white_score, board


# ---------------------------------------------------------------------- #
# Quick evaluation                                                       #
# ---------------------------------------------------------------------- #
def quick_winrate(drl: DRLAgent, opponent, n_games: int, max_plies: int = 120) -> float:
    """Score per game in [0, 1]: wins=1, draws=0.5, losses=0."""
    score = 0.0
    for g in range(n_games):
        white = drl if g % 2 == 0 else opponent
        black = opponent if g % 2 == 0 else drl
        board = chess.Board()
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            agent = white if board.turn == chess.WHITE else black
            try:
                mv = agent.select_move(board)
            except Exception:
                break
            if mv not in board.legal_moves:
                break
            board.push(mv)
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            score += 0.5
        elif (outcome.winner is True) == (g % 2 == 0):  # DRL was white in even games
            score += 1.0
        # else: DRL lost -> 0
    return score / n_games


# ---------------------------------------------------------------------- #
# Main training loop                                                     #
# ---------------------------------------------------------------------- #
def train(
    n_games: int = 400,
    max_plies: int = 120,
    batch_size: int = 256,
    updates_per_game: int = 8,
    lr: float = 1e-3,
    buffer_capacity: int = 50_000,
    epsilon_start: float = 0.30,
    epsilon_end: float = 0.05,
    alpha_start: float = 0.5,
    alpha_end: float = 0.1,
    eval_every: int = 20,
    eval_games: int = 6,
    save_dir: str = "chess_ext/results",
    model_path: str = "chess_ext/models/drl_value.pt",
    seed: int = 42,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drl = DRLAgent(device=device, seed=seed)
    drl.net.train()
    optimizer = torch.optim.Adam(drl.net.parameters(), lr=lr)
    replay = ChessReplay(capacity=buffer_capacity, seed=seed)

    opponent = StockfishAgent(skill=0, depth=1)
    rnd = RandomAgent(seed=seed)
    heur = HeuristicAgent(seed=seed)

    losses: List[float] = []
    eval_log: List[dict] = []
    best_score = -1.0
    t0 = time.time()

    try:
        for g in range(1, n_games + 1):
            # Linearly decay epsilon and shaping alpha
            frac = g / n_games
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
            alpha   = alpha_start   + (alpha_end   - alpha_start)   * frac

            drl_white = (g % 2 == 0)
            traj, white_score, _ = play_training_game(
                drl, opponent, drl_is_white=drl_white,
                max_plies=max_plies, epsilon=epsilon,
            )

            # Push (state, target_value) pairs into replay.
            for tensor, side_to_move_white in traj:
                outcome_pov = white_score if side_to_move_white else -white_score
                shaping = math.tanh(heur_eval_from_tensor_perspective(tensor, side_to_move_white) / 800.0)
                target = (1.0 - alpha) * outcome_pov + alpha * shaping
                replay.push(tensor, target)

            # Gradient updates
            if len(replay) >= batch_size:
                drl.net.train()
                for _ in range(updates_per_game):
                    states, targets = replay.sample(batch_size)
                    states = states.to(device)
                    targets = targets.to(device)
                    pred = drl.net(states)
                    loss = F.mse_loss(pred, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(drl.net.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                drl.net.eval()

            # Periodic evaluation against the baselines
            if g % eval_every == 0 or g == n_games:
                drl.net.eval()
                wr_rnd  = quick_winrate(drl, rnd,  eval_games, max_plies)
                wr_heur = quick_winrate(drl, heur, eval_games, max_plies)
                combined = 0.5 * (wr_rnd + wr_heur)
                eval_log.append({
                    "game": g,
                    "wr_random": wr_rnd,
                    "wr_heuristic": wr_heur,
                    "combined": combined,
                    "buffer": len(replay),
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "loss_recent": float(np.mean(losses[-200:])) if losses else float("nan"),
                })
                print(
                    f"[Game {g:4d}/{n_games}]  "
                    f"vs Random: {wr_rnd:.2%}  vs Heuristic: {wr_heur:.2%}  "
                    f"buf={len(replay):5d}  eps={epsilon:.2f}  "
                    f"loss={np.mean(losses[-200:]) if losses else float('nan'):.4f}",
                    flush=True,
                )
                if combined > best_score:
                    best_score = combined
                    drl.save(model_path)
                    print(f"  -> New best (combined={combined:.2%}); saved.", flush=True)

                # Periodic snapshot of training data
                json.dump(
                    {"eval_log": eval_log, "best_combined": best_score,
                     "n_games_done": g, "duration_s": time.time() - t0},
                    open(os.path.join(save_dir, "training_summary.json"), "w"),
                    indent=2,
                )
                np.save(os.path.join(save_dir, "losses.npy"), np.asarray(losses))

    finally:
        opponent.close()

    # Final save
    drl.save(os.path.splitext(model_path)[0] + "_final.pt")
    summary = {
        "eval_log": eval_log,
        "best_combined": best_score,
        "n_games": n_games,
        "duration_s": time.time() - t0,
        "n_updates": len(losses),
    }
    json.dump(summary, open(os.path.join(save_dir, "training_summary.json"), "w"), indent=2)
    np.save(os.path.join(save_dir, "losses.npy"), np.asarray(losses))
    print(
        f"\nDone in {(time.time()-t0)/60:.1f} min. "
        f"Best combined win-rate vs (Random+Heuristic)/2: {best_score:.2%}.",
        flush=True,
    )
    return summary


def heur_eval_from_tensor_perspective(tensor: np.ndarray, side_to_move_white: bool) -> float:
    """Reconstruct a python-chess board from the encoding to call heur_eval.

    A small inefficiency, but training is dominated by the actual games
    (Stockfish moves) so the cost is negligible.
    """
    # Reverse-engineer FEN from tensor planes - cheap because boards are tiny.
    PIECE_SYMS = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    rows = []
    for r in range(8):  # rank 8 -> rank 1 in tensor row order
        empty = 0
        row = ""
        for c in range(8):
            piece_idx = -1
            for p in range(12):
                if tensor[p, r, c] > 0.5:
                    piece_idx = p
                    break
            if piece_idx == -1:
                empty += 1
            else:
                if empty:
                    row += str(empty); empty = 0
                row += PIECE_SYMS[piece_idx]
        if empty:
            row += str(empty)
        rows.append(row)
    board_part = "/".join(rows)
    side = "w" if side_to_move_white else "b"
    fen = f"{board_part} {side} - - 0 1"
    try:
        board = chess.Board(fen)
        e = heur_eval(board)
        return e if side_to_move_white else -e
    except Exception:
        return 0.0


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Train chess value network")
    p.add_argument("--n-games", type=int, default=400)
    p.add_argument("--max-plies", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--updates-per-game", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=6)
    p.add_argument("--save-dir", default="chess_ext/results")
    p.add_argument("--model-path", default="chess_ext/models/drl_value.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(
        n_games=args.n_games, max_plies=args.max_plies,
        batch_size=args.batch_size, updates_per_game=args.updates_per_game,
        lr=args.lr, eval_every=args.eval_every, eval_games=args.eval_games,
        save_dir=args.save_dir, model_path=args.model_path, seed=args.seed,
    )


if __name__ == "__main__":
    main()
