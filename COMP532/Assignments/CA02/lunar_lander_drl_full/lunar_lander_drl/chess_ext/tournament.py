"""
Tournament harness.

Runs N games between two agents alternating colours and records the
result of each game (1-0 / 0-1 / 1/2-1/2) plus statistics. Used to
produce the win-rate table required by the brief and the ablation
between the random / heuristic / LLM-stub / DRL agents.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn

from .agents.base import BaseAgent


@dataclass
class GameRecord:
    white: str
    black: str
    result: str           # "1-0", "0-1", "1/2-1/2", "*"
    termination: str      # "checkmate" / "stalemate" / "insufficient material" / "max plies" / "fivefold" / etc.
    plies: int
    moves: List[str] = field(default_factory=list)


@dataclass
class TournamentResult:
    agent_a: str
    agent_b: str
    n_games: int
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    avg_plies: float = 0.0
    duration_s: float = 0.0
    games: List[GameRecord] = field(default_factory=list)

    @property
    def a_winrate(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / max(1, self.n_games)

    @property
    def b_winrate(self) -> float:
        return (self.b_wins + 0.5 * self.draws) / max(1, self.n_games)


def play_game(
    white: BaseAgent,
    black: BaseAgent,
    max_plies: int = 200,
    starting_fen: Optional[str] = None,
) -> GameRecord:
    """Play a single game. Returns a GameRecord."""
    board = chess.Board(starting_fen) if starting_fen else chess.Board()
    moves: List[str] = []

    white.reset()
    black.reset()

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        agent = white if board.turn == chess.WHITE else black
        try:
            mv = agent.select_move(board)
        except Exception as e:
            # Defensive: if an agent crashes, treat it as a loss for that side.
            losing = "1" if board.turn == chess.WHITE else "0"
            return GameRecord(
                white=white.name, black=black.name,
                result=("0-1" if losing == "1" else "1-0"),
                termination=f"agent crash: {type(e).__name__}",
                plies=ply, moves=moves,
            )
        if mv not in board.legal_moves:
            # An agent that proposes an illegal move forfeits the game.
            losing = "1" if board.turn == chess.WHITE else "0"
            return GameRecord(
                white=white.name, black=black.name,
                result=("0-1" if losing == "1" else "1-0"),
                termination="illegal move",
                plies=ply, moves=moves,
            )
        moves.append(board.san(mv))
        board.push(mv)

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = outcome.result()
        termination = outcome.termination.name.lower().replace("_", " ")
    else:
        result = "1/2-1/2"
        termination = "max plies"

    return GameRecord(
        white=white.name, black=black.name,
        result=result, termination=termination,
        plies=board.ply(), moves=moves,
    )


def play_match(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    n_games: int = 20,
    max_plies: int = 200,
    alternate_colours: bool = True,
    verbose: bool = True,
    pgn_dir: Optional[str] = None,
) -> TournamentResult:
    """Play a match between A and B, alternating colours by default.

    The convention: a_wins counts wins for ``agent_a`` regardless of
    which colour they played in a given game.
    """
    res = TournamentResult(
        agent_a=agent_a.name, agent_b=agent_b.name, n_games=n_games,
    )
    t0 = time.time()
    plies_total = 0

    for g in range(n_games):
        a_white = (g % 2 == 0) if alternate_colours else True
        white, black = (agent_a, agent_b) if a_white else (agent_b, agent_a)
        rec = play_game(white, black, max_plies=max_plies)
        res.games.append(rec)
        plies_total += rec.plies

        if rec.result == "1-0":
            if a_white: res.a_wins += 1
            else:        res.b_wins += 1
        elif rec.result == "0-1":
            if a_white: res.b_wins += 1
            else:        res.a_wins += 1
        else:
            res.draws += 1

        if verbose:
            print(
                f"[{g+1:3d}/{n_games}] {white.name} vs {black.name}: "
                f"{rec.result} ({rec.termination}, {rec.plies} plies)  "
                f"score: A={res.a_wins} B={res.b_wins} D={res.draws}",
                flush=True,
            )

        if pgn_dir is not None:
            os.makedirs(pgn_dir, exist_ok=True)
            game = chess.pgn.Game()
            game.headers["Event"]  = "COMP532 chess extension"
            game.headers["White"]  = white.name
            game.headers["Black"]  = black.name
            game.headers["Result"] = rec.result
            node = game
            board = chess.Board()
            for san in rec.moves:
                mv = board.parse_san(san)
                node = node.add_variation(mv)
                board.push(mv)
            with open(os.path.join(pgn_dir, f"game_{g+1:03d}.pgn"), "w") as f:
                f.write(str(game))

    res.duration_s = time.time() - t0
    res.avg_plies = plies_total / max(1, n_games)
    if verbose:
        print(
            f"\nFinal: {agent_a.name} {res.a_wins}-{res.b_wins}-{res.draws} {agent_b.name}  "
            f"(win-rate A={res.a_winrate:.2%})  duration={res.duration_s:.1f}s",
            flush=True,
        )
    return res


def save_results(result: TournamentResult, path: str) -> None:
    """Write a JSON summary (without per-game move lists for size)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary = {
        "agent_a": result.agent_a,
        "agent_b": result.agent_b,
        "n_games": result.n_games,
        "a_wins": result.a_wins,
        "b_wins": result.b_wins,
        "draws": result.draws,
        "a_winrate": result.a_winrate,
        "b_winrate": result.b_winrate,
        "avg_plies": result.avg_plies,
        "duration_s": result.duration_s,
        "games": [
            {k: v for k, v in asdict(g).items() if k != "moves"}
            for g in result.games
        ],
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
