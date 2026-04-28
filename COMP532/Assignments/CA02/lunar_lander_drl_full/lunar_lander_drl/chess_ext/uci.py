"""
UCI engine shim.

Wraps any of our BaseAgent subclasses as a UCI-protocol-speaking
process. Once running, this can be plugged into:

    - any UCI-compatible chess GUI (Cute Chess, Arena, BanksiaGUI),
    - the lichess-bot bridge (https://github.com/lichess-bot-devs/lichess-bot)
      to play live games on lichess.org as a registered BOT account.

Why not chess.com?
------------------
chess.com does NOT provide a public bot/play API. Building a bot for
chess.com would require authenticated browser automation against
chess.com's HTML, which violates their Terms of Service. The
intentional, legal route to "let the agent play online" is via lichess
(BOT API, OAuth2 'bot:play' scope) using lichess-bot, which speaks
exactly the UCI protocol implemented here.

Usage
-----
    python -m chess_ext.uci --agent heuristic
    python -m chess_ext.uci --agent drl --model chess_ext/models/drl_value.pt
    python -m chess_ext.uci --agent llm  --backend openai --model gpt-4o-mini

then point a UCI GUI at this command, or hand it to lichess-bot.
"""

from __future__ import annotations

import argparse
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_ext.agents import (  # noqa: E402
    RandomAgent, HeuristicAgent, LLMAgent, DRLAgent,
)


def make_agent(args) -> object:
    if args.agent == "random":
        return RandomAgent(seed=args.seed)
    if args.agent == "heuristic":
        return HeuristicAgent(seed=args.seed)
    if args.agent == "drl":
        return DRLAgent(model_path=args.model, seed=args.seed)
    if args.agent == "llm":
        return LLMAgent(
            backend=args.backend, model=args.model or "gpt-4o-mini",
            api_key=args.api_key, seed=args.seed,
        )
    raise SystemExit(f"Unknown agent: {args.agent}")


def uci_loop(agent, name: str = "comp532-bot") -> None:
    """Minimal UCI loop. Implements just enough commands for lichess-bot."""
    board = chess.Board()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()

        if cmd == "uci":
            print(f"id name {name}")
            print("id author COMP532-group")
            print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            board = chess.Board()
            agent.reset()
        elif cmd.startswith("position"):
            board = _parse_position(cmd)
        elif cmd.startswith("go"):
            move = agent.select_move(board)
            print(f"bestmove {move.uci()}", flush=True)
        elif cmd in ("quit", "stop"):
            break


def _parse_position(cmd: str) -> chess.Board:
    """Parse a UCI 'position ...' command into a chess.Board."""
    parts = cmd.split(" ", 1)[1].strip()
    if parts.startswith("startpos"):
        board = chess.Board()
        rest = parts[len("startpos"):].strip()
    elif parts.startswith("fen"):
        # 'fen <FEN>' optionally followed by 'moves ...'
        after_fen = parts[len("fen"):].strip()
        if " moves " in after_fen:
            fen, _, rest = after_fen.partition(" moves ")
            rest = "moves " + rest
        else:
            fen = after_fen
            rest = ""
        board = chess.Board(fen.strip())
    else:
        return chess.Board()

    if rest.startswith("moves"):
        for u in rest[len("moves"):].split():
            try:
                board.push(chess.Move.from_uci(u))
            except (ValueError, chess.IllegalMoveError):
                break
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["random", "heuristic", "drl", "llm"], default="heuristic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default="stub")  # for LLM
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="comp532-bot")
    args = ap.parse_args()
    agent = make_agent(args)
    try:
        uci_loop(agent, name=args.name)
    finally:
        try: agent.close()
        except Exception: pass


if __name__ == "__main__":
    main()
