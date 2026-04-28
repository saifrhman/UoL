# COMP532 — Assignment 2: Deep RL on LunarLander + Chess Extension

A from-scratch **Dueling Double DQN (D3QN)** agent for *LunarLander*
(Problem 1) plus a **chess extension** comparing four agents (Random,
Heuristic, LLM, DRL) against a Stockfish reference in a complete
round-robin tournament.

## Headline results

| Metric | Value |
|---|---|
| **Problem 1** | |
| Algorithm | Dueling Double DQN (D3QN) |
| Solved at (mean₁₀₀ ≥ 200) | **Episode 462** |
| Best 100-ep mean | **252.87** |
| Greedy eval (30 unseen seeds) | 228.80 ± 69.41 (70 % above 200) |
| Training time | 19.6 min CPU |
| **Problem 2** | Conceptual essay on exploration vs exploitation in deep RL |
| **Chess extension** | |
| DRL training | 300 self-play games, 9.8 min CPU |
| DRL best combined win-rate vs (Random, Heuristic) | **75 %** |
| Tournament: Stockfish-0 | 98.8 % (39-0-1 over 40 games) |
| Tournament: LLM-stub | 55.0 % (15-11-14) |
| Tournament: Heuristic | 51.2 % (13-12-15) |
| Tournament: DRL | 42.5 % (12-18-10) |
| Tournament: Random | 2.5 % (0-38-2) |

## Files

```
lunar_lander_drl/
├── report.tex                    # LaTeX report (Problems 1+2 + chess extension)
├── report.pdf                    # Compiled report (15 pages)
├── COMP532_CA2_Report.docx       # Earlier DOCX version (no chess section)
│
├── src/                          # Problem 1: LunarLander DDDQN
│   ├── network.py                #   DuelingQNetwork
│   ├── replay_buffer.py          #   ReplayBuffer (NumPy ring buffer)
│   ├── agent.py                  #   DuelingDoubleDQNAgent + AgentConfig
│   ├── train.py                  #   training loop
│   ├── evaluate.py               #   greedy roll-outs + GIF
│   └── plotting.py               #   reward/loss figures
├── main.py                       # CLI: train -> plot -> eval -> GIF
├── grab_frame.py                 # extract one frame from a roll-out
├── models/d3qn_lunarlander.pt    # best checkpoint
├── plots/{rewards,loss,demo_frame}.png
├── results/{training_summary,eval_summary}.json + .npy logs
├── videos/agent_demo.gif
│
├── chess_ext/                    # Optional chess extension
│   ├── agents/
│   │   ├── base.py               #   BaseAgent interface
│   │   ├── heuristic.py          #   RandomAgent + HeuristicAgent (PSTs)
│   │   ├── llm.py                #   LLMAgent (real OpenAI/Anthropic + offline stub)
│   │   ├── drl.py                #   DRLAgent (CNN value net + 1-ply lookahead)
│   │   ├── stockfish_agent.py    #   Wrapper around Stockfish 16 UCI binary
│   │   └── encoding.py           #   18-channel board tensor
│   ├── tournament.py             #   play_match harness
│   ├── train_drl.py              #   value-network training
│   ├── run_tournament.py         #   round-robin ablation
│   ├── plotting.py               #   training/loss/matrix plots
│   ├── uci.py                    #   UCI shim (deploy via lichess-bot)
│   ├── PLAYING_ONLINE.md         #   How to play on lichess (NOT chess.com)
│   ├── models/drl_value.pt       #   trained value network
│   ├── plots/{training,loss,tournament_matrix,sample_game_final}.png
│   ├── results/{training_summary,tournament,winrate_table}.json/csv
│   └── games/                    #   PGNs of every tournament game
│
└── README.md (this file)
```

## Reproducing every number

Three commands. CPU only; total ~30 minutes.

```bash
# Set up
pip install -r requirements.txt
sudo apt-get install -y stockfish      # for chess training & reference

# 1. Problem 1 (LunarLander) - 19.6 min, produces rewards.png, loss.png, GIF
python main.py --episodes 1000 --seed 42

# 2. Chess extension: train DRL value network - 9.8 min
python -m chess_ext.train_drl --n-games 300 --seed 42

# 3. Chess ablation: full round-robin - ~1 min
python -m chess_ext.run_tournament --n-games 10 --seed 123
```

Re-build the LaTeX report:

```bash
pdflatex report.tex && pdflatex report.tex     # twice for cross-refs
```

## Why python-chess (not OpenAI Gym) for chess?

Gymnasium has no canonical chess environment. The PyPI `gym-chess`
package is unmaintained, uses a clunky one-hot action space, and is
not used in the published chess-RL literature. AlphaZero, ChessGPT,
and the LLM-Chess benchmark all wrap **python-chess** directly. Same
choice here.

## Why lichess (not chess.com)?

`chess.com` has no public bot/play API. The two ways to make a program
play there both violate the Terms of Service (browser automation; or
reverse-engineering their private REST endpoints). The supported route
is **lichess.org**, which has a first-class Bot API
(OAuth2 `bot:play` scope). All our agents speak UCI via
`python -m chess_ext.uci`, so any of them can be deployed on lichess
in three configuration lines using the official `lichess-bot` bridge.
Full instructions in `chess_ext/PLAYING_ONLINE.md`.
