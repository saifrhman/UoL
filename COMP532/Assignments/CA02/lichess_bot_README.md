# lichess-bot files for COMP532 chess extension

These two files plug into your existing
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) clone
to deploy any of our chess agents (Heuristic, DRL, or LLM via
OpenRouter / Anthropic) as a real bot on lichess.org.

## Files

- **`homemade.py`** — a `MinimalEngine` subclass `Comp532Bot`. Replaces
  (or is appended to) the `homemade.py` in your `lichess-bot/` clone.
- **`config.yml`** — minimal lichess-bot config that registers
  `Comp532Bot` as the homemade engine. Replaces the `config.yml` in
  your `lichess-bot/` clone.

Neither file contains any secret. All credentials come from
environment variables.

## Quickstart

From inside your `lichess-bot/` clone:

```bash
# 0. Copy these two files in
cp /path/to/lichess_bot_files/homemade.py ./homemade.py
cp /path/to/lichess_bot_files/config.yml  ./config.yml

# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install openai numpy torch python-chess

# 2. Set credentials in the shell (NOT in any committed file)
export LICHESS_BOT_TOKEN='lip_...'                # bot:play OAuth token
export OPENROUTER_API_KEY='sk-or-v1-...'          # only for COMP532_AGENT=llm
export COMP532_AGENT='heuristic'                  # or 'drl' or 'llm'
export COMP532_DRL_MODEL='/path/to/chess_drl.pt'  # only for COMP532_AGENT=drl

# 3. Verify the lichess account is registered as a BOT
curl -H "Authorization: Bearer $LICHESS_BOT_TOKEN" \
     https://lichess.org/api/account/me | python -m json.tool
# Expect to see  "title": "BOT"  in the response.

# 4. Start the bot
python lichess-bot.py
```

## Picking the agent

| `COMP532_AGENT` | What it does | Per-game cost | Strength |
|---|---|---|---|
| `heuristic` (default) | Material + PSTs + 1-ply lookahead | £0 | ~1000 Elo (club beginner) |
| `drl`                 | Trained CNN value network        | £0 | ~700 Elo without good weights, higher with training |
| `llm`                 | Calls Opus 4.6 over OpenRouter    | ~£0.06 / game | ~1500–1800 Elo (per LLM-Chess literature) |

For sustained operation use `heuristic`. Use `llm` for a small number
of showcase games to stay within budget — Opus 4.6 costs $5/MTok input
and $25/MTok output (about £6 per 100 games at current rates).

## Why not chess.com?

Chess.com offers no public bot/play API. Programmatic play there only
works through (i) authenticated browser automation against the chess.com
web UI, or (ii) reverse-engineering their private REST endpoints —
both violate the chess.com Terms of Service. Lichess is the supported
equivalent: bots there are clearly labelled (`title: "BOT"`), get a
real Glicko-2 rating, and are first-class citizens of the platform.
