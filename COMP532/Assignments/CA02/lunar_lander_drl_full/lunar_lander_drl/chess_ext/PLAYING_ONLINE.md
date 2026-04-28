# Playing online with the chess agent

## Why lichess and not chess.com?

The brief asks whether the agent can play chess.com. The honest engineering
answer is: **not via a supported API**. chess.com does not expose a public
"play games as a bot" interface. The two ways to make a program play
chess.com — both **against chess.com's Terms of Service** — are:

1. Browser automation against the chess.com web UI (Selenium/Playwright
   logged into a real account). This is account-bannable and would
   require Cloudflare/anti-bot bypass.
2. Reverse-engineering the private chess.com API used by their own
   front-end. Same ToS problem, plus rate limits the moment they notice.

The intended-by-design route is **lichess.org**, which provides a first-
class **Bot API** (OAuth2 scope `bot:play`) explicitly for projects like
this. Bots show up clearly as "BOT" accounts, are rate-limited fairly,
and the bridge software (`lichess-bot`) is officially blessed by the
lichess team. Engines played on lichess get a real Glicko-2 rating, so
performance is measurable.

If you want chess.com-style play — i.e. play against humans through a
mainstream chess website — **lichess is the legal equivalent**.

## Step-by-step: deploy our agent to lichess

These steps assume you already have a Python environment set up.

### 1. Create a lichess BOT account

1. Sign up at https://lichess.org/signup (do NOT play any games on the
   account first; lichess only converts brand-new accounts to BOTs).
2. Generate an OAuth2 token:
   `Settings -> API access tokens -> New personal API access token`,
   tick `Play games with the bot API` (scope `bot:play`).
3. Convert the account: with the token in `$TOKEN`,
   ```
   curl -X POST -H "Authorization: Bearer $TOKEN" \
        https://lichess.org/api/bot/account/upgrade
   ```

### 2. Install the lichess-bot bridge

```
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip install -r requirements.txt
```

### 3. Configure it to use our UCI shim

Edit `config.yml`:

```yaml
token: "YOUR_LICHESS_BOT_TOKEN"

engine:
  dir: "../lunar_lander_drl"
  name: "uci_runner.sh"          # see below
  protocol: "uci"
  variants:
    - standard
  uci_options:
    Threads: 1
    Hash: 64
```

Create `lunar_lander_drl/uci_runner.sh` (and `chmod +x` it):

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
exec python -m chess_ext.uci \
    --agent drl \
    --model chess_ext/models/drl_value.pt \
    --name comp532-d3qn-bot
```

For the LLM agent against a real model, swap `--agent llm --backend openai
--model gpt-4o-mini` and export `OPENAI_API_KEY`.

### 4. Run

```
cd lichess-bot
python lichess-bot.py
```

Now anyone can challenge your bot from its lichess profile page; the bot
will accept according to `config.yml` and play with the agent of your
choice.

## Local play (no internet required)

Any UCI-compatible GUI works. With Cute Chess installed:

```
cutechess-gui --engine cmd="python -m chess_ext.uci --agent drl --model chess_ext/models/drl_value.pt" \
              --engine cmd="python -m chess_ext.uci --agent heuristic" \
              --rounds 10
```

The same `chess_ext.uci` entry point speaks fluent UCI to any
contemporary chess GUI.
