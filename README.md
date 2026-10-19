# Value Betting Engine

> A multi-sport value-betting toolkit. It models game outcomes from historical
> data, compares its calibrated probabilities against live sportsbook odds
> (including DraftKings) to find positive-expected-value bets, sizes stakes with
> a correlation-aware fractional Kelly optimiser, scans for cross-book
> arbitrage, and tracks your results with proper risk-adjusted metrics — all
> behind an accessible web interface.

Currently ships with four markets — **NBA**, **WNBA**, **MLB**, and the
**English Premier League** — and is built so a new sport is a single ~100-line
plugin.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Training the models](#training-the-models)
5. [The web app](#the-web-app)
6. [Adding a new sport](#adding-a-new-sport)
7. [Risk analysis](#risk-analysis)
8. [Parlays vs. straight bets](#parlays-vs-straight-bets)
9. [Prediction-market whale tracker](#prediction-market-whale-tracker)
10. [Polymarket copy trader](#polymarket-copy-trader)
11. [Configuration](#configuration)
12. [Project structure](#project-structure)
13. [Disclaimer](#disclaimer)

---

## How it works

Most bettors size each wager in isolation and trust the sportsbook's implied
odds. This engine does neither:

- **Finds edges.** For every game it builds a feature vector (ELO, rolling
  form, exponentially-weighted attack/defense ratings, matchup ratios), predicts
  a calibrated win probability with a gradient-boosted model, and compares it to
  the best price available across books. When the model's probability beats the
  implied probability, that's positive expected value.
- **Sizes stakes sensibly.** Positive-edge bets are fed to a Modern-Portfolio-
  Theory / fractional-Kelly optimiser that accounts for correlation between
  wagers and caps total exposure according to your personal risk profile.
- **Finds free money.** A cross-book arbitrage scanner flags any game where the
  best prices across sportsbooks let you bet every outcome for a guaranteed
  profit.
- **Keeps you honest.** Every bet you record is tracked, and your history is
  scored with Sharpe, Sortino, Calmar, and maximum-drawdown metrics so you can
  see whether your returns are actually *stable* or just lucky.

### What this is not

This is not a guaranteed-profit machine. Betting markets are efficient and the
edges are thin. The value here is disciplined process: better-calibrated
probabilities, systematic stake sizing, correlation-aware diversification, and
honest performance tracking — the things that separate steady growth from
variance.

---

## Architecture

```
                          ┌────────────────────────┐
   historical results ───▶│  sports/ plugins       │  one class per sport;
   (MLB API, nba_api,     │  (fetch_history)       │  returns date/teams/scores
    football-data.co.uk)  └───────────┬────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │  core/generic_features │  ELO + form + attack/def,
                          │  (sport-agnostic)      │  strictly no look-ahead
                          └───────────┬────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │  predictive_model/     │  LightGBM (+ isotonic
                          │  ModelTrainer          │  calibration for 2-way)
                          └───────────┬────────────┘
                                      ▼
   live odds ───────────▶ ┌────────────────────────┐
   (The Odds API:         │  core/scoring_engine   │  edge, Kelly, arbitrage
    DraftKings, FanDuel…) └───────────┬────────────┘
                                      ▼
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                        ▼
┌────────────────────┐                              ┌────────────────────────┐
│ portfolio_manager/ │  correlation-aware           │  webapp/ (Flask)       │
│ optimizer + arb    │  fractional-Kelly SLSQP      │  accessible UI + SQLite│
└────────────────────┘                              │  bet ledger + metrics  │
                                                    └────────────────────────┘
```

Everything below the plugin layer is sport-agnostic. A plugin only has to answer
one question — *"give me every historical game as date / home / away / scores"* —
and declare a handful of tuning constants; the shared code does the rest.

---

## Quick start

```bash
# 1. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2. Configure your odds API key
cp .env.example .env
#   then edit .env and set THE_ODDS_API_KEY (free tier at the-odds-api.com)

# 3. Train the models you want (see next section)
python scripts/train_sport.py --sport mlb epl

# 4. Launch the web app
python -m webapp.app
#   open http://127.0.0.1:5000
```

Run the tests (no keys or network needed) and the synthetic-data benchmark:

```bash
pytest tests/ --cov=. --cov-fail-under=70
python scripts/benchmark_model.py
```

---

## Training the models

Each sport trains independently from public data — no paid feeds required.

```bash
python scripts/train_sport.py --sport mlb      # MLB Stats API
python scripts/train_sport.py --sport epl      # football-data.co.uk
python scripts/train_sport.py --sport nba      # stats.nba.com
python scripts/train_sport.py --sport wnba     # stats.nba.com
python scripts/train_sport.py --sport all      # everything
```

For each sport the trainer:

1. pulls full history via the plugin,
2. builds the generic feature table (ELO, form, attack/defense),
3. trains LightGBM — a calibrated binary model for 2-way sports (NBA/WNBA/MLB),
   a 3-class model for draw sports (EPL) — on a temporal 85/15 split,
4. reports held-out log-loss, accuracy, and feature importances,
5. saves `<sport>_model.pkl` and `<sport>_state.pkl` into `artefacts/`.

The `state` file holds each team's current ELO and latest rolling stats, so live
scoring never has to re-walk history — it just needs today's odds.

Model artefacts and training CSVs are git-ignored; regenerate them locally.

---

## The web app

`python -m webapp.app` starts an accessible Flask interface:

| Page | What it does |
|------|--------------|
| **Dashboard** | Market status, your risk profile, and a snapshot of performance. |
| **Slates** | Per-sport board of today's games: model probability vs. implied probability, per-outcome edge, the recommended Kelly allocation, and any arbitrage. One click records a bet to your ledger. |
| **Parlay Analyzer** | Enter legs and get an EV/Kelly verdict — and an honest comparison against betting them straight. |
| **History & Risk** | Your full bet ledger plus Sharpe, Sortino, Calmar, max-drawdown, ROI, and a cumulative-P&L chart. |
| **Settings** | Your bankroll, betting budget, and volatility tolerance — which drive every stake the engine recommends. |

Accessibility: semantic landmarks, skip-to-content link, labelled form controls,
keyboard-navigable menus, visible focus outlines, `prefers-color-scheme` light/
dark theming, `prefers-reduced-motion` support, and a text-described,
JavaScript-free P&L chart.

Your bet ledger lives in a local SQLite file (`webapp/data/app.db`) that is
git-ignored — your personal betting record never leaves your machine.

---

## Adding a new sport

Two steps:

1. Create `sports/<key>.py` with a `SportPlugin` subclass exposing a module-level
   `PLUGIN`. Implement `fetch_history()` (return a DataFrame with `date`,
   `home_team`, `away_team`, `home_score`, `away_score`) and fill in a
   `SportConfig` (odds key, whether the market has draws, ELO tuning, typical
   score). Override `normalize_name()` if the odds provider and your data source
   spell team names differently.
2. Add `"<key>"` to `_PLUGIN_MODULES` in `sports/__init__.py`.

That's it — training, scoring, arbitrage, the parlay tool, and every web page
pick the sport up automatically.

---

## Risk analysis

The engine optimises for *steady* growth, so the History page reports the
metrics that actually capture stability, not just hit rate:

- **Sharpe ratio** — return per unit of total volatility.
- **Sortino ratio** — return per unit of *downside* volatility (upside swings
  don't count against you).
- **Maximum drawdown** — the largest peak-to-trough fall of your bankroll; the
  single most important survival metric. Above ~25% is where things get
  dangerous.
- **Calmar ratio** — total return divided by max drawdown (worst-case
  efficiency).

These roll up into a plain-language **stability grade** so you don't have to
interpret four ratios yourself.

Stake sizing is governed by your risk profile via `risk_bridge/`: your
volatility tolerance (1–10) maps to hard caps on total exposure, single-bet
size, and a session stop-loss.

---

## Parlays vs. straight bets

Parlays multiply your edge *and* the sportsbook's vig across every leg — a
two-leg parlay typically carries a ~20% hold versus ~4–5% on a straight bet, and
it compounds with each leg. The Parlay Analyzer therefore never just prices a
parlay: it computes the parlay's expected value, compares it to betting the same
legs straight, flags correlated same-game legs (where the independence math
breaks down), and gives a blunt verdict — **parlay**, **bet straight**, or
**pass**. For steady growth the answer is usually "bet straight."

---

## Prediction-market whale tracker

`whale_tracker/` watches **Polymarket** and **Kalshi** for unusually large
trades and alerts you as they happen. It uses only public, keyless endpoints:
Polymarket's Data API (`data-api.polymarket.com/trades`, `/v1/leaderboard`) and
Kalshi's market-data API (`api.elections.kalshi.com/trade-api/v2/markets/trades`).

```bash
python -m whale_tracker test-alert                 # check your alert channels
python -m whale_tracker import-leaderboard --top 20  # follow Polymarket's top traders
python -m whale_tracker follow 0xabc… --label "sharp"
python -m whale_tracker watch                      # poll every 30s, alert on whales
python -m whale_tracker recent                     # what fired lately
```

**What counts as a whale**

| Alert | Platform | Rule |
|---|---|---|
| Whale trade | both | one fill ≥ `WHALE_MIN_USD` (default $10k) |
| Whale accumulating | Polymarket | one wallet's smaller fills on the same market and side sum to the threshold within an hour (catches split orders) |
| Large order (sweep) | Kalshi | fills on the same market and side within 10s sum to the threshold, which most likely means one order walked the book |
| Followed wallet traded | Polymarket | a wallet you follow trades ≥ `WHALE_WATCH_MIN_USD` (default $1k) |

Each alert says why it fired and adds context: longshot buys (≤15¢, with the
payout multiple), heavy-favourite buys (≥90¢), and brand-new Polymarket
wallets (fewer than 5 trades on record), a common sign of someone trading on
fresh information.

**Polymarket vs. Kalshi.** Polymarket settles on-chain, so every trade carries
the trader's wallet and whales can be followed individually. Kalshi's public
feed is anonymous: it shows size, price and time but not who traded, so Kalshi
alerts are about size only and sweeps are an inference.

**Alert channels.** The terminal always gets alerts. Set any of
`WHALE_DISCORD_WEBHOOK_URL`, `WHALE_SLACK_WEBHOOK_URL`, `WHALE_NTFY_TOPIC`
(free phone push via the [ntfy](https://ntfy.sh) app) or
`WHALE_TELEGRAM_BOT_TOKEN` + `WHALE_TELEGRAM_CHAT_ID` to fan out further. See
`.env.example` for every setting.

Seen trades, alerts, followed wallets and poll cursors live in SQLite
(`data/whales.db`), so restarts never re-send an alert. A cold start only
looks back 15 minutes (`WHALE_LOOKBACK_SECONDS`).

---

## Polymarket copy trader

`copy_trader/` is a separate program from the whale tracker. It finds
Polymarket traders with **high win rates and large profits**, watches what
they hold, and when they buy or sell it tells you the exact order to place to
match them, sized to your bankroll. It only gives signals: it never touches
your account or places orders.

```bash
python -m copy_trader discover --candidates 50        # score the all-time leaderboard
python -m copy_trader discover --follow 10            # ...and follow the 10 best
python -m copy_trader score 0xabc…                    # one wallet's scorecard
python -m copy_trader --bankroll 2000 watch           # alert on every move they make
python -m copy_trader --bankroll 2000 open            # their current positions still near entry price
python -m copy_trader signals                         # recent signals
```

**Picking traders.** Each candidate's most recent 500 resolved positions are
scored (newest first, not biggest wins first, so the win rate isn't
inflated). By default a trader qualifies with at least 30 resolved positions,
a win rate of 60% or more, $50k+ profit, activity in the last 30 days, and an
**edge** of at least 5 points. Edge is the win rate minus the average price
paid. Someone who only buys 95¢ favourites wins 95% of the time and makes
almost nothing; edge filters them out. Ranking uses the 95%-confidence lower
bound of the win rate, so 9-for-10 doesn't beat 90-for-100. Traders are
re-scored daily and stop being copied if they fall below the bar (wallets you
`follow` by hand are still copied, with a warning).

**What a signal looks like**

```
COPY ▸ BUY 232.55 Yes @ limit 43¢ (≈$100.00)
   Will the Fed cut rates in December?
   Market now 42¢ · sharp opened $38,400 at ~41¢ (6.0% of their portfolio)
   Trader: sharp: 68% win rate over 214 resolved, +11 pts vs prices paid, $1,200,000 profit · https://polymarket.com/profile/0x…
   capped at 5% of your bankroll (they put 6% of their portfolio in)
   1 other followed trader also holds Yes
   https://polymarket.com/event/…
```

- **OPEN / ADD**: buy. You put the same share of your bankroll in as they put
  of their portfolio (`COPY_SCALE` scales that), capped at 5% per position. The
  limit is the market price plus 1¢.
- **TRIM / EXIT**: sell the same fraction they sold. Shares are worked out
  from the buys it told you to make.
- **WAIT**: they bought, but the price has run more than 5¢ past their entry,
  or it's above 97¢, or your copy would be under $1. It tells you the most to
  pay.
- Markets that resolve are payouts, not sells, so they never trigger an exit.
  A failed API call never reads as a sell-off either.

Signals go to the terminal and optionally Discord, Slack, ntfy (phone push) or
Telegram via the `COPY_*` settings in `.env.example`.

**Caveats.** You always enter after the trader and at a worse price; only
their positions of $500+ are tracked; portfolio weight uses their open
positions, not their uninvested cash; past win rates don't guarantee future
ones. Check that Polymarket is available where you live.

---

## Configuration

All configuration is via `.env` (copied from `.env.example`):

| Variable | Purpose |
|----------|---------|
| `THE_ODDS_API_KEY` | The Odds API key (free tier covers all four sports). |
| `ODDS_PROVIDER` | `theodds` (default) or `oddsjam`. |
| `NBA_API_TIMEOUT` | Timeout for stats.nba.com calls (NBA/WNBA). |
| `MODEL_DIR` | Where model artefacts are written (default `artefacts`). |
| `BETS_DB_PATH` | SQLite ledger path (default `webapp/data/app.db`). |
| `MAX_PORTFOLIO_KELLY_FRACTION` / `MAX_SINGLE_BET_FRACTION` | Hard exposure caps. |

**Never commit `.env`** — it holds your real API key and is git-ignored.

---

## Project structure

```
config/                 settings + user risk profile
core/
  generic_features.py   sport-agnostic ELO / form / attack-defense engine
  scoring_engine.py     live scoring: edge, Kelly, arbitrage (shared path)
  parlay.py             parlay EV / Kelly analysis
sports/
  base.py               SportPlugin + SportConfig contract
  __init__.py           plugin registry
  mlb.py  epl.py  nba.py  wnba.py   the four sport plugins
predictive_model/       ModelTrainer (LightGBM) + isotonic calibrator
portfolio_manager/      covariance estimator, SLSQP optimiser, arbitrage
risk_bridge/            maps a user's finances to portfolio constraints
webapp/
  app.py                Flask routes (app factory)
  store.py              SQLite: risk profile + bet ledger
  risk_metrics.py       Sharpe / Sortino / Calmar / drawdown
  templates/  static/   accessible UI
scripts/
  train_sport.py        unified per-sport trainer
data_pipeline/          odds client + raw stats clients
whale_tracker/
  sources/              Polymarket + Kalshi public-API clients
  detector.py           whale rules: large trade, accumulation, sweep, followed wallet
  watcher.py  store.py  poll loop + SQLite (dedupe, alerts, watchlist)
  notify.py  cli.py     terminal / Discord / Slack / ntfy / Telegram; CLI
copy_trader/
  polymarket.py         leaderboard, open + closed positions
  scoring.py            win rate, edge vs price paid, Wilson bound, qualification
  signals.py            position diffs → OPEN/ADD/TRIM/EXIT/WAIT orders, sizing
  watcher.py  store.py  poll loop + SQLite (snapshots, copied book, signal log)
  notify.py  cli.py     alert channels; CLI
tests/                  pytest suite
```

---

## Disclaimer

This software is for research and educational purposes. Model probabilities are
estimates, not guarantees, and no betting strategy can eliminate the house edge
or the risk of loss. Only wager money you can afford to lose, and make sure
sports betting is legal in your jurisdiction before using live odds to place
real bets.
