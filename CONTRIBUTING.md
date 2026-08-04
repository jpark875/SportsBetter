# Contributing to NBA Bet Portfolio Engine

Thank you for your interest in contributing. This document outlines the standards and workflow for submitting changes.

## Development Setup

```bash
git clone https://github.com/jpark875/SportsBetter.git
cd SportsBetter
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # fill in your API keys
```

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/description` | `feat/xgboost-backend` |
| Bug fix | `fix/description` | `fix/covariance-negative-definite` |
| Docs | `docs/description` | `docs/prop-betting-guide` |
| Refactor | `refactor/description` | `refactor/odds-client-retry` |

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(portfolio): add CVaR constraint to SLSQP objective
fix(nba_client): handle 429 rate-limit with exponential back-off
docs(readme): add real training data section
test(calibrator): add ECE regression test
```

## Code Standards

- Python 3.10+ only. No walrus-operator workarounds for older versions.
- PEP 8. Line length 100.
- Every public function must have a Google-style docstring with `Parameters` and `Returns` sections.
- Type hints on all function signatures (`from __future__ import annotations` at the top of every file).
- No bare `except:` — catch specific exceptions or use `except Exception as exc:` with a logged warning.

## Testing Requirements

All PRs must:

1. Pass the full test suite: `pytest tests/ -v`
2. Keep coverage above 70 %: `pytest tests/ --cov=. --cov-fail-under=70`
3. Pass the ML benchmark: `python scripts/benchmark_model.py`

The benchmark enforces minimum thresholds (ROC-AUC ≥ 0.55, ECE ≤ 0.05). A PR that degrades these metrics will not be merged.

## Module Boundaries

Each package has a strict ownership:

| Package | Owns | Must NOT |
|---|---|---|
| `data_pipeline` | API calls, raw JSON → DataFrame | Contain any ML logic |
| `predictive_model` | Feature matrix, model, calibration | Call external APIs directly |
| `portfolio_manager` | Covariance, allocation | Know about user finances |
| `risk_bridge` | User profile → constraints | Perform model inference |
| `config` | Constants, env vars | Perform I/O at import time |

Cross-boundary dependencies flow **downward only**: `risk_bridge` → `portfolio_manager` → `predictive_model` → `data_pipeline` → `config`.

## Adding a New Odds Provider

1. Create a `_YourProviderAdapter` class in [data_pipeline/odds_client.py](data_pipeline/odds_client.py) following the `_TheOddsAdapter` interface.
2. Implement `fetch_game_odds()` and `fetch_player_props()` returning the canonical schema.
3. Wire it into `OddsClient.__init__` with a new provider key string.
4. Add the provider name to `.env.example` and the README provider table.

## Pull Request Checklist

- [ ] Branch is up-to-date with `main`
- [ ] All tests pass locally
- [ ] New code has docstrings and type hints
- [ ] `.env.example` updated if new env vars added
- [ ] `requirements.txt` updated if new dependencies added
- [ ] `README.md` updated if user-facing behavior changed
- [ ] No API keys, `.env` files, or model `.pkl` artefacts committed
