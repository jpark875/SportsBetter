"""
core/scoring_engine.py

The single scoring path shared by the web app and the CLI.  Given a
registered sport key it:

    loads that sport's trained model + saved state (current ELO, latest
    team stats, calibrator),
    pulls the live slate from the odds provider,
    builds one generic feature row per game,
    predicts calibrated win probabilities (2-way or 3-way),
    joins them to the best available price per outcome,
    computes edge, full-Kelly, and a correlation-aware allocation.

Nothing here is sport-specific: every branch that differs between, say,
MLB and the Premier League is driven by the plugin's :class:`SportConfig`
(``has_draws``, ``default_score``) and the state saved at train time.
That's what makes a new sport a drop-in.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# StandardScaler drops column names inside the pipeline before LightGBM sees
# them; the resulting sklearn warning is cosmetic and identical to training.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from core.generic_features import build_scoring_row
from data_pipeline.odds_client import OddsClient
from portfolio_manager.arbitrage import scan_h2h_arbitrage
from portfolio_manager.covariance_estimator import CovarianceEstimator
from portfolio_manager.optimizer import PortfolioOptimizer
from predictive_model.model_trainer import ModelTrainer
from risk_bridge.risk_manager import RiskManager
from sports import get_plugin
from sports.base import SportPlugin

logger = logging.getLogger(__name__)


@dataclass
class SlateResult:
    """Everything the UI needs to render one sport's scored slate."""

    sport_key: str
    display_name: str
    has_draws: bool
    trained: bool
    bets: pd.DataFrame = field(default_factory=pd.DataFrame)       # all scored candidate bets
    allocation: pd.DataFrame = field(default_factory=pd.DataFrame)  # non-zero Kelly stakes
    arbitrage: pd.DataFrame = field(default_factory=pd.DataFrame)   # guaranteed-profit rows
    n_games: int = 0
    message: str = ""             # human-readable status / why-empty explanation
    portfolio_edge: float = 0.0


def _american_to_decimal(american: float) -> float:
    return american / 100.0 + 1.0 if american >= 100 else 100.0 / abs(american) + 1.0


def _load_state(plugin: SportPlugin) -> dict:
    with open(plugin.state_path, "rb") as fh:
        return pickle.load(fh)


def score_slate(
    sport_key: str,
    risk_manager: RiskManager,
    provider: Optional[str] = None,
) -> SlateResult:
    """
    Score today's live slate for one sport.

    Never raises for the ordinary "no games / not trained / no edges"
    cases — those come back as a populated ``message`` so the web layer
    can show them plainly.
    """
    plugin = get_plugin(sport_key)
    cfg = plugin.config
    result = SlateResult(
        sport_key=cfg.key,
        display_name=cfg.display_name,
        has_draws=cfg.has_draws,
        trained=plugin.is_trained(),
    )

    if not plugin.is_trained():
        result.message = (
            f"No trained model for {cfg.display_name}. "
            f"Run:  python scripts/train_sport.py --sport {cfg.key}"
        )
        return result

    # --- live odds ----------------------------------------------------
    odds_client = OddsClient(provider=provider, sport_key=cfg.odds_sport_key)
    try:
        odds_df = odds_client.get_game_odds(markets=["h2h"])
    except Exception as exc:  # noqa: BLE001 — provider/network issues are expected
        result.message = f"Odds provider error: {exc}"
        return result

    if odds_df.empty:
        result.message = f"No live {cfg.display_name} games on the board right now."
        return result

    odds_df = odds_client.get_implied_probabilities(odds_df)
    result.n_games = int(odds_df["game_id"].nunique())

    # --- arbitrage (independent of our model) -------------------------
    result.arbitrage = scan_h2h_arbitrage(odds_df)

    # --- model + saved state -----------------------------------------
    trainer = ModelTrainer.load(plugin.model_path)
    state = _load_state(plugin)
    current_elo: Dict[str, float] = state["current_elo"]
    latest_stats: pd.DataFrame = state["latest_stats"]
    calibrator = state.get("calibrator")
    feature_cols: List[str] = state["feature_cols"]

    # --- one feature row per game ------------------------------------
    games = (
        odds_df[["game_id", "home_team", "away_team", "commence_time"]]
        .drop_duplicates("game_id")
        .copy()
    )
    rows = []
    for _, g in games.iterrows():
        home = plugin.normalize_name(g["home_team"])
        away = plugin.normalize_name(g["away_team"])
        row = build_scoring_row(home, away, current_elo, latest_stats, cfg.default_score)
        row.update({
            "game_id": g["game_id"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "commence_time": g["commence_time"],
        })
        rows.append(row)
    feat_df = pd.DataFrame(rows).set_index("game_id")
    score_X = feat_df[feature_cols].fillna(feat_df[feature_cols].median())

    # --- probabilities -----------------------------------------------
    # prob_by_game[game_id] = {"home": p, "draw": p, "away": p}
    prob_by_game: Dict[str, Dict[str, float]] = {}
    if cfg.has_draws:
        proba = trainer.predict_proba_all(score_X)  # cols: 0=away,1=draw,2=home
        for i, gid in enumerate(feat_df.index):
            prob_by_game[gid] = {
                "away": float(proba[i, 0]),
                "draw": float(proba[i, 1]),
                "home": float(proba[i, 2]),
            }
    else:
        raw = trainer.predict_proba(score_X)
        p_home = calibrator.transform(raw) if calibrator is not None else raw
        for i, gid in enumerate(feat_df.index):
            ph = float(np.clip(p_home[i], 1e-4, 1 - 1e-4))
            prob_by_game[gid] = {"home": ph, "away": 1.0 - ph, "draw": 0.0}

    result.bets = _build_bets(odds_df, feat_df, prob_by_game, cfg.has_draws)
    if result.bets.empty:
        result.message = "Odds and model team names could not be aligned for any game."
        return result

    # --- allocation ---------------------------------------------------
    positive = result.bets[result.bets["edge"] > 0].copy()
    if positive.empty:
        result.message = "No positive-edge bets in the current slate (market looks efficient)."
        return result

    max_port, max_single, risk_scaling, bankroll = risk_manager.get_portfolio_constraints()
    cov = CovarianceEstimator()
    cov.fit(positive)
    optimizer = PortfolioOptimizer(
        max_portfolio_fraction=max_port,
        max_single_bet_fraction=max_single,
        risk_scaling=risk_scaling,
    )
    alloc = optimizer.optimise(positive, cov, bankroll)
    alloc = risk_manager.apply_constraints_to_allocation(alloc)

    active = alloc[alloc["wager_amount"] > 0].sort_values("wager_amount", ascending=False)
    result.allocation = active.reset_index(drop=True)
    try:
        result.portfolio_edge = optimizer.expected_portfolio_edge
    except Exception:  # noqa: BLE001
        result.portfolio_edge = 0.0

    if active.empty:
        result.message = (
            "Positive edges exist but all fall below the $1 minimum stake "
            "at your current bankroll/risk settings."
        )
    return result


def _build_bets(
    odds_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    prob_by_game: Dict[str, Dict[str, float]],
    has_draws: bool,
) -> pd.DataFrame:
    """One row per (game, outcome) at the best available price, with edge."""
    h2h = odds_df[odds_df["bet_type"] == "moneyline"]
    outcomes = [("home", "Home"), ("away", "Away")] + ([("draw", "Draw")] if has_draws else [])

    bet_rows = []
    for gid, feat in feat_df.iterrows():
        home, away = feat["home_team"], feat["away_team"]
        g_odds = h2h[h2h["game_id"] == gid]
        probs = prob_by_game.get(gid, {})

        for key, label in outcomes:
            true_p = probs.get(key, 0.0)
            if key == "home":
                match = lambda n: home.lower() in n.lower() or n.lower() in home.lower()
            elif key == "away":
                match = lambda n: away.lower() in n.lower() or n.lower() in away.lower()
            else:
                match = lambda n: "draw" in n.lower() or "tie" in n.lower()

            sub = g_odds[g_odds["outcome_name"].apply(match)]
            if sub.empty:
                continue
            best = sub.loc[sub["price"].idxmax()]
            american = float(best["price"])
            edge = true_p * _american_to_decimal(american) - 1.0

            bet_rows.append({
                "bet_id": f"{gid}__{key}",
                "game_id": gid,
                "matchup": f"{away} @ {home}",
                "bet_type": "moneyline",
                "outcome_name": label,
                "selection": {"home": home, "away": away, "draw": "Draw"}[key],
                "bookmaker": best["bookmaker"],
                "price": american,
                "decimal_odds": _american_to_decimal(american),
                "point": float("nan"),
                "true_prob": true_p,
                "implied_prob": 1.0 / _american_to_decimal(american),
                "edge": edge,
                "commence_time": feat["commence_time"],
            })

    return pd.DataFrame(bet_rows)
