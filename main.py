"""
main.py

NBA Bet Portfolio Engine — entry script.

Execution order
---------------
1.  Prompt user for risk profile (bankroll, disposable income, tolerance).
2.  Ingest live odds for tonight's slate via OddsClient.
3.  Pull player & team stats from nba_api.
4.  Engineer features and align by game_id / player_id.
5.  Load (or train) the LightGBM model + isotonic calibrator.
6.  Generate calibrated win probabilities for every active wager.
7.  Estimate the inter-bet covariance matrix.
8.  Run the SLSQP portfolio optimiser with risk-bridge constraints.
9.  Print the final allocation table.

Run
---
    python main.py [--train] [--season 2024-25] [--provider theodds|oddsjam]

Flags
-----
    --train     Re-train the model on cached historical data before scoring.
    --season    NBA season string (default: ``"2024-25"``).
    --provider  Odds provider (default from .env / ``"theodds"``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup (must happen before any module-level loggers fire)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Internal imports (after logging is configured)
# ---------------------------------------------------------------------------
from config.settings import UserRiskProfile, WC_SPORT_KEY, NBA_SPORT_KEY
from data_pipeline.nba_stats_client import NBAStatsClient
from data_pipeline.odds_client import OddsClient
from predictive_model.feature_engineering import FeatureEngineer
from predictive_model.model_trainer import ModelTrainer
from predictive_model.probability_calibrator import IsotonicCalibrator
from portfolio_manager.covariance_estimator import CovarianceEstimator
from portfolio_manager.optimizer import PortfolioOptimizer
from risk_bridge.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA Bet Portfolio Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Re-train the model before scoring tonight's slate.",
    )
    parser.add_argument(
        "--season",
        default="2024-25",
        help="NBA season string passed to nba_api.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Odds provider: 'theodds' or 'oddsjam'. Overrides .env setting.",
    )
    parser.add_argument(
        "--sport",
        default="nba",
        choices=["nba", "wc"],
        help="Sport to score: 'nba' (basketball) or 'wc' (FIFA World Cup).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# User risk profile prompt
# ---------------------------------------------------------------------------

def _prompt_risk_profile() -> UserRiskProfile:
    """Interactively collect the user's financial parameters."""
    print("\n=== NBA Bet Portfolio Engine — Risk Profile Setup ===")
    try:
        bankroll = float(input("  Liquid bankroll ($): ").strip())
        disposable = float(input("  Disposable income for betting ($): ").strip())
        tolerance = int(input("  Volatility tolerance (1 = conservative … 10 = aggressive): ").strip())
        return UserRiskProfile(
            liquid_bankroll=bankroll,
            disposable_income=disposable,
            volatility_tolerance=tolerance,
        )
    except (ValueError, EOFError) as exc:
        logger.warning("Invalid input (%s). Using conservative defaults.", exc)
        return UserRiskProfile()


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------

def _load_or_train_model(
    train: bool,
    feature_df: pd.DataFrame,
) -> tuple[ModelTrainer, IsotonicCalibrator]:
    """
    Return a fitted (ModelTrainer, IsotonicCalibrator) pair.

    If ``--train`` is set, or if no saved model exists, trains from scratch
    using labelled rows in ``feature_df`` (rows where ``HOME_WIN`` is not NaN).

    Parameters
    ----------
    train : bool
    feature_df : pd.DataFrame
        Feature matrix from FeatureEngineer.  Must include ``HOME_WIN`` column
        for training rows.

    Returns
    -------
    tuple[ModelTrainer, IsotonicCalibrator]
    """
    from config.settings import LGBM_MODEL_PATH, CALIBRATOR_PATH

    if not train and os.path.exists(LGBM_MODEL_PATH) and os.path.exists(CALIBRATOR_PATH):
        logger.info("Loading existing model artefacts …")
        trainer = ModelTrainer.load(LGBM_MODEL_PATH)
        calibrator = IsotonicCalibrator.load(CALIBRATOR_PATH)
        return trainer, calibrator

    logger.info("Training new model …")
    labelled = feature_df[feature_df["HOME_WIN"].notna()].copy()
    if len(labelled) < 50:
        logger.error(
            "Insufficient labelled data (%d rows). Provide historical data to train.",
            len(labelled),
        )
        sys.exit(1)

    y = labelled.pop("HOME_WIN")
    X = labelled.drop(columns=["home_team", "away_team", "commence_time"], errors="ignore")

    # 80 / 20 temporal split — keep future rows as calibration holdout
    split = int(len(X) * 0.8)
    X_train, X_cal = X.iloc[:split], X.iloc[split:]
    y_train, y_cal = y.iloc[:split], y.iloc[split:]

    trainer = ModelTrainer(backend="lgbm")
    trainer.fit(X_train, y_train)

    raw_cal_probs = trainer.predict_proba(X_cal)
    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_cal_probs, y_cal.to_numpy())

    trainer.save()
    calibrator.save()
    return trainer, calibrator


# ---------------------------------------------------------------------------
# Scoring-time stats loader (fast path: 2 API calls instead of 30+)
# ---------------------------------------------------------------------------

def _load_scoring_stats(
    nba_client: NBAStatsClient,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch team advanced metrics and schedule features for live scoring.

    Replaces ``load_all()`` at scoring time — makes 2 API calls instead of
    the 30+ TeamGameLog calls that ``fetch_schedule_features()`` requires.
    Both failures are non-fatal: FeatureEngineer fills missing stat values
    with NaN, and schedule columns default to NaN if unavailable.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(team_df, sched_df)``
    """
    team_df = pd.DataFrame()
    sched_df = pd.DataFrame()

    try:
        team_df = nba_client.fetch_team_advanced_stats()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Team stats unavailable (%s). DELTA_* features will be NaN.", exc
        )

    try:
        sched_df = nba_client.fetch_schedule_features_fast()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Schedule features unavailable (%s). Rest-day features will be NaN.", exc
        )

    return team_df, sched_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the full pipeline for tonight's slate."""

    # ------------------------------------------------------------------
    # 1. Risk profile
    # ------------------------------------------------------------------
    profile = _prompt_risk_profile()
    risk_manager = RiskManager(profile)

    if not risk_manager.is_session_live:
        logger.info("Session stop-loss already reached. Exiting.")
        return

    max_port_frac, max_single, risk_scaling, bankroll = (
        risk_manager.get_portfolio_constraints()
    )
    logger.info(
        "Effective bankroll: $%.2f | Max total exposure: %.1f%%",
        bankroll, max_port_frac * 100,
    )

    # ------------------------------------------------------------------
    # 2. Ingest live odds
    # ------------------------------------------------------------------
    logger.info("Fetching live odds …")
    odds_client = OddsClient(provider=args.provider)
    game_odds_df = odds_client.get_game_odds()

    if game_odds_df.empty:
        logger.warning("No live odds returned. NBA may be off-season or API key invalid.")
        return

    game_odds_df = odds_client.get_implied_probabilities(game_odds_df)

    logger.info("  %d odds records received.", len(game_odds_df))

    # Attempt to fetch props for each event (TheOddsAPI requires event IDs)
    prop_df = pd.DataFrame()
    if hasattr(odds_client._adapter, "fetch_event_ids"):
        try:
            events = odds_client._adapter.fetch_event_ids()
            prop_frames = []
            for _, ev_row in events.iterrows():
                try:
                    pf = odds_client.get_player_props(event_id=ev_row["event_id"])
                    pf = odds_client.get_implied_probabilities(pf)
                    prop_frames.append(pf)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Props fetch failed for event %s: %s", ev_row["event_id"], exc)
            if prop_frames:
                prop_df = pd.concat(prop_frames, ignore_index=True)
                logger.info("  %d player-prop records received.", len(prop_df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch player props: %s", exc)

    # ------------------------------------------------------------------
    # 3. Pull NBA stats (2 API calls: team advanced + schedule log)
    # ------------------------------------------------------------------
    logger.info("Fetching NBA stats (season: %s) …", args.season)
    nba_client = NBAStatsClient(season=args.season)
    team_df, sched_df = _load_scoring_stats(nba_client)
    player_df = pd.DataFrame()  # prop features not needed for moneyline/spread

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Engineering features …")
    engineer = FeatureEngineer(
        team_stats=team_df,
        schedule_features=sched_df,
        player_stats=player_df,
    )
    game_features = engineer.build_game_features(game_odds_df)

    prop_features = pd.DataFrame()
    if not prop_df.empty:
        prop_features = engineer.build_prop_features(prop_df)

    # ------------------------------------------------------------------
    # 5. Load / train model
    # ------------------------------------------------------------------
    trainer, calibrator = _load_or_train_model(args.train, game_features)

    # ------------------------------------------------------------------
    # 6. Predict calibrated probabilities
    # ------------------------------------------------------------------
    score_df = game_features.drop(
        columns=["home_team", "away_team", "commence_time", "HOME_WIN"],
        errors="ignore",
    )
    raw_probs = trainer.predict_proba(score_df)
    cal_probs = calibrator.transform(raw_probs)
    game_features["predicted_home_win_prob"] = cal_probs

    # Build unified bets table: one row per distinct wager opportunity
    bets_df = _build_bets_table(game_odds_df, game_features)

    if bets_df.empty:
        logger.warning("No bets constructed from the feature matrix.")
        return

    logger.info("  %d candidate wagers assembled.", len(bets_df))

    # ------------------------------------------------------------------
    # 7. Covariance estimation
    # ------------------------------------------------------------------
    cov_estimator = CovarianceEstimator()
    cov_estimator.fit(bets_df)

    high_corr = cov_estimator.highly_correlated_pairs(threshold=0.70)
    if high_corr:
        logger.info(
            "High-correlation pairs (≥70%%): %s",
            [(a, b, f"{c:.2f}") for a, b, c in high_corr[:5]],
        )

    # ------------------------------------------------------------------
    # 8. Portfolio optimisation
    # ------------------------------------------------------------------
    optimizer = PortfolioOptimizer(
        max_portfolio_fraction=max_port_frac,
        max_single_bet_fraction=max_single,
        risk_scaling=risk_scaling,
    )
    allocation_df = optimizer.optimise(bets_df, cov_estimator, bankroll)
    allocation_df = risk_manager.apply_constraints_to_allocation(allocation_df)

    # ------------------------------------------------------------------
    # 9. Output
    # ------------------------------------------------------------------
    _print_allocation(allocation_df, optimizer)


def _build_bets_table(
    odds_df: pd.DataFrame,
    game_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join best-line odds with calibrated probabilities to produce the
    unified bets table consumed by the covariance estimator and optimiser.

    Each row represents one independently actionable wager with columns:
    ``bet_id, game_id, bet_type, outcome_name, bookmaker, price, point,
    true_prob, edge``.
    """
    # Use best available line per (game_id, bet_type, outcome_name)
    best = (
        odds_df.sort_values("price", ascending=False)
        .groupby(["game_id", "bet_type", "outcome_name", "point"], dropna=False)
        .first()
        .reset_index()
    )

    # Attach calibrated home-win probability from game features
    prob_map = game_features["predicted_home_win_prob"].to_dict()
    best["true_prob"] = best["game_id"].map(prob_map)

    # For away-team moneylines flip the probability
    away_mask = best["bet_type"] == "moneyline"
    away_mask &= best.apply(
        lambda r: r["outcome_name"].lower() in r.get("away_team", "").lower()
        if "away_team" in r.index else False,
        axis=1,
    )
    best.loc[away_mask, "true_prob"] = 1.0 - best.loc[away_mask, "true_prob"]

    # Drop rows without probability estimate
    best = best.dropna(subset=["true_prob"]).copy()

    # Construct unique bet_id
    best["bet_id"] = (
        best["game_id"] + "__"
        + best["bet_type"] + "__"
        + best["outcome_name"].str.replace(" ", "_")
        + "__"
        + best["point"].astype(str)
    )

    return best.reset_index(drop=True)


def _print_allocation(
    allocation_df: pd.DataFrame,
    optimizer: PortfolioOptimizer,
) -> None:
    """Pretty-print the final allocation table."""
    active = allocation_df[allocation_df["wager_amount"] > 0].sort_values(
        "wager_amount", ascending=False
    )

    print("\n" + "=" * 72)
    print("  NBA BET PORTFOLIO — TONIGHT'S ALLOCATION")
    print("=" * 72)

    if active.empty:
        print("  No positive-edge bets with non-zero allocation found.")
        print("=" * 72)
        return

    col_map = {
        "game_id": "Game",
        "bet_type": "Type",
        "outcome_name": "Selection",
        "bookmaker": "Book",
        "price": "Odds",
        "true_prob": "P(Win)",
        "edge": "Edge",
        "kelly_fraction": "Full Kelly",
        "alloc_fraction": "Alloc %",
        "wager_amount": "Wager $",
    }

    display_cols = [c for c in col_map if c in active.columns]
    display = active[display_cols].rename(columns=col_map).copy()

    for pct_col in ("P(Win)", "Edge", "Full Kelly", "Alloc %"):
        if pct_col in display.columns:
            display[pct_col] = display[pct_col].map("{:.1%}".format)
    if "Wager $" in display.columns:
        display["Wager $"] = display["Wager $"].map("${:.2f}".format)
    if "Odds" in display.columns:
        display["Odds"] = display["Odds"].map("{:+.0f}".format)

    print(display.to_string(index=False))
    print("-" * 72)
    print(
        f"  Total wagered : ${active['wager_amount'].sum():.2f}"
        f"  |  Portfolio edge : {optimizer.expected_portfolio_edge:.1%}"
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# World Cup pipeline
# ---------------------------------------------------------------------------

def _run_wc(args: argparse.Namespace) -> None:
    """Full World Cup scoring pipeline (3-way: home / draw / away)."""
    from config.settings import (
        FOOTBALL_DATA_URL, GOALSCORERS_DATA_URL,
        WC_FEATURE_COLS, WC_MODEL_PATH,
    )
    from data_pipeline.football_stats_client import (
        load_data, load_goalscorers,
        compute_elo_history, compute_form_history,
        compute_attack_defense_ratings, compute_player_concentration,
        get_current_state, build_scoring_row, normalize_name,
    )
    from portfolio_manager.arbitrage import scan_for_arbitrage, print_arb_report

    FEAT_COLS = WC_FEATURE_COLS

    # 1. Risk profile
    profile = _prompt_risk_profile()
    risk_manager = RiskManager(profile)
    if not risk_manager.is_session_live:
        logger.info("Session stop-loss already reached. Exiting.")
        return
    max_port_frac, max_single, risk_scaling, bankroll = (
        risk_manager.get_portfolio_constraints()
    )
    logger.info(
        "Effective bankroll: $%.2f | Max total exposure: %.1f%%",
        bankroll, max_port_frac * 100,
    )

    # 2. Fetch World Cup odds
    logger.info("Fetching World Cup odds (sport_key=%s) …", WC_SPORT_KEY)
    odds_client = OddsClient(provider=args.provider, sport_key=WC_SPORT_KEY)
    odds_df = odds_client.get_game_odds(markets=["h2h"])  # 1X2 only
    if odds_df.empty:
        logger.warning(
            "No World Cup odds returned. Possible causes:\n"
            "  - No games scheduled today\n"
            "  - Sport key mismatch: check WC_SPORT_KEY in .env",
        )
        return
    odds_df = odds_client.get_implied_probabilities(odds_df)
    logger.info("  %d odds rows received for %d games.",
                len(odds_df), odds_df["game_id"].nunique())

    # 2b. Arbitrage scan (before Kelly — guaranteed profit takes priority)
    arb_df = scan_for_arbitrage(odds_df)
    print_arb_report(arb_df, budget=bankroll / max(len(odds_df["game_id"].unique()), 1))

    # 3. Build full historical feature matrix
    logger.info("Downloading match results …")
    raw = load_data(FOOTBALL_DATA_URL)

    logger.info("Downloading goalscorer data …")
    goalscorers = load_goalscorers(GOALSCORERS_DATA_URL)

    logger.info("Computing ELO history …")
    raw = compute_elo_history(raw)

    logger.info("Computing rolling form features …")
    raw = compute_form_history(raw, window=5, competitive_only=True)

    logger.info("Computing EWM attack/defense ratings …")
    raw = compute_attack_defense_ratings(raw, span=7)

    logger.info("Computing player star-concentration …")
    raw = compute_player_concentration(raw, goalscorers, window=20)

    raw["IS_NEUTRAL"] = raw["neutral"].astype(int)
    current_elo, latest_stats = get_current_state(raw)

    # 4. Build one feature row per unique game in tonight's slate
    games = (
        odds_df[["game_id", "home_team", "away_team", "commence_time"]]
        .drop_duplicates("game_id")
        .copy()
    )
    games["home_team"] = games["home_team"].apply(normalize_name)
    games["away_team"] = games["away_team"].apply(normalize_name)

    feature_rows = []
    for _, g in games.iterrows():
        row = build_scoring_row(
            g["home_team"], g["away_team"],
            current_elo, latest_stats, is_neutral=True,
        )
        row["game_id"]       = g["game_id"]
        row["home_team"]     = g["home_team"]
        row["away_team"]     = g["away_team"]
        row["commence_time"] = g["commence_time"]
        feature_rows.append(row)
    feature_df = pd.DataFrame(feature_rows).set_index("game_id")

    # 5. Load WC model
    if not os.path.exists(WC_MODEL_PATH):
        logger.error(
            "WC model not found: %s\n"
            "Run:  python scripts/pull_wc_data.py && python scripts/train_wc_model.py",
            WC_MODEL_PATH,
        )
        return
    logger.info("Loading WC model from %s …", WC_MODEL_PATH)
    trainer = ModelTrainer.load(WC_MODEL_PATH)

    # 6. Predict P(away win), P(draw), P(home win) for each game
    missing_feats = [c for c in FEAT_COLS if c not in feature_df.columns]
    if missing_feats:
        logger.warning("Feature columns missing from scoring row: %s", missing_feats)
    score_df = feature_df[FEAT_COLS].fillna(feature_df[FEAT_COLS].median())
    proba = trainer.predict_proba_all(score_df)  # shape (n_games, 3)

    # 7. Build bets table — one row per (game, outcome)
    bets_rows = []
    result_labels = {0: "Away win", 1: "Draw", 2: "Home win"}
    for i, (gid, g_feat) in enumerate(feature_df.iterrows()):
        home = g_feat["home_team"]
        away = g_feat["away_team"]
        h_atk   = g_feat.get("HOME_ATK_RATING", float("nan"))
        a_atk   = g_feat.get("AWAY_ATK_RATING", float("nan"))
        h_conc  = g_feat.get("HOME_STAR_CONC",  float("nan"))
        a_conc  = g_feat.get("AWAY_STAR_CONC",  float("nan"))

        g_odds = odds_df[
            (odds_df["game_id"] == gid) & (odds_df["bet_type"] == "moneyline")
        ]
        for outcome_idx, outcome_label in result_labels.items():
            true_p = float(proba[i, outcome_idx])
            if outcome_idx == 2:
                name_match = home
            elif outcome_idx == 0:
                name_match = away
            else:
                name_match = "Draw"

            mask = g_odds["outcome_name"].str.lower().apply(
                lambda n: name_match.lower() in n or n in name_match.lower()
                if outcome_idx != 1
                else "draw" in n.lower()
            )
            sub = g_odds[mask]
            if sub.empty:
                continue
            best_row = sub.loc[sub["price"].idxmax()]

            american = float(best_row["price"])
            decimal  = (american / 100 + 1) if american >= 100 else (100 / abs(american) + 1)
            edge     = true_p * decimal - 1

            bets_rows.append({
                "bet_id":        f"{gid}__{outcome_label.replace(' ', '_')}",
                "game_id":       gid,
                "bet_type":      "moneyline",
                "outcome_name":  outcome_label,
                "bookmaker":     best_row["bookmaker"],
                "price":         american,
                "point":         float("nan"),
                "true_prob":     true_p,
                "edge":          edge,
                "home_team":     home,
                "away_team":     away,
                "home_atk":      h_atk,
                "away_atk":      a_atk,
                "home_star_conc": h_conc,
                "away_star_conc": a_conc,
            })

    bets_df = pd.DataFrame(bets_rows)
    if bets_df.empty:
        logger.warning("No bets assembled — odds and feature team names may not align.")
        return

    bets_df = bets_df[bets_df["edge"] > 0].copy()
    if bets_df.empty:
        logger.info("No positive-edge bets found in today's World Cup slate.")
        return

    logger.info("  %d positive-edge bets across %d games.",
                len(bets_df), bets_df["game_id"].nunique())

    # 8. Covariance + optimisation
    cov_estimator = CovarianceEstimator()
    cov_estimator.fit(bets_df)

    optimizer = PortfolioOptimizer(
        max_portfolio_fraction=max_port_frac,
        max_single_bet_fraction=max_single,
        risk_scaling=risk_scaling,
    )
    allocation_df = optimizer.optimise(bets_df, cov_estimator, bankroll)
    allocation_df = risk_manager.apply_constraints_to_allocation(allocation_df)

    # 9. Print allocation + context
    _print_wc_allocation(allocation_df, optimizer, bets_df)


def _print_wc_allocation(
    allocation_df: pd.DataFrame,
    optimizer: PortfolioOptimizer,
    bets_df: pd.DataFrame,
) -> None:
    """Print the WC allocation with attack/defense and star-concentration context."""
    active = allocation_df[allocation_df["wager_amount"] > 0].sort_values(
        "wager_amount", ascending=False
    )

    print("\n" + "=" * 72)
    print("  WORLD CUP BET PORTFOLIO — TODAY'S ALLOCATION")
    print("=" * 72)

    if active.empty:
        print("  No positive-edge bets with non-zero allocation found.")
        print("=" * 72)
        return

    # Merge context columns from the full bets table
    ctx_cols = ["bet_id", "home_atk", "away_atk", "home_star_conc", "away_star_conc"]
    ctx = bets_df[[c for c in ctx_cols if c in bets_df.columns]].copy()
    if "bet_id" in active.columns and not ctx.empty:
        active = active.merge(ctx, on="bet_id", how="left")

    col_map = {
        "game_id":       "Game",
        "outcome_name":  "Outcome",
        "bookmaker":     "Book",
        "price":         "Odds",
        "true_prob":     "P(Win)",
        "edge":          "Edge",
        "alloc_fraction":"Alloc %",
        "wager_amount":  "Wager $",
        "home_atk":      "H-Atk",
        "away_atk":      "A-Atk",
        "home_star_conc":"H-Star%",
        "away_star_conc":"A-Star%",
    }
    display_cols = [c for c in col_map if c in active.columns]
    display = active[display_cols].rename(columns=col_map).copy()

    for pct_col in ("P(Win)", "Edge", "Alloc %"):
        if pct_col in display.columns:
            display[pct_col] = display[pct_col].map("{:.1%}".format)
    if "Wager $" in display.columns:
        display["Wager $"] = display["Wager $"].map("${:.2f}".format)
    if "Odds" in display.columns:
        display["Odds"] = display["Odds"].map("{:+.0f}".format)
    for atk_col in ("H-Atk", "A-Atk"):
        if atk_col in display.columns:
            display[atk_col] = display[atk_col].map("{:.2f}".format)
    for conc_col in ("H-Star%", "A-Star%"):
        if conc_col in display.columns:
            display[conc_col] = display[conc_col].map("{:.0%}".format)

    print(display.to_string(index=False))
    print("-" * 72)
    print(
        f"  Total wagered : ${active['wager_amount'].sum():.2f}"
        f"  |  Portfolio edge : {optimizer.expected_portfolio_edge:.1%}"
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args = _parse_args()
    if _args.sport == "wc":
        _run_wc(_args)
    else:
        run(_args)
