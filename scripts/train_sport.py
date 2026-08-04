"""
scripts/train_sport.py

Unified per-sport trainer.  One command per sport, any registered sport:

    python scripts/train_sport.py --sport mlb
    python scripts/train_sport.py --sport epl mlb wnba nba   # several in a row

For each sport it:
    1. fetches full history via the sport's plugin,
    2. builds the generic ELO/form/attack-defense feature table,
    3. trains LightGBM — binary + isotonic calibration for 2-way sports,
       3-class for draw sports — with a temporal 85/15 split,
    4. reports held-out log-loss/accuracy and feature importances,
    5. saves  <key>_model.pkl  and  <key>_state.pkl  (current ELO + latest
       team stats + calibrator), which is everything the web app needs to
       score live games without re-walking history.

Model choice rationale: for betting, calibration beats raw accuracy —
Kelly stakes are wrong exactly as much as the probabilities are.  Hence
gradient boosting for signal extraction plus isotonic regression on a
temporal holdout for honest probabilities.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.metrics import log_loss

from core.generic_features import GENERIC_FEATURE_COLS, build_feature_table, latest_team_stats
from predictive_model.model_trainer import ModelTrainer
from predictive_model.probability_calibrator import IsotonicCalibrator
from sports import available_sports, get_plugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_sport")

_LGBM_BASE = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}


def train_sport(key: str) -> None:
    plugin = get_plugin(key)
    cfg = plugin.config
    log.info("=== %s ===", cfg.display_name)

    history = plugin.fetch_history()
    feat_df, current_elo = build_feature_table(
        history,
        k_factor=cfg.elo_k,
        hfa_elo=cfg.hfa_elo,
        form_window=cfg.form_window,
        ewm_span=cfg.ewm_span,
    )

    df = feat_df.dropna(subset=GENERIC_FEATURE_COLS).copy()
    cutoff = df["date"].quantile(0.85)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]
    log.info("Train %d / test %d games (cutoff %s).",
             len(train_df), len(test_df), cutoff.date())

    X_train = train_df[GENERIC_FEATURE_COLS]
    X_test = test_df[GENERIC_FEATURE_COLS]

    calibrator = None
    if cfg.has_draws:
        y_train = train_df["RESULT"].astype(int)
        y_test = test_df["RESULT"].astype(int)
        trainer = ModelTrainer(
            backend="lgbm",
            feature_cols=GENERIC_FEATURE_COLS,
            hyper_params={
                **_LGBM_BASE,
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": 3,
            },
        )
        trainer.fit(X_train, y_train)
        proba = trainer.predict_proba_all(X_test)
        acc = (proba.argmax(axis=1) == y_test.to_numpy()).mean()
        log.info("Test: log-loss %.4f | accuracy %.1f%% (3-way)",
                 log_loss(y_test, proba, labels=[0, 1, 2]), acc * 100)
    else:
        y_train = (train_df["RESULT"] == 2).astype(int)
        y_test = (test_df["RESULT"] == 2).astype(int)
        trainer = ModelTrainer(
            backend="lgbm",
            feature_cols=GENERIC_FEATURE_COLS,
            hyper_params={**_LGBM_BASE, "objective": "binary", "metric": "binary_logloss"},
        )
        trainer.fit(X_train, y_train)

        # Calibrate on the temporal holdout, report on its second half so
        # the calibration metric isn't graded on its own fit data.
        raw_test = trainer.predict_proba(X_test)
        half = len(test_df) // 2
        calibrator = IsotonicCalibrator()
        calibrator.fit(raw_test[:half], y_test.to_numpy()[:half])
        cal_probs = calibrator.transform(raw_test[half:])
        y_eval = y_test.to_numpy()[half:]
        acc = ((cal_probs > 0.5).astype(int) == y_eval).mean()
        log.info("Test: log-loss %.4f | accuracy %.1f%% | home base rate %.1f%%",
                 log_loss(y_eval, cal_probs), acc * 100, y_eval.mean() * 100)

    log.info("Top-5 features (gain):")
    for feat, score in trainer.feature_importances.head(5).items():
        log.info("  %-28s %.0f", feat, score)

    os.makedirs(os.path.dirname(plugin.model_path) or ".", exist_ok=True)
    trainer.save(plugin.model_path)

    state = {
        "sport_key": cfg.key,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "has_draws": cfg.has_draws,
        "feature_cols": GENERIC_FEATURE_COLS,
        "current_elo": current_elo,
        "latest_stats": latest_team_stats(feat_df),
        "calibrator": calibrator,
        "history_through": str(df["date"].max().date()),
        "n_games": len(df),
    }
    with open(plugin.state_path, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("State saved -> %s (history through %s)\n",
             plugin.state_path, state["history_through"])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train per-sport models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sport", nargs="+", required=True,
        choices=available_sports() + ["all"],
        help="Sport key(s) to train, or 'all'.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    keys = available_sports() if "all" in args.sport else args.sport
    failures = []
    for k in keys:
        try:
            train_sport(k)
        except Exception as exc:  # noqa: BLE001 — train the rest even if one source is down
            log.error("%s failed: %s", k, exc)
            failures.append(k)
    if failures:
        log.error("Failed sports: %s", ", ".join(failures))
        sys.exit(1)
