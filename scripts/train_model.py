"""
scripts/train_model.py

Trains the LightGBM win-probability model from a local CSV file produced
by pull_training_data.py.  Requires NO API keys.

What it does
------------
1. Loads data/training_features.csv
2. Defines the stats-only feature set (no market odds required)
3. Temporal train/test split — trains on all-but-last-season, tests on last
4. Trains LightGBM with 5-fold cross-validation
5. Fits an Isotonic Regression calibrator on out-of-fold predictions
6. Prints a full evaluation report
7. Saves artefacts/lgbm_win_prob.pkl and artefacts/isotonic_calibrator.pkl

Usage
-----
  # Run after pull_training_data.py has finished:
  python scripts/train_model.py

  # Override the CSV path or artefact directory:
  python scripts/train_model.py --data data/training_features.csv --artefacts artefacts/

  # Quick smoke-test on a small sample:
  python scripts/train_model.py --sample 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict

from predictive_model.model_trainer import ModelTrainer
from predictive_model.probability_calibrator import IsotonicCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Feature columns — stats only, no market odds required
# ---------------------------------------------------------------------------
# These are the columns produced by pull_training_data.py.
# At *prediction* time, main.py will also have ML_IMPLIED_HOME_PROB,
# CONSENSUS_SPREAD, and CONSENSUS_TOTAL from the live odds feed.
# The model is trained without them so it works even if the odds feed
# is unavailable; they naturally add edge when present.

TRAINING_FEATURE_COLS = [
    # Team quality deltas (home minus away, from lagged season stats)
    "DELTA_PACE",
    "DELTA_OFF_RATING",
    "DELTA_DEF_RATING",
    "DELTA_NET_RATING",
    "DELTA_AST_PCT",
    "DELTA_OREB_PCT",
    "DELTA_DREB_PCT",
    "DELTA_EFG_PCT",
    "DELTA_TS_PCT",
    # Schedule / fatigue
    "HOME_REST_DAYS",
    "AWAY_REST_DAYS",
    "HOME_IS_B2B",
    "AWAY_IS_B2B",
    "HOME_IS_B2B_FIRST",
    "AWAY_IS_B2B_FIRST",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_validate(path: str, sample: int) -> pd.DataFrame:
    if not os.path.exists(path):
        log.error(
            "Data file not found: %s\n"
            "Run this first:  python scripts/pull_training_data.py",
            path,
        )
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    log.info("Loaded %d rows from %s", len(df), path)

    missing = [c for c in TRAINING_FEATURE_COLS + ["HOME_WIN"] if c not in df.columns]
    if missing:
        log.error("CSV is missing required columns: %s", missing)
        sys.exit(1)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        log.info("Sampled down to %d rows for quick test.", len(df))

    return df


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all seasons except the last; test on the last season."""
    seasons_sorted = sorted(df["SEASON"].unique())
    test_season = seasons_sorted[-1]
    log.info("Train seasons: %s", seasons_sorted[:-1])
    log.info("Test season:   %s  (held out)", test_season)
    train = df[df["SEASON"] != test_season].sort_values("GAME_DATE")
    test  = df[df["SEASON"] == test_season].sort_values("GAME_DATE")
    return train, test


def _print_metrics(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    auc    = roc_auc_score(y_true, y_pred)
    brier  = brier_score_loss(y_true, y_pred)
    ll     = log_loss(y_true, y_pred)
    win_rt = y_true.mean()
    log.info(
        "  %-22s  AUC=%.4f  Brier=%.4f  LogLoss=%.4f  (n=%d, win_rate=%.3f)",
        label, auc, brier, ll, len(y_true), win_rt,
    )


def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_pred[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(data_path: str, artefacts_dir: str, sample: int) -> None:

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = _load_and_validate(data_path, sample)
    log.info("Home-win rate: %.3f | Date range: %s → %s",
             df["HOME_WIN"].mean(),
             df["GAME_DATE"].min().date(),
             df["GAME_DATE"].max().date())

    train_df, test_df = _temporal_split(df)
    log.info("Train: %d games  |  Test: %d games", len(train_df), len(test_df))

    X_train = train_df[TRAINING_FEATURE_COLS].copy()
    y_train = train_df["HOME_WIN"].astype(int)
    X_test  = test_df[TRAINING_FEATURE_COLS].copy()
    y_test  = test_df["HOME_WIN"].astype(int)

    # Fill any NaN values (e.g. first game of season has no previous rest data)
    X_train = X_train.fillna(X_train.median())
    X_test  = X_test.fillna(X_train.median())   # use train medians for test

    # ------------------------------------------------------------------
    # 2. Train LightGBM
    # ------------------------------------------------------------------
    log.info("Training LightGBM …")
    trainer = ModelTrainer(
        backend="lgbm",
        feature_cols=TRAINING_FEATURE_COLS,
        hyper_params={
            "n_estimators":      500,
            "learning_rate":     0.03,
            "num_leaves":        31,
            "max_depth":         5,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "min_child_samples": 20,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "verbose":           -1,
        },
    )
    trainer.fit(X_train, y_train)

    # Cross-val log-loss
    log.info("Running 5-fold cross-validation …")
    mean_ll, std_ll = trainer.cross_val_log_loss(X_train, y_train, n_splits=5)
    log.info("  CV log-loss: %.4f +/- %.4f", mean_ll, std_ll)

    # Raw test-set metrics
    raw_train_probs = trainer.predict_proba(X_train)
    raw_test_probs  = trainer.predict_proba(X_test)
    log.info("Raw (uncalibrated) metrics:")
    _print_metrics("train (in-sample)",  y_train.to_numpy(), raw_train_probs)
    _print_metrics("test  (out-of-fold)", y_test.to_numpy(),  raw_test_probs)

    # ------------------------------------------------------------------
    # 3. Calibrate with out-of-fold predictions
    # ------------------------------------------------------------------
    log.info("Fitting isotonic calibrator (5-fold OOF) …")

    X_clean = X_train.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof_probs = cross_val_predict(
            trainer._pipeline,
            X_clean,
            y_train.to_numpy(),
            cv=5,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

    calibrator = IsotonicCalibrator(y_min=0.02, y_max=0.98)
    calibrator.fit(oof_probs, y_train.to_numpy())

    cal_test_probs = calibrator.transform(raw_test_probs)

    # If calibration hurts on test set, use raw probabilities
    ece_raw = _ece(y_test.to_numpy(), raw_test_probs)
    ece_cal = _ece(y_test.to_numpy(), cal_test_probs)
    if ece_cal > ece_raw:
        log.info(
            "Calibration raised ECE (%.4f -> %.4f) — using raw probs as final output.",
            ece_raw, ece_cal,
        )
        final_test_probs = raw_test_probs
    else:
        final_test_probs = cal_test_probs

    # ------------------------------------------------------------------
    # 4. Evaluation report
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 65)
    log.info("  FINAL EVALUATION REPORT")
    log.info("=" * 65)

    _print_metrics("Test — final probabilities", y_test.to_numpy(), final_test_probs)
    log.info("  ECE (raw):        %.4f", ece_raw)
    log.info("  ECE (calibrated): %.4f", ece_cal)
    log.info("  ECE (final used): %.4f", min(ece_raw, ece_cal))

    log.info("")
    log.info("  Top-10 feature importances (gain):")
    for feat, score in trainer.feature_importances.head(10).items():
        bar = "#" * int(score / trainer.feature_importances.max() * 30)
        log.info("    %-28s %s  %.0f", feat, bar, score)

    # Grade the model
    auc = roc_auc_score(y_test.to_numpy(), final_test_probs)
    bs  = brier_score_loss(y_test.to_numpy(), final_test_probs)
    ll  = log_loss(y_test.to_numpy(), final_test_probs)
    log.info("")
    _grade("ROC-AUC",     auc, good=0.60, ok=0.57, direction="higher")
    _grade("Brier Score", bs,  good=0.228, ok=0.232, direction="lower")
    _grade("Log-Loss",    ll,  good=0.640, ok=0.660, direction="lower")

    # ------------------------------------------------------------------
    # 5. Save artefacts
    # ------------------------------------------------------------------
    os.makedirs(artefacts_dir, exist_ok=True)
    model_path = os.path.join(artefacts_dir, "lgbm_win_prob.pkl")
    cal_path   = os.path.join(artefacts_dir, "isotonic_calibrator.pkl")

    trainer.save(model_path)
    calibrator.save(cal_path)

    log.info("")
    log.info("Artefacts saved:")
    log.info("  %s", model_path)
    log.info("  %s", cal_path)
    log.info("")
    log.info("You can now run:  python main.py")


def _grade(metric: str, value: float, good: float, ok: float, direction: str) -> None:
    if direction == "higher":
        if value >= good:
            tag = "GOOD"
        elif value >= ok:
            tag = "OK  "
        else:
            tag = "POOR"
    else:
        if value <= good:
            tag = "GOOD"
        elif value <= ok:
            tag = "OK  "
        else:
            tag = "POOR"
    log.info("  [%s]  %-14s %.4f", tag, metric, value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LightGBM model from training_features.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       default="data/training_features.csv")
    p.add_argument("--artefacts",  default="artefacts")
    p.add_argument(
        "--sample", type=int, default=0,
        help="If > 0, randomly sample this many rows for a quick smoke-test.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.data, args.artefacts, args.sample)
