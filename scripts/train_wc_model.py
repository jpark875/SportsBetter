"""
scripts/train_wc_model.py

Train a 3-class LightGBM model on the World Cup training CSV.

Classes
-------
  0 = away win
  1 = draw
  2 = home win

Output
------
  artefacts/wc_lgbm.pkl         -- fitted ModelTrainer (3-class)
  artefacts/wc_calibrator.pkl   -- passthrough placeholder for pipeline compat

Usage
-----
  python scripts/train_wc_model.py
  python scripts/train_wc_model.py --data data/wc_training_features.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize

from config.settings import WC_FEATURE_COLS, WC_MODEL_PATH, WC_CALIBRATOR_PATH
from predictive_model.model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_wc")


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].quantile(0.85)
    train = df[df["date"] <= cutoff].copy()
    test  = df[df["date"] >  cutoff].copy()
    log.info("Train: %d matches (through %s)", len(train), cutoff.date())
    log.info("Test:  %d matches (%s onward)", len(test), cutoff.date())
    return train, test


def _multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> None:
    ll = log_loss(y_true, proba)
    acc = (proba.argmax(axis=1) == y_true).mean()
    log.info("  Log-loss: %.4f  |  Accuracy: %.1f%%", ll, acc * 100)

    for label, name in zip([0, 1, 2], ["Away win", "Draw    ", "Home win"]):
        n = (y_true == label).sum()
        log.info("    %s  n=%4d  avg_pred=%.3f", name, n, proba[:, label].mean())


def train(data_path: str, artefacts_dir: str) -> None:
    if not os.path.exists(data_path):
        log.error(
            "Training data not found: %s\n"
            "Run first:  python scripts/pull_wc_data.py",
            data_path,
        )
        sys.exit(1)

    df = pd.read_csv(data_path, parse_dates=["date"])
    log.info("Loaded %d rows from %s", len(df), data_path)

    missing = [c for c in WC_FEATURE_COLS + ["RESULT"] if c not in df.columns]
    if missing:
        log.error("CSV missing columns: %s", missing)
        sys.exit(1)

    df = df.dropna(subset=WC_FEATURE_COLS + ["RESULT"])
    log.info("After dropping NaN rows: %d", len(df))

    train_df, test_df = _temporal_split(df)

    X_train = train_df[WC_FEATURE_COLS].fillna(train_df[WC_FEATURE_COLS].median())
    y_train = train_df["RESULT"].astype(int)
    X_test  = test_df[WC_FEATURE_COLS].fillna(train_df[WC_FEATURE_COLS].median())
    y_test  = test_df["RESULT"].astype(int)

    log.info("Training 3-class LightGBM …")
    trainer = ModelTrainer(
        backend="lgbm",
        feature_cols=WC_FEATURE_COLS,
        hyper_params={
            "objective":         "multiclass",
            "metric":            "multi_logloss",
            "num_class":         3,
            "n_estimators":      600,
            "learning_rate":     0.02,
            "num_leaves":        31,
            "max_depth":         5,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "min_child_samples": 15,
            "reg_alpha":         0.1,
            "reg_lambda":        1.0,
            "verbose":           -1,
        },
    )
    trainer.fit(X_train, y_train)

    log.info("Train-set metrics:")
    train_proba = trainer.predict_proba_all(X_train)
    _multiclass_metrics(y_train.to_numpy(), train_proba)

    log.info("Test-set metrics:")
    test_proba = trainer.predict_proba_all(X_test)
    _multiclass_metrics(y_test.to_numpy(), test_proba)

    log.info("Top-10 feature importances (gain):")
    for feat, score in trainer.feature_importances.head(10).items():
        bar = "#" * int(score / trainer.feature_importances.max() * 30)
        log.info("  %-32s %s  %.0f", feat, bar, score)

    os.makedirs(artefacts_dir, exist_ok=True)
    trainer.save(WC_MODEL_PATH)

    # Save a passthrough calibrator placeholder so main.py can load it
    # without special-casing the WC pipeline.
    with open(WC_CALIBRATOR_PATH, "wb") as fh:
        pickle.dump(None, fh)
    log.info("Placeholder calibrator saved -> %s", WC_CALIBRATOR_PATH)

    log.info("")
    log.info("Done. Run:  python main.py --sport wc")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3-class LightGBM World Cup model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      default="data/wc_training_features.csv")
    p.add_argument("--artefacts", default="artefacts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.data, args.artefacts)
