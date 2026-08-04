"""
predictive_model/probability_calibrator.py

Isotonic Regression calibration layer that converts raw LightGBM/XGBoost
output scores into true probabilities.

Why calibrate?
--------------
Gradient-boosted classifiers are well-ranked but systematically
miscalibrated — they tend to push confident predictions toward 0.0 and 1.0
more than the true data-generating process warrants.  Isotonic Regression
non-parametrically corrects the score-to-probability mapping using a
held-out calibration set without disturbing the model's rank order.

Usage pattern
-------------
::

    trainer    = ModelTrainer().fit(X_train, y_train)
    raw_probs  = trainer.predict_proba(X_cal)
    calibrator = IsotonicCalibrator().fit(raw_probs, y_cal)
    true_probs = calibrator.transform(trainer.predict_proba(X_test))
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from config.settings import CALIBRATOR_PATH

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Wraps scikit-learn's :class:`~sklearn.isotonic.IsotonicRegression` with
    diagnostic utilities and a persistence interface.

    Parameters
    ----------
    y_min : float
        Lower clip bound for output probabilities.  Default ``0.01``.
    y_max : float
        Upper clip bound for output probabilities.  Default ``0.99``.
    """

    def __init__(self, y_min: float = 0.01, y_max: float = 0.99) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=y_min, y_max=y_max)
        self._is_fitted: bool = False
        self.y_min = y_min
        self.y_max = y_max

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Fit the isotonic mapping on a held-out calibration set.

        Parameters
        ----------
        raw_scores : np.ndarray, shape (n_samples,)
            Uncalibrated probabilities from :meth:`ModelTrainer.predict_proba`.
        y_true : np.ndarray, shape (n_samples,)
            Binary ground-truth labels (0 or 1).

        Returns
        -------
        IsotonicCalibrator
            Returns ``self``.
        """
        raw_scores = np.asarray(raw_scores, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()

        logger.info(
            "Fitting isotonic calibrator on %d samples …", len(raw_scores)
        )
        self._iso.fit(raw_scores, y_true)
        self._is_fitted = True

        # Diagnostic log
        calibrated = self._iso.transform(raw_scores)
        ll_before = log_loss(y_true, raw_scores)
        ll_after = log_loss(y_true, calibrated)
        bs_before = brier_score_loss(y_true, raw_scores)
        bs_after = brier_score_loss(y_true, calibrated)
        logger.info(
            "Calibration complete.  "
            "Log-loss: %.4f → %.4f | Brier: %.4f → %.4f",
            ll_before, ll_after, bs_before, bs_after,
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Map raw model scores to calibrated probabilities.

        Parameters
        ----------
        raw_scores : np.ndarray, shape (n_samples,)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Calibrated probabilities clipped to ``[y_min, y_max]``.
        """
        self._assert_fitted()
        raw_scores = np.asarray(raw_scores, dtype=float).ravel()
        return self._iso.transform(raw_scores)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def reliability_diagram_data(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Compute data for a reliability (calibration) diagram.

        Returns a dict with keys ``fraction_of_positives`` and
        ``mean_predicted_value`` for both before and after calibration,
        suitable for plotting with matplotlib.

        Parameters
        ----------
        raw_scores : np.ndarray
        y_true : np.ndarray
        n_bins : int

        Returns
        -------
        dict
            ``{
                "before": (fraction_of_positives, mean_predicted),
                "after":  (fraction_of_positives, mean_predicted),
            }``
        """
        self._assert_fitted()
        frac_before, mean_before = calibration_curve(
            y_true, raw_scores, n_bins=n_bins, strategy="uniform"
        )
        calibrated = self.transform(raw_scores)
        frac_after, mean_after = calibration_curve(
            y_true, calibrated, n_bins=n_bins, strategy="uniform"
        )
        return {
            "before": (frac_before, mean_before),
            "after": (frac_after, mean_after),
        }

    def ece(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE) of the *calibrated* probabilities.

        Lower is better; 0.0 is perfect calibration.

        Parameters
        ----------
        raw_scores : np.ndarray
        y_true : np.ndarray
        n_bins : int

        Returns
        -------
        float
        """
        self._assert_fitted()
        calibrated = self.transform(raw_scores)
        y_true = np.asarray(y_true, dtype=float).ravel()

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece_val = 0.0
        n = len(y_true)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (calibrated >= lo) & (calibrated < hi)
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = calibrated[mask].mean()
            ece_val += (mask.sum() / n) * abs(acc - conf)
        return float(ece_val)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Serialise calibrator to disk."""
        self._assert_fitted()
        path = path or CALIBRATOR_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Calibrator saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "IsotonicCalibrator":
        """Load a previously saved calibrator."""
        path = path or CALIBRATOR_PATH
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected IsotonicCalibrator, got {type(obj)}.")
        logger.info("Calibrator loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "IsotonicCalibrator is not fitted. Call .fit(raw_scores, y_true) first."
            )
