"""Estimates the covariance (correlation) matrix among the active wagers in tonight's
betting slate.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import CORRELATION_SHRINKAGE_ALPHA

logger = logging.getLogger(__name__)

MIN_WITHIN_GAME_CORR: float = 0.50   # structural lower bound for same-game pairs
N_SIM_PATHS: int = 10_000            # Monte Carlo paths for sample covariance


class CovarianceEstimator:
    """Estimates a regularised covariance / correlation matrix for a slate of bets."""

    def __init__(
        self,
        n_sim: int = N_SIM_PATHS,
        shrinkage_alpha: float = CORRELATION_SHRINKAGE_ALPHA,
        min_within_game_corr: float = MIN_WITHIN_GAME_CORR,
    ) -> None:
        self.n_sim = n_sim
        self.shrinkage_alpha = shrinkage_alpha
        self.min_within_game_corr = min_within_game_corr
        self._cov_matrix: Optional[np.ndarray] = None
        self._corr_matrix: Optional[np.ndarray] = None
        self._bet_ids: Optional[List[str]] = None


    def fit(
        self,
        bets: pd.DataFrame,
    ) -> "CovarianceEstimator":
        """Estimate the covariance matrix for the given slate of bets."""
        required = {"bet_id", "game_id", "true_prob"}
        missing = required - set(bets.columns)
        if missing:
            raise ValueError(f"bets DataFrame is missing columns: {missing}")

        self._bet_ids = bets["bet_id"].tolist()
        probs = bets["true_prob"].to_numpy(dtype=float)
        n = len(probs)

        logger.info(
            "Estimating covariance for %d bets via %d MC paths …", n, self.n_sim
        )

        rng = np.random.default_rng(seed=42)
        # Shape: (n_sim, n_bets) — each row is one hypothetical outcome vector
        sim_matrix = (rng.uniform(size=(self.n_sim, n)) < probs).astype(float)

        sample_cov = np.cov(sim_matrix.T)          # (n, n)
        if n == 1:
            sample_cov = sample_cov.reshape(1, 1)

        alpha = np.clip(self.shrinkage_alpha, 0.0, 1.0)
        diag_target = np.diag(np.diag(sample_cov))  # scaled identity proxy
        shrunk_cov = (1.0 - alpha) * sample_cov + alpha * diag_target

        game_ids = bets["game_id"].to_numpy()
        shrunk_cov = self._enforce_game_structure(
            shrunk_cov, game_ids, self.min_within_game_corr
        )

        self._cov_matrix = shrunk_cov

        std = np.sqrt(np.diag(shrunk_cov))
        std_outer = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._corr_matrix = np.where(
                std_outer > 0, shrunk_cov / std_outer, 0.0
            )
        np.fill_diagonal(self._corr_matrix, 1.0)

        logger.info("Covariance estimation complete.  Matrix shape: %s", shrunk_cov.shape)
        return self

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Estimated covariance matrix as a numpy array."""
        self._assert_fitted()
        return self._cov_matrix

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Derived correlation matrix."""
        self._assert_fitted()
        return self._corr_matrix

    def as_dataframe(self, kind: str = "correlation") -> pd.DataFrame:
        """Return the matrix as a labelled DataFrame."""
        self._assert_fitted()
        mat = self._corr_matrix if kind == "correlation" else self._cov_matrix
        return pd.DataFrame(mat, index=self._bet_ids, columns=self._bet_ids)

    def highly_correlated_pairs(
        self, threshold: float = 0.70
    ) -> List[Tuple[str, str, float]]:
        """Return all bet pairs with correlation >= ``threshold``."""
        self._assert_fitted()
        n = len(self._bet_ids)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._corr_matrix[i, j]
                if corr >= threshold:
                    pairs.append((self._bet_ids[i], self._bet_ids[j], float(corr)))
        pairs.sort(key=lambda x: -x[2])
        return pairs


    @staticmethod
    def _enforce_game_structure(
        cov: np.ndarray,
        game_ids: np.ndarray,
        min_corr: float,
    ) -> np.ndarray:
        """Boost within-game off-diagonal covariances so that the implied correlation is at
        least ``min_corr``.
        """
        cov = cov.copy()
        n = len(game_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if game_ids[i] == game_ids[j]:
                    # Implied correlation: cov_ij / sqrt(var_i * var_j)
                    denom = np.sqrt(abs(cov[i, i]) * abs(cov[j, j]))
                    if denom == 0:
                        continue
                    current_corr = cov[i, j] / denom
                    if current_corr < min_corr:
                        # Scale covariance up to hit the minimum
                        cov[i, j] = min_corr * denom
                        cov[j, i] = cov[i, j]
        return cov

    def _assert_fitted(self) -> None:
        if self._cov_matrix is None:
            raise RuntimeError(
                "CovarianceEstimator has not been fitted. Call .fit(bets) first."
            )
