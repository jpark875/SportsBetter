"""
portfolio_manager/covariance_estimator.py

Estimates the covariance (correlation) matrix among the active wagers
in tonight's betting slate.

Why does covariance matter for sports bets?
-------------------------------------------
Wagers within the same game are highly correlated:
  - Lakers ML and Lakers −3.5 spread move together almost perfectly.
  - LeBron 25+ points Over and Lakers team totals share common variance.

Naïvely treating each bet as independent massively over-allocates capital
because the joint downside risk is far larger than the sum of individual
bet risks.  A correlation-aware Kelly or mean-variance framework corrects
for this by penalising portfolios that are over-concentrated in a single
game or player cluster.

Estimation strategy
-------------------
1. **Simulation-based**: draw N Monte Carlo paths from each bet's
   Bernoulli distribution, compute the sample correlation matrix.
2. **Ledoit-Wolf shrinkage**: shrink the noisy sample matrix toward the
   identity to improve conditioning, using the shrinkage coefficient
   ``config.settings.CORRELATION_SHRINKAGE_ALPHA``.
3. **Game-cluster override**: set within-game off-diagonal entries to a
   minimum of ``MIN_WITHIN_GAME_CORR`` regardless of simulation results,
   because we know structurally that outcomes within the same game are tied.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config.settings import CORRELATION_SHRINKAGE_ALPHA

logger = logging.getLogger(__name__)

MIN_WITHIN_GAME_CORR: float = 0.50   # structural lower bound for same-game pairs
N_SIM_PATHS: int = 10_000            # Monte Carlo paths for sample covariance


class CovarianceEstimator:
    """
    Estimates a regularised covariance / correlation matrix for a slate of bets.

    Parameters
    ----------
    n_sim : int
        Number of Monte Carlo simulation paths.  More paths = lower variance
        in the estimate, at the cost of runtime.
    shrinkage_alpha : float
        Ledoit-Wolf shrinkage coefficient applied *on top of* the simulation.
        0.0 = no shrinkage, 1.0 = identity matrix.
    min_within_game_corr : float
        Minimum correlation enforced between any two bets sharing a ``game_id``.
    """

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        bets: pd.DataFrame,
    ) -> "CovarianceEstimator":
        """
        Estimate the covariance matrix for the given slate of bets.

        Parameters
        ----------
        bets : pd.DataFrame
            Must contain at minimum:

            * ``bet_id``     : str  — unique identifier for each wager
            * ``game_id``    : str  — groups wagers within the same game
            * ``true_prob``  : float — calibrated P(win) for the wager

        Returns
        -------
        CovarianceEstimator
            Returns ``self``.
        """
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

        # ------------------------------------------------------------------
        # 1. Monte Carlo Bernoulli simulation
        # ------------------------------------------------------------------
        rng = np.random.default_rng(seed=42)
        # Shape: (n_sim, n_bets) — each row is one hypothetical outcome vector
        sim_matrix = (rng.uniform(size=(self.n_sim, n)) < probs).astype(float)

        # ------------------------------------------------------------------
        # 2. Sample covariance
        # ------------------------------------------------------------------
        sample_cov = np.cov(sim_matrix.T)          # (n, n)
        if n == 1:
            sample_cov = sample_cov.reshape(1, 1)

        # ------------------------------------------------------------------
        # 3. Ledoit-Wolf shrinkage toward scaled identity
        # ------------------------------------------------------------------
        alpha = np.clip(self.shrinkage_alpha, 0.0, 1.0)
        diag_target = np.diag(np.diag(sample_cov))  # scaled identity proxy
        shrunk_cov = (1.0 - alpha) * sample_cov + alpha * diag_target

        # ------------------------------------------------------------------
        # 4. Enforce within-game minimum correlation
        # ------------------------------------------------------------------
        game_ids = bets["game_id"].to_numpy()
        shrunk_cov = self._enforce_game_structure(
            shrunk_cov, game_ids, self.min_within_game_corr
        )

        self._cov_matrix = shrunk_cov

        # Derive correlation matrix
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
        """
        Return the matrix as a labelled DataFrame.

        Parameters
        ----------
        kind : str
            ``"correlation"`` (default) or ``"covariance"``.

        Returns
        -------
        pd.DataFrame
            Square DataFrame indexed and columned by ``bet_id``.
        """
        self._assert_fitted()
        mat = self._corr_matrix if kind == "correlation" else self._cov_matrix
        return pd.DataFrame(mat, index=self._bet_ids, columns=self._bet_ids)

    def highly_correlated_pairs(
        self, threshold: float = 0.70
    ) -> List[Tuple[str, str, float]]:
        """
        Return all bet pairs with correlation >= ``threshold``.

        Useful for flagging over-concentration before optimisation.

        Parameters
        ----------
        threshold : float

        Returns
        -------
        list[tuple[str, str, float]]
            Each tuple is ``(bet_id_a, bet_id_b, correlation)``.
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_game_structure(
        cov: np.ndarray,
        game_ids: np.ndarray,
        min_corr: float,
    ) -> np.ndarray:
        """
        Boost within-game off-diagonal covariances so that the implied
        correlation is at least ``min_corr``.
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
