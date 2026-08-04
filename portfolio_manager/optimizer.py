"""
portfolio_manager/optimizer.py

Capital allocation engine based on Modern Portfolio Theory (MPT) and
the fractional Kelly Criterion.

Optimisation problem (simplified)
----------------------------------
Given N wagers with calibrated probabilities p_i, American odds o_i,
and a covariance matrix Σ:

    maximise   E[log W]  ≈  Σ_i f_i * edge_i  −  (1/2) f^T Σ f
    subject to 0 ≤ f_i ≤ MAX_SINGLE_BET_FRACTION
               Σ f_i ≤ MAX_PORTFOLIO_KELLY_FRACTION

where  f_i  = fraction of bankroll to wager on bet i
       edge_i = p_i * decimal_odds_i − 1  (expected value per unit wagered)

The quadratic penalty term (f^T Σ f) naturally shrinks allocations to
correlated bets, implementing the risk-adjusted Kelly derivation.

The optimiser uses ``scipy.optimize.minimize`` with the SLSQP method
(Sequential Least-Squares Programming), which handles the linear
inequality constraints and bounds efficiently.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from config.settings import MAX_PORTFOLIO_KELLY_FRACTION, MAX_SINGLE_BET_FRACTION
from portfolio_manager.covariance_estimator import CovarianceEstimator

logger = logging.getLogger(__name__)

# Minimum edge required to include a bet in the portfolio (avoids rounding noise)
MIN_EDGE_THRESHOLD: float = 0.01


class PortfolioOptimizer:
    """
    Allocates capital across a slate of wagers using a correlation-aware,
    fractional Kelly objective.

    Parameters
    ----------
    max_portfolio_fraction : float
        Maximum total bankroll fraction to wager across all bets combined.
    max_single_bet_fraction : float
        Maximum bankroll fraction on any individual bet.
    risk_scaling : float
        Risk-aversion multiplier applied to the covariance penalty term.
        Higher values = more conservative allocation.  Default ``1.0``.
    """

    def __init__(
        self,
        max_portfolio_fraction: float = MAX_PORTFOLIO_KELLY_FRACTION,
        max_single_bet_fraction: float = MAX_SINGLE_BET_FRACTION,
        risk_scaling: float = 1.0,
    ) -> None:
        self.max_portfolio_fraction = max_portfolio_fraction
        self.max_single_bet_fraction = max_single_bet_fraction
        self.risk_scaling = risk_scaling
        self._result: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimise(
        self,
        bets: pd.DataFrame,
        cov_estimator: CovarianceEstimator,
        bankroll: float,
    ) -> pd.DataFrame:
        """
        Run the portfolio optimisation and return the allocation table.

        Parameters
        ----------
        bets : pd.DataFrame
            Must contain:

            * ``bet_id``      : str   — unique identifier
            * ``game_id``     : str
            * ``true_prob``   : float — calibrated P(win)
            * ``price``       : float — American odds
            * ``bet_type``    : str
            * ``outcome_name``: str
            * ``bookmaker``   : str

        cov_estimator : CovarianceEstimator
            Already-fitted estimator for this slate.
        bankroll : float
            Effective wagerable bankroll (from risk_bridge).

        Returns
        -------
        pd.DataFrame
            Input bets with additional columns:

            * ``edge``           — expected value per unit wagered
            * ``kelly_fraction`` — unconstrained full-Kelly fraction
            * ``alloc_fraction`` — portfolio-constrained allocation fraction
            * ``wager_amount``   — dollar amount to bet
        """
        bets = bets.copy()
        bets["edge"] = bets.apply(
            lambda r: self._compute_edge(r["true_prob"], r["price"]), axis=1
        )

        # Drop negative-edge bets
        viable = bets[bets["edge"] >= MIN_EDGE_THRESHOLD].reset_index(drop=True)
        if viable.empty:
            logger.warning("No positive-edge bets found in the slate.")
            self._result = bets.assign(
                kelly_fraction=0.0, alloc_fraction=0.0, wager_amount=0.0
            )
            return self._result

        logger.info(
            "Optimising allocation for %d / %d positive-edge bets …",
            len(viable), len(bets),
        )

        edges = viable["edge"].to_numpy(dtype=float)
        cov = cov_estimator.covariance_matrix  # shape (N_all, N_all)

        # Slice covariance to the viable subset
        all_ids = cov_estimator._bet_ids
        viable_idx = [all_ids.index(bid) for bid in viable["bet_id"]]
        cov_sub = cov[np.ix_(viable_idx, viable_idx)]

        # ------------------------------------------------------------------
        # Unconstrained Kelly fractions (single-bet approximation)
        # ------------------------------------------------------------------
        viable["kelly_fraction"] = viable.apply(
            lambda r: self._kelly_fraction(r["true_prob"], r["price"]), axis=1
        )

        # ------------------------------------------------------------------
        # Constrained optimisation
        # ------------------------------------------------------------------
        alloc = self._run_slsqp(edges, cov_sub)
        viable["alloc_fraction"] = alloc
        viable["wager_amount"] = (viable["alloc_fraction"] * bankroll).round(2)

        # Merge back into original bet table
        bets = bets.merge(
            viable[["bet_id", "kelly_fraction", "alloc_fraction", "wager_amount"]],
            on="bet_id",
            how="left",
        )
        for col in ("kelly_fraction", "alloc_fraction", "wager_amount"):
            bets[col] = bets[col].fillna(0.0)

        self._result = bets
        total_risk = viable["alloc_fraction"].sum()
        total_wager = viable["wager_amount"].sum()
        logger.info(
            "Allocation complete.  Total risk: %.1f%% of bankroll | $%.2f wagered.",
            total_risk * 100,
            total_wager,
        )
        return self._result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def allocation_summary(self) -> pd.DataFrame:
        """Return only bets with a non-zero allocation, sorted by wager amount."""
        if self._result is None:
            raise RuntimeError("Call .optimise() before accessing allocation_summary.")
        return (
            self._result[self._result["wager_amount"] > 0]
            .sort_values("wager_amount", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def expected_portfolio_edge(self) -> float:
        """Weighted-average expected value across allocated bets."""
        if self._result is None:
            raise RuntimeError("Call .optimise() first.")
        alloc = self._result[self._result["wager_amount"] > 0]
        if alloc.empty:
            return 0.0
        weights = alloc["alloc_fraction"] / alloc["alloc_fraction"].sum()
        return float((weights * alloc["edge"]).sum())

    # ------------------------------------------------------------------
    # SLSQP core
    # ------------------------------------------------------------------

    def _run_slsqp(self, edges: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Solve the quadratic Kelly objective with SLSQP.

        Objective (negated for minimisation)::

            -( Σ f_i * e_i  −  (λ/2) * f^T Σ f )

        Returns
        -------
        np.ndarray, shape (n,)
            Optimal allocation fractions.
        """
        n = len(edges)
        lam = self.risk_scaling

        def neg_objective(f: np.ndarray) -> float:
            return -(np.dot(edges, f) - 0.5 * lam * f @ cov @ f)

        def grad(f: np.ndarray) -> np.ndarray:
            return -(edges - lam * cov @ f)

        # Bounds: each fraction in [0, max_single]
        bounds = [(0.0, self.max_single_bet_fraction)] * n

        # Constraint: sum of fractions ≤ max_portfolio
        constraints = {
            "type": "ineq",
            "fun": lambda f: self.max_portfolio_fraction - np.sum(f),
            "jac": lambda f: -np.ones(n),
        }

        f0 = np.full(n, min(0.01, self.max_single_bet_fraction))
        result = minimize(
            neg_objective,
            f0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            logger.warning("SLSQP did not converge: %s", result.message)

        alloc = np.clip(result.x, 0.0, self.max_single_bet_fraction)
        return alloc

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_edge(p_win: float, american_odds: float) -> float:
        """
        Expected value per unit wagered.

        edge = p_win * decimal_payout − 1
        """
        if american_odds >= 100:
            decimal = american_odds / 100.0 + 1.0
        else:
            decimal = 100.0 / (-american_odds) + 1.0
        return p_win * decimal - 1.0

    @staticmethod
    def _kelly_fraction(p_win: float, american_odds: float) -> float:
        """
        Full-Kelly fraction: f* = (b*p - q) / b
        where b = decimal_payout − 1, q = 1 − p_win.
        """
        if american_odds >= 100:
            b = american_odds / 100.0
        else:
            b = 100.0 / (-american_odds)
        q = 1.0 - p_win
        return max(0.0, (b * p_win - q) / b)
