"""
risk_bridge/risk_manager.py

Translates the user's personal financial parameters into hard portfolio
constraints that are passed directly to the optimiser.

Design philosophy
-----------------
The "risk bridge" exists because the portfolio optimiser alone does not
know *who* is using it.  A professional bettor with a $100,000 bankroll
and a volatility tolerance of 9 should receive very different constraints
than a casual bettor with $500 and a tolerance of 2, even if the
underlying odds are identical.

Three-layer constraint system
------------------------------
1. **Drawdown limit** — derived from ``volatility_tolerance``:
   tolerance 1 → max 5 % drawdown, 10 → 30 %.
2. **Wager ceiling** — individual bet cap tied to disposable income.
3. **Session stop-loss** — real-time trip-wire.  If accumulated losses in
   the current session exceed the session limit, ``RiskManager.is_session_live``
   returns ``False`` and the main loop should abort further betting.

The class also maintains a simple session ledger so P&L can be tracked
across multiple ``record_result`` calls within the same run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    MAX_PORTFOLIO_KELLY_FRACTION,
    MAX_SINGLE_BET_FRACTION,
    UserRiskProfile,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Applies user financial constraints to the portfolio optimiser parameters.

    Parameters
    ----------
    profile : UserRiskProfile
        User's financial and risk-tolerance configuration.
    session_stop_loss_pct : float, optional
        Fraction of effective bankroll that, when lost in a single session,
        triggers a hard stop.  Defaults to ``max_drawdown_pct * 0.5``.
    """

    def __init__(
        self,
        profile: UserRiskProfile,
        session_stop_loss_pct: Optional[float] = None,
    ) -> None:
        self.profile = profile
        self.session_stop_loss_pct = (
            session_stop_loss_pct
            if session_stop_loss_pct is not None
            else profile.max_drawdown_pct * 0.5
        )
        self._session_ledger: List[dict] = []
        self._session_start: datetime = datetime.now(timezone.utc)

        logger.info(
            "RiskManager initialised.  "
            "Bankroll: $%.2f | Tolerance: %d/10 | "
            "Max drawdown: %.1f%% | Session stop-loss: %.1f%%",
            profile.effective_bankroll,
            profile.volatility_tolerance,
            profile.max_drawdown_pct * 100,
            self.session_stop_loss_pct * 100,
        )

    # ------------------------------------------------------------------
    # Constraint derivation
    # ------------------------------------------------------------------

    def get_portfolio_constraints(self) -> Tuple[float, float, float, float]:
        """
        Derive the four key optimiser parameters from the user's risk profile.

        Returns
        -------
        tuple[float, float, float, float]
            ``(max_portfolio_fraction, max_single_bet_fraction,
               risk_scaling, effective_bankroll)``

            * ``max_portfolio_fraction`` — total fraction of bankroll to risk
            * ``max_single_bet_fraction`` — max fraction on any single bet
            * ``risk_scaling`` — lambda for the quadratic covariance penalty
            * ``effective_bankroll`` — dollar amount available for wagering
        """
        tol = self.profile.volatility_tolerance  # 1–10

        # Scale max portfolio exposure linearly with tolerance
        # tol 1 → 5 %, tol 10 → 25 %
        max_port_frac = 0.05 + (tol - 1) * (0.20 / 9)
        max_port_frac = min(max_port_frac, MAX_PORTFOLIO_KELLY_FRACTION)

        # Single bet cap scales from 1 % to 5 %
        max_single = 0.01 + (tol - 1) * (0.04 / 9)
        max_single = min(max_single, MAX_SINGLE_BET_FRACTION)

        # Risk scaling: high tolerance → lower penalty (more aggressive Kelly)
        # tol 1 → lambda 3.0, tol 10 → lambda 0.5
        risk_scaling = 3.0 - (tol - 1) * (2.5 / 9)

        bankroll = self.profile.effective_bankroll

        logger.debug(
            "Constraints — port: %.2f%%  single: %.2f%%  lambda: %.2f  bankroll: $%.2f",
            max_port_frac * 100,
            max_single * 100,
            risk_scaling,
            bankroll,
        )
        return max_port_frac, max_single, risk_scaling, bankroll

    def apply_constraints_to_allocation(
        self, allocation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Post-process an allocation DataFrame to enforce absolute dollar caps.

        Specifically:
        * Each individual ``wager_amount`` is capped at
          ``max_single_bet_fraction * effective_bankroll``.
        * The total ``wager_amount`` is capped at
          ``max_portfolio_fraction * effective_bankroll``.
        * Any bet whose wager drops below $1.00 is zeroed out.

        Parameters
        ----------
        allocation_df : pd.DataFrame
            Output of :meth:`~portfolio_manager.optimizer.PortfolioOptimizer.optimise`.

        Returns
        -------
        pd.DataFrame
            Modified allocation table.
        """
        max_port_frac, max_single, _, bankroll = self.get_portfolio_constraints()
        df = allocation_df.copy()

        single_cap = max_single * bankroll
        df["wager_amount"] = df["wager_amount"].clip(upper=single_cap)

        total = df["wager_amount"].sum()
        port_cap = max_port_frac * bankroll
        if total > port_cap and total > 0:
            scale = port_cap / total
            df["wager_amount"] = (df["wager_amount"] * scale).round(2)

        # Re-derive fractions
        df["alloc_fraction"] = df["wager_amount"] / bankroll if bankroll > 0 else 0.0

        # Zero out bets below minimum threshold
        df.loc[df["wager_amount"] < 1.0, ["wager_amount", "alloc_fraction"]] = 0.0

        return df

    # ------------------------------------------------------------------
    # Session ledger
    # ------------------------------------------------------------------

    def record_result(
        self,
        bet_id: str,
        wager_amount: float,
        outcome: str,
        price: float,
    ) -> float:
        """
        Record the outcome of a completed bet and return the P&L.

        Parameters
        ----------
        bet_id : str
        wager_amount : float
            Amount wagered in dollars.
        outcome : str
            ``"win"`` or ``"loss"``.
        price : float
            American odds at the time of the wager.

        Returns
        -------
        float
            Net profit/loss for this bet.
        """
        if outcome.lower() == "win":
            if price >= 100:
                pnl = wager_amount * (price / 100.0)
            else:
                pnl = wager_amount * (100.0 / (-price))
        else:
            pnl = -wager_amount

        self._session_ledger.append({
            "bet_id": bet_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wager_amount": wager_amount,
            "outcome": outcome,
            "price": price,
            "pnl": pnl,
        })
        logger.info(
            "Bet %s → %s | PnL: $%+.2f | Session total: $%+.2f",
            bet_id, outcome, pnl, self.session_pnl,
        )
        return pnl

    @property
    def session_pnl(self) -> float:
        """Total session profit / loss in dollars."""
        return sum(r["pnl"] for r in self._session_ledger)

    @property
    def is_session_live(self) -> bool:
        """
        ``True`` if the session is still within acceptable loss limits.

        Returns ``False`` once cumulative losses exceed the session stop-loss
        threshold, signalling that the main loop should halt.
        """
        bankroll = self.profile.effective_bankroll
        if bankroll <= 0:
            return False
        loss_pct = -self.session_pnl / bankroll
        if loss_pct >= self.session_stop_loss_pct:
            logger.warning(
                "SESSION STOP-LOSS TRIGGERED: cumulative loss %.1f%% >= threshold %.1f%%",
                loss_pct * 100,
                self.session_stop_loss_pct * 100,
            )
            return False
        return True

    def session_report(self) -> pd.DataFrame:
        """
        Return the full session ledger as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``bet_id, timestamp, wager_amount, outcome, price, pnl``.
        """
        if not self._session_ledger:
            return pd.DataFrame(columns=[
                "bet_id", "timestamp", "wager_amount", "outcome", "price", "pnl"
            ])
        df = pd.DataFrame(self._session_ledger)
        df["cumulative_pnl"] = df["pnl"].cumsum()
        return df

    def print_summary(self) -> None:
        """Log a formatted summary of the current session to stdout."""
        pnl = self.session_pnl
        n_bets = len(self._session_ledger)
        wins = sum(1 for r in self._session_ledger if r["outcome"] == "win")
        bankroll = self.profile.effective_bankroll

        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("  Bets placed : %d  (W: %d / L: %d)", n_bets, wins, n_bets - wins)
        logger.info("  Net P&L     : $%+.2f  (%.1f%%)", pnl, (pnl / bankroll * 100) if bankroll else 0)
        logger.info("  Session live: %s", self.is_session_live)
        logger.info("=" * 60)
