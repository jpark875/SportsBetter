"""
tests/test_pipeline.py

Unit and integration smoke-tests for the NBA Bet Portfolio Engine.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import UserRiskProfile
from portfolio_manager.covariance_estimator import CovarianceEstimator
from portfolio_manager.optimizer import PortfolioOptimizer
from predictive_model.probability_calibrator import IsotonicCalibrator
from risk_bridge.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bets() -> pd.DataFrame:
    """Minimal bets DataFrame covering three games."""
    return pd.DataFrame([
        # Game 1 — Lakers vs Celtics
        {"bet_id": "LAL_BOS_20250101__moneyline__LAL__nan",
         "game_id": "LAL_BOS_20250101", "bet_type": "moneyline",
         "outcome_name": "Los Angeles Lakers", "bookmaker": "draftkings",
         "price": -110.0, "point": float("nan"), "true_prob": 0.56},
        {"bet_id": "LAL_BOS_20250101__spread__LAL__-2.5",
         "game_id": "LAL_BOS_20250101", "bet_type": "spread",
         "outcome_name": "Los Angeles Lakers", "bookmaker": "fanduel",
         "price": -110.0, "point": -2.5, "true_prob": 0.53},
        # Game 2 — Warriors vs Suns
        {"bet_id": "GSW_PHX_20250101__moneyline__GSW__nan",
         "game_id": "GSW_PHX_20250101", "bet_type": "moneyline",
         "outcome_name": "Golden State Warriors", "bookmaker": "draftkings",
         "price": +135.0, "point": float("nan"), "true_prob": 0.47},
        # Game 3 — Heat vs Bucks
        {"bet_id": "MIA_MIL_20250101__moneyline__MIL__nan",
         "game_id": "MIA_MIL_20250101", "bet_type": "moneyline",
         "outcome_name": "Milwaukee Bucks", "bookmaker": "betmgm",
         "price": -120.0, "point": float("nan"), "true_prob": 0.60},
    ])


@pytest.fixture
def fitted_covariance(sample_bets) -> CovarianceEstimator:
    est = CovarianceEstimator(n_sim=1_000)
    est.fit(sample_bets)
    return est


# ---------------------------------------------------------------------------
# CovarianceEstimator tests
# ---------------------------------------------------------------------------

class TestCovarianceEstimator:
    def test_fit_produces_square_matrix(self, sample_bets):
        est = CovarianceEstimator(n_sim=500)
        est.fit(sample_bets)
        n = len(sample_bets)
        assert est.covariance_matrix.shape == (n, n)

    def test_correlation_diagonal_is_one(self, fitted_covariance):
        diag = np.diag(fitted_covariance.correlation_matrix)
        np.testing.assert_allclose(diag, 1.0, atol=1e-6)

    def test_within_game_min_correlation_enforced(self, fitted_covariance):
        """LAL ML and LAL spread share game_id → corr must be ≥ MIN_WITHIN_GAME_CORR."""
        df = fitted_covariance.as_dataframe(kind="correlation")
        lal_ml = "LAL_BOS_20250101__moneyline__LAL__nan"
        lal_sp = "LAL_BOS_20250101__spread__LAL__-2.5"
        corr = df.loc[lal_ml, lal_sp]
        assert corr >= fitted_covariance.min_within_game_corr - 1e-6

    def test_as_dataframe_shape(self, fitted_covariance, sample_bets):
        df = fitted_covariance.as_dataframe()
        assert df.shape == (len(sample_bets), len(sample_bets))

    def test_highly_correlated_pairs_returns_list(self, fitted_covariance):
        pairs = fitted_covariance.highly_correlated_pairs(threshold=0.3)
        assert isinstance(pairs, list)

    def test_unfitted_raises(self):
        est = CovarianceEstimator()
        with pytest.raises(RuntimeError):
            _ = est.covariance_matrix


# ---------------------------------------------------------------------------
# PortfolioOptimizer tests
# ---------------------------------------------------------------------------

class TestPortfolioOptimizer:
    def test_optimise_returns_dataframe(self, sample_bets, fitted_covariance):
        opt = PortfolioOptimizer()
        result = opt.optimise(sample_bets, fitted_covariance, bankroll=1_000.0)
        assert isinstance(result, pd.DataFrame)
        assert "wager_amount" in result.columns

    def test_total_wager_within_cap(self, sample_bets, fitted_covariance):
        max_frac = 0.20
        bankroll = 1_000.0
        opt = PortfolioOptimizer(max_portfolio_fraction=max_frac)
        result = opt.optimise(sample_bets, fitted_covariance, bankroll=bankroll)
        assert result["wager_amount"].sum() <= max_frac * bankroll + 1e-2

    def test_single_bet_within_cap(self, sample_bets, fitted_covariance):
        max_single = 0.05
        bankroll = 1_000.0
        opt = PortfolioOptimizer(max_single_bet_fraction=max_single)
        result = opt.optimise(sample_bets, fitted_covariance, bankroll=bankroll)
        assert result["wager_amount"].max() <= max_single * bankroll + 1e-2

    def test_negative_edge_bets_zeroed(self, fitted_covariance):
        bad_bets = pd.DataFrame([{
            "bet_id": "BAD__moneyline__TeamX__nan",
            "game_id": "BAD_GAME",
            "bet_type": "moneyline",
            "outcome_name": "TeamX",
            "bookmaker": "draftkings",
            "price": -200.0,
            "point": float("nan"),
            "true_prob": 0.30,   # heavy favourite priced wrong — negative edge
        }])
        # Need a fresh estimator for single-bet slate
        est = CovarianceEstimator(n_sim=500)
        est.fit(bad_bets)
        opt = PortfolioOptimizer()
        result = opt.optimise(bad_bets, est, bankroll=1_000.0)
        assert result["wager_amount"].iloc[0] == 0.0

    def test_expected_portfolio_edge_non_negative(self, sample_bets, fitted_covariance):
        opt = PortfolioOptimizer()
        opt.optimise(sample_bets, fitted_covariance, bankroll=1_000.0)
        assert opt.expected_portfolio_edge >= 0.0


# ---------------------------------------------------------------------------
# IsotonicCalibrator tests
# ---------------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.3, 0.7, 200)
        y = (rng.uniform(size=200) < raw).astype(int)
        cal = IsotonicCalibrator()
        cal.fit(raw, y)
        out = cal.transform(raw)
        assert out.shape == raw.shape

    def test_output_within_bounds(self):
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.0, 1.0, 100)
        y = (rng.uniform(size=100) < raw).astype(int)
        cal = IsotonicCalibrator(y_min=0.01, y_max=0.99)
        cal.fit(raw, y)
        out = cal.transform(raw)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_ece_is_float(self):
        rng = np.random.default_rng(2)
        raw = rng.uniform(0.3, 0.7, 300)
        y = (rng.uniform(size=300) < raw).astype(int)
        cal = IsotonicCalibrator()
        cal.fit(raw, y)
        ece = cal.ece(raw, y)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_unfitted_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_constraint_derivation(self):
        profile = UserRiskProfile(
            liquid_bankroll=5_000.0,
            disposable_income=500.0,
            volatility_tolerance=5,
        )
        rm = RiskManager(profile)
        port_frac, single_frac, lam, bankroll = rm.get_portfolio_constraints()
        assert 0 < port_frac <= 0.25
        assert 0 < single_frac <= 0.05
        assert lam > 0
        assert bankroll == 500.0  # min(5000, 500)

    def test_session_stop_loss(self):
        profile = UserRiskProfile(
            liquid_bankroll=1_000.0,
            disposable_income=1_000.0,
            volatility_tolerance=1,
        )
        rm = RiskManager(profile, session_stop_loss_pct=0.10)
        assert rm.is_session_live is True
        # Simulate a $150 loss on a $1000 bankroll (15% > 10% threshold)
        rm.record_result("bet_001", 150.0, "loss", -110)
        assert rm.is_session_live is False

    def test_apply_constraints_caps_wager(self):
        profile = UserRiskProfile(
            liquid_bankroll=1_000.0,
            disposable_income=1_000.0,
            volatility_tolerance=3,
        )
        rm = RiskManager(profile)
        df = pd.DataFrame([
            {"bet_id": "A", "wager_amount": 999.0, "alloc_fraction": 0.99},
        ])
        result = rm.apply_constraints_to_allocation(df)
        _, max_single, _, bankroll = rm.get_portfolio_constraints()
        assert result["wager_amount"].iloc[0] <= max_single * bankroll + 1e-2

    def test_user_risk_profile_effective_bankroll(self):
        profile = UserRiskProfile(
            liquid_bankroll=10_000.0,
            disposable_income=200.0,
            volatility_tolerance=7,
        )
        assert profile.effective_bankroll == 200.0

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError):
            UserRiskProfile(liquid_bankroll=1_000.0, disposable_income=500.0,
                            volatility_tolerance=11)
