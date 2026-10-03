"""Integration tests: a toy sport trained from synthetic history, scored against fake odds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sports
from core.generic_features import (
    GENERIC_FEATURE_COLS,
    build_feature_table,
    build_scoring_row,
    latest_team_stats,
)
from core.scoring_engine import score_slate
from portfolio_manager.arbitrage import (
    compute_arb_stakes,
    scan_for_arbitrage,
    scan_h2h_arbitrage,
)
from predictive_model.model_trainer import ModelTrainer
from risk_bridge.risk_manager import RiskManager
from config.settings import UserRiskProfile
from sports.base import SportConfig, SportPlugin

TEAMS = ["Aces", "Bears", "Comets", "Dragons", "Eagles", "Foxes", "Giants", "Hawks"]


def toy_history(draws: bool, n_games: int = 700, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-1.0, 1.0, len(TEAMS))))
    rows = []
    start = pd.Timestamp("2022-01-01")
    for i in range(n_games):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        diff = strength[home] - strength[away] + 0.3
        home_score = max(0, round(rng.normal(2.0 + 0.5 * diff, 1.0)))
        away_score = max(0, round(rng.normal(2.0 - 0.5 * diff, 1.0)))
        if not draws and home_score == away_score:
            home_score += 1
        rows.append(
            {
                "date": start + pd.Timedelta(days=i // 2),
                "home_team": home,
                "away_team": away,
                "home_score": float(home_score),
                "away_score": float(away_score),
            }
        )
    return pd.DataFrame(rows)


def make_plugin(key: str, draws: bool) -> SportPlugin:
    class Toy(SportPlugin):
        config = SportConfig(
            key=key,
            display_name=f"Toy {key}",
            odds_sport_key=f"toy_{key}",
            has_draws=draws,
            elo_k=20.0,
            hfa_elo=50.0,
            default_score=2.0,
            score_label="goals",
        )

        def fetch_history(self) -> pd.DataFrame:
            return toy_history(draws)

    return Toy()


@pytest.fixture
def toy_sports(tmp_path, monkeypatch):
    plugins = {"toy2": make_plugin("toy2", False), "toy3": make_plugin("toy3", True)}
    monkeypatch.setattr(sports, "_PLUGIN_MODULES", [*sports._PLUGIN_MODULES, *plugins])
    monkeypatch.setattr(sports, "_cache", {**sports._cache, **plugins})
    monkeypatch.setattr("sports.base.MODEL_DIR", str(tmp_path))
    return plugins


def odds_rows(game_id, home, away, books, three_way=False):
    rows = []
    for book, (h, a, d) in books.items():
        outcomes = [(home, h), (away, a)] + ([("Draw", d)] if three_way else [])
        for name, price in outcomes:
            rows.append(
                {
                    "game_id": game_id,
                    "commence_time": pd.Timestamp("2030-01-01 20:00", tz="UTC"),
                    "home_team": home,
                    "away_team": away,
                    "bet_type": "moneyline",
                    "bookmaker": book,
                    "outcome_name": name,
                    "price": float(price),
                    "point": float("nan"),
                }
            )
    return rows


class FakeOdds:
    frame = pd.DataFrame()
    error: Exception | None = None

    def __init__(self, provider=None, sport_key=None):
        pass

    def get_game_odds(self, **kwargs):
        if self.error:
            raise self.error
        return self.frame

    def get_implied_probabilities(self, df):
        df = df.copy()
        df["implied_prob"] = 0.5
        return df


@pytest.fixture
def fake_odds(monkeypatch):
    monkeypatch.setattr("core.scoring_engine.OddsClient", FakeOdds)
    FakeOdds.frame = pd.DataFrame()
    FakeOdds.error = None
    return FakeOdds


def risk_manager() -> RiskManager:
    return RiskManager(UserRiskProfile(liquid_bankroll=5000, disposable_income=2000))


class TestFeatures:
    def test_features_have_no_lookahead_and_cover_every_column(self):
        history = toy_history(draws=False)
        table, elo = build_feature_table(history)
        assert set(GENERIC_FEATURE_COLS) <= set(table.columns)
        assert set(table["RESULT"]) <= {0, 2}
        # The first game of the data set cannot have seen any rolling history.
        assert pd.isna(table.iloc[0]["HOME_SCORE_FOR_LAST5"])
        assert max(elo, key=elo.get) in {"Giants", "Hawks"}

    def test_scoring_row_uses_state_and_falls_back_for_new_teams(self):
        table, elo = build_feature_table(toy_history(draws=False))
        stats = latest_team_stats(table)
        known = build_scoring_row("Hawks", "Aces", elo, stats, 2.0)
        unknown = build_scoring_row("Nobody", "Aces", elo, stats, 2.0)
        assert list(known) == GENERIC_FEATURE_COLS
        assert known["DELTA_ELO"] > 0
        assert unknown["HOME_ELO"] == 1500.0
        assert unknown["HOME_ATK_RATING"] == 2.0


class TestArbitrage:
    def test_two_way_arbitrage_is_found_and_stakes_lock_a_profit(self):
        rows = odds_rows("g1", "A", "B", {"bk1": (150, -200, 0), "bk2": (-200, 150, 0)})
        arbs = scan_h2h_arbitrage(pd.DataFrame(rows))
        assert len(arbs) == 1
        assert arbs.iloc[0]["profit_pct"] > 0
        legs = arbs.iloc[0]["legs"]
        assert sum(leg["stake_fraction"] for leg in legs) == pytest.approx(1.0)

    def test_three_way_scan_and_stakes_guarantee_equal_returns(self):
        rows = odds_rows(
            "g1", "A", "B", {"bk1": (+250, +300, +400), "bk2": (+200, +350, +350)}, three_way=True
        )
        arbs = scan_for_arbitrage(pd.DataFrame(rows))
        assert len(arbs) == 1
        stakes = compute_arb_stakes(arbs.iloc[0], budget=100.0)
        assert stakes["guaranteed_profit"] > 0
        total = stakes["home_stake"] + stakes["draw_stake"] + stakes["away_stake"]
        assert total == pytest.approx(100.0, abs=0.05)

    def test_efficient_market_has_no_arbitrage(self):
        rows = odds_rows("g1", "A", "B", {"bk1": (-110, -110, 0)})
        assert scan_h2h_arbitrage(pd.DataFrame(rows)).empty

    def test_empty_input(self):
        assert scan_h2h_arbitrage(pd.DataFrame()).empty


class TestTrainAndScore:
    @pytest.fixture
    def trained(self, toy_sports):
        from scripts.train_sport import train_sport

        train_sport("toy2")
        train_sport("toy3")
        return toy_sports

    def test_training_writes_loadable_artefacts(self, trained):
        plugin = trained["toy2"]
        assert plugin.is_trained()
        trainer = ModelTrainer.load(plugin.model_path)
        table, _ = build_feature_table(toy_history(draws=False))
        proba = trainer.predict_proba(table[GENERIC_FEATURE_COLS].dropna())
        assert ((proba > 0) & (proba < 1)).all()
        assert not trainer.feature_importances.empty

    def test_two_way_slate_is_scored_and_allocated(self, trained, fake_odds):
        rows = odds_rows(
            "g1", "Hawks", "Aces", {"bk1": (+400, -500, 0), "bk2": (+380, -450, 0)}
        )
        rows += odds_rows("g2", "Aces", "Hawks", {"bk1": (-110, -110, 0)})
        fake_odds.frame = pd.DataFrame(rows)

        result = score_slate("toy2", risk_manager())

        assert result.trained and result.n_games == 2
        assert set(result.bets["outcome_name"]) == {"Home", "Away"}
        assert result.bets["true_prob"].between(0, 1).all()
        # A strong home side at +400 is the planted edge.
        best = result.bets.sort_values("edge").iloc[-1]
        assert best["edge"] > 0
        assert not result.allocation.empty
        assert result.allocation["wager_amount"].sum() <= 2000

    def test_three_way_slate_includes_the_draw(self, trained, fake_odds):
        fake_odds.frame = pd.DataFrame(
            odds_rows(
                "g1", "Hawks", "Aces", {"bk1": (-150, 400, 300)}, three_way=True
            )
        )
        result = score_slate("toy3", risk_manager())
        assert set(result.bets["outcome_name"]) == {"Home", "Away", "Draw"}
        probs = result.bets.groupby("game_id")["true_prob"].sum()
        assert probs.iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_slate_reports_arbitrage(self, trained, fake_odds):
        fake_odds.frame = pd.DataFrame(
            odds_rows("g1", "Hawks", "Aces", {"bk1": (+150, -200, 0), "bk2": (-200, +150, 0)})
        )
        assert not score_slate("toy2", risk_manager()).arbitrage.empty

    def test_unaligned_team_names_explain_themselves(self, trained, fake_odds):
        rows = odds_rows("g1", "Hawks", "Aces", {"bk1": (-110, -110, 0)})
        frame = pd.DataFrame(rows)
        frame["outcome_name"] = "Somebody Else"
        fake_odds.frame = frame
        assert "could not be aligned" in score_slate("toy2", risk_manager()).message


class TestSlateEdgeCases:
    def test_untrained_sport_names_the_training_command(self, toy_sports):
        result = score_slate("toy2", risk_manager())
        assert not result.trained
        assert "train_sport.py --sport toy2" in result.message

    def test_provider_failure_becomes_a_message(self, toy_sports, fake_odds, tmp_path):
        from scripts.train_sport import train_sport

        train_sport("toy2")
        fake_odds.error = RuntimeError("quota exceeded")
        assert "quota exceeded" in score_slate("toy2", risk_manager()).message

    def test_no_games_on_the_board(self, toy_sports, fake_odds):
        from scripts.train_sport import train_sport

        train_sport("toy2")
        assert "No live" in score_slate("toy2", risk_manager()).message
