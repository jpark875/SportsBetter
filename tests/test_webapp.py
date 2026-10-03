"""Store, risk metrics, parlay maths and the Flask routes."""

from __future__ import annotations

import math

import pytest

from core.parlay import ParlayLeg, analyze_parlay
from webapp.app import _chart_svg, create_app
from webapp.risk_metrics import bankroll_curve, compute_metrics
from webapp.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "app.db"))


class TestStore:
    def test_defaults_then_saved_profile(self, store):
        assert store.get_profile()["liquid_bankroll"] == 1000.0
        store.save_profile(2500.0, 300.0, 7)
        store.save_profile(2600.0, 310.0, 8)
        profile = store.get_profile()
        assert (profile["liquid_bankroll"], profile["volatility_tolerance"]) == (2600.0, 8)

    @pytest.mark.parametrize(
        ("price", "status", "pnl"),
        [(150, "won", 150.0), (-200, "won", 50.0), (150, "lost", -100.0), (150, "void", 0.0)],
    )
    def test_settlement_pnl(self, store, price, status, pnl):
        bet_id = store.add_bet("mlb", "A @ B", "B", price=price, stake=100.0)
        store.settle_bet(bet_id, status)
        assert store.list_bets()[0]["pnl"] == pytest.approx(pnl)

    def test_settle_validates(self, store):
        bet_id = store.add_bet("mlb", "A @ B", "B", price=110, stake=10.0)
        with pytest.raises(ValueError):
            store.settle_bet(bet_id, "maybe")
        with pytest.raises(KeyError):
            store.settle_bet(999, "won")

    def test_listing_filters_and_voids_are_excluded_from_metrics_input(self, store):
        won = store.add_bet("mlb", "m1", "s", price=100, stake=10.0)
        void = store.add_bet("mlb", "m2", "s", price=100, stake=10.0)
        store.add_bet("mlb", "m3", "s", price=100, stake=10.0)
        store.settle_bet(won, "won")
        store.settle_bet(void, "void")
        assert len(store.list_bets(status="pending")) == 1
        assert [b["id"] for b in store.settled_bets_chronological()] == [won]
        store.delete_bet(won)
        assert len(store.list_bets()) == 2


def settled(*pairs):
    return [
        {"stake": stake, "pnl": pnl, "status": "won" if pnl > 0 else "lost"}
        for stake, pnl in pairs
    ]


class TestRiskMetrics:
    def test_empty_ledger(self):
        metrics = compute_metrics([])
        assert metrics.n_bets == 0 and metrics.stability_grade == "N/A"

    def test_basic_ratios(self):
        metrics = compute_metrics(settled((100, 100), (100, -100), (100, 100), (100, -100)))
        assert metrics.win_rate == 0.5
        assert metrics.roi == 0.0
        assert metrics.stability_grade == "Provisional"
        assert metrics.max_drawdown_dollars == 100.0

    def test_all_winners_have_capped_infinite_ratios(self):
        metrics = compute_metrics(settled(*[(10, 10)] * 12))
        assert metrics.sortino == 99.99
        assert metrics.stability_grade == "Stable"

    def test_deep_drawdown_is_flagged_volatile(self):
        data = settled(*[(100, 100)] * 5, *[(100, -100)] * 8)
        metrics = compute_metrics(data)
        assert metrics.max_drawdown >= 0.3
        assert metrics.stability_grade == "Volatile"

    def test_losing_ledger_is_underwater(self):
        data = settled(*[(10, -10)] * 9, (10, 1))
        assert compute_metrics(data).stability_grade in {"Underwater", "Volatile"}

    def test_curve_starts_at_zero(self):
        curve = bankroll_curve(settled((10, 5), (10, -2)))
        assert [p["pnl"] for p in curve] == [0.0, 5.0, 3.0]

    def test_as_dict_round_trips(self):
        assert compute_metrics(settled((10, 5))).as_dict()["n_bets"] == 1


def leg(label, prob, odds, game="g"):
    return ParlayLeg(label=label, game_id=game, true_prob=prob, decimal_odds=odds)


class TestParlay:
    def test_needs_two_legs(self):
        with pytest.raises(ValueError):
            analyze_parlay([leg("a", 0.5, 2.0)])

    def test_negative_ev_is_a_pass(self):
        result = analyze_parlay([leg("a", 0.5, 1.8, "1"), leg("b", 0.5, 1.8, "2")])
        assert result.verdict == "pass" and result.edge < 0

    def test_same_game_legs_are_flagged(self):
        result = analyze_parlay([leg("a", 0.7, 2.0), leg("b", 0.7, 2.0)])
        assert result.correlated and result.verdict == "pass"

    def test_strong_independent_edges_prefer_straight_or_parlay(self):
        result = analyze_parlay([leg("a", 0.6, 2.0, "1"), leg("b", 0.6, 2.0, "2")])
        assert result.edge > 0
        assert result.verdict in {"straight", "parlay"}
        assert result.combined_decimal == pytest.approx(4.0)
        assert result.fair_decimal == pytest.approx(1 / 0.36)
        assert math.isclose(result.straight_total_edge, 0.4)


@pytest.fixture
def client(store):
    app = create_app(store)
    app.config.update(TESTING=True)
    return app.test_client()


class TestRoutes:
    @pytest.mark.parametrize("path", ["/", "/history", "/parlay", "/settings"])
    def test_pages_render(self, client, path):
        assert client.get(path).status_code == 200

    def test_unknown_page_and_sport_are_404(self, client):
        assert client.get("/nope").status_code == 404
        assert client.get("/slate/curling").status_code == 404

    def test_untrained_slate_renders_its_explanation(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("sports.base.MODEL_DIR", str(tmp_path))
        response = client.get("/slate/mlb")
        assert response.status_code == 200
        assert b"No trained model" in response.data

    def test_place_settle_and_delete_a_bet(self, client, store):
        form = {
            "sport": "mlb", "matchup": "A @ B", "selection": "B", "price": "150",
            "stake": "20", "bookmaker": "dk", "model_prob": "0.5", "edge": "0.1",
        }
        assert client.post("/bet/place", data=form).status_code == 302
        bet_id = store.list_bets()[0]["id"]
        client.post(f"/bet/{bet_id}/settle", data={"status": "won"})
        page = client.get("/history")
        assert b"A @ B" in page.data
        assert store.list_bets()[0]["pnl"] == pytest.approx(30.0)
        client.post(f"/bet/{bet_id}/delete")
        assert store.list_bets() == []

    def test_bad_bet_input_is_reported_not_raised(self, client, store):
        response = client.post("/bet/place", data={"sport": "mlb"}, follow_redirects=True)
        assert response.status_code == 200
        assert store.list_bets() == []
        assert client.post("/bet/1/settle", data={"status": "won"}).status_code == 302

    def test_settings_validation_and_save(self, client, store):
        bad = client.post(
            "/settings",
            data={"liquid_bankroll": "100", "disposable_income": "50", "volatility_tolerance": "11"},
        )
        assert b"tolerance must be 1-10" in bad.data
        ok = client.post(
            "/settings",
            data={"liquid_bankroll": "900", "disposable_income": "50", "volatility_tolerance": "4"},
        )
        assert ok.status_code == 302
        assert store.get_profile()["liquid_bankroll"] == 900.0

    def test_parlay_form_analyses_valid_legs(self, client):
        data = {
            "label": ["A ML", "B ML", ""],
            "prob": ["0.6", "0.6", ""],
            "decimal_odds": ["2.0", "2.0", ""],
            "game_id": ["1", "2", ""],
        }
        response = client.post("/parlay", data=data)
        assert response.status_code == 200
        assert b"straight" in response.data.lower() or b"parlay" in response.data.lower()

    def test_parlay_with_one_leg_asks_for_more(self, client):
        response = client.post(
            "/parlay", data={"label": ["A"], "prob": ["0.5"], "decimal_odds": ["2"]}
        )
        assert b"at least two" in response.data


def test_chart_needs_two_points_and_marks_direction():
    assert str(_chart_svg([{"n": 0, "pnl": 0.0}])) == ""
    up = str(_chart_svg([{"n": 0, "pnl": 0.0}, {"n": 1, "pnl": 5.0}]))
    down = str(_chart_svg([{"n": 0, "pnl": 0.0}, {"n": 1, "pnl": -5.0}]))
    assert "chart-pos" in up and "chart-neg" in down
