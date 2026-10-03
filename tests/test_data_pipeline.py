"""Data clients, feature engineering and sport-history fetchers, all against stubbed I/O."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from data_pipeline import football_stats_client as fb
from data_pipeline import nba_stats_client as nba
from data_pipeline import odds_client as odds
from predictive_model.feature_engineering import FeatureEngineer, resolve_team_id

# --------------------------------------------------------------------------- odds


EVENT = {
    "id": "evt1",
    "home_team": "Boston Celtics",
    "away_team": "Miami Heat",
    "commence_time": "2030-01-02T01:00:00Z",
    "bookmakers": [
        {
            "key": "dk",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -150},
                        {"name": "Miami Heat", "price": 130},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": -3.5},
                        {"name": "Miami Heat", "price": -110, "point": 3.5},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": -110, "point": 215.5},
                        {"name": "Under", "price": -110, "point": 215.5},
                    ],
                },
            ],
        },
        {
            "key": "fd",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -140},
                        {"name": "Miami Heat", "price": 125},
                    ],
                }
            ],
        },
    ],
}


@pytest.fixture
def api(monkeypatch):
    calls = []
    responses: dict[str, object] = {}

    def fake_get(url, params):
        calls.append((url, params))
        for suffix, payload in responses.items():
            if url.endswith(suffix):
                return payload
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(odds, "_get", fake_get)
    return SimpleNamespace(calls=calls, responses=responses)


class TestOddsHelpers:
    def test_implied_probability_and_game_id(self):
        assert odds._american_to_implied(100) == pytest.approx(0.5)
        assert odds._american_to_implied(-200) == pytest.approx(2 / 3)
        assert odds._build_game_id("Boston Celtics", "Miami Heat", "2030-01-02T01:00:00Z") == (
            "BOSTON_MIAMIH_20300102"
        )

    def test_get_raises_on_http_errors(self, monkeypatch):
        class Resp:
            def raise_for_status(self):
                raise requests.HTTPError("boom")

        monkeypatch.setattr(odds._SESSION, "get", lambda *a, **k: Resp())
        with pytest.raises(requests.HTTPError):
            odds._get("https://x.invalid", {})


class TestTheOddsApi:
    def test_game_odds_are_normalised(self, api):
        api.responses["/odds"] = [EVENT]
        client = odds.OddsClient(provider="theodds", sport_key="basketball_nba")

        frame = client.get_game_odds()

        assert set(frame["bet_type"]) == {"moneyline", "spread", "total"}
        assert set(frame["bookmaker"]) == {"dk", "fd"}
        assert frame["game_id"].nunique() == 1
        assert api.calls[0][1]["markets"] == "h2h,spreads,totals"

    def test_bookmaker_filter_is_forwarded(self, api):
        api.responses["/odds"] = []
        odds.OddsClient(provider="theodds").get_game_odds(markets=["h2h"], bookmakers=["dk"])
        assert api.calls[0][1]["bookmakers"] == "dk"

    def test_player_props_and_event_ids(self, api):
        api.responses["/events/evt1/odds"] = {
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "commence_time": "2030-01-02T01:00:00Z",
            "bookmakers": [
                {
                    "key": "dk",
                    "markets": [
                        {
                            "key": "player_points",
                            "outcomes": [
                                {"name": "Over", "description": "Jayson Tatum", "price": -115,
                                 "point": 27.5}
                            ],
                        }
                    ],
                }
            ],
        }
        api.responses["/events"] = [
            {"id": "evt1", "home_team": "A", "away_team": "B",
             "commence_time": "2030-01-02T01:00:00Z"}
        ]
        client = odds.OddsClient(provider="theodds")

        props = client.get_player_props(event_id="evt1")
        events = odds._TheOddsAdapter("k", "https://x.invalid").fetch_event_ids()

        assert props.iloc[0]["player_name"] == "Jayson Tatum"
        assert props.iloc[0]["prop_stat"] == "points"
        assert list(events["event_id"]) == ["evt1"]


class TestOddsJam:
    def test_game_odds_and_props(self, api):
        api.responses["/game-odds"] = {
            "data": [
                {
                    "home_team": "Boston Celtics",
                    "away_team": "Miami Heat",
                    "start_time": "2030-01-02T01:00:00Z",
                    "odds": [
                        {"market_name": "Moneyline", "sportsbook": "dk", "selection": "Boston",
                         "price": -150},
                        {"market_name": "Point Spread", "sportsbook": "dk", "selection": "Boston",
                         "price": -110, "handicap": -3.5},
                        {"market_name": "Total Points", "sportsbook": "dk", "selection": "Over",
                         "price": -110},
                        {"market_name": "Something", "sportsbook": "dk", "selection": "x",
                         "price": 100},
                    ],
                }
            ]
        }
        api.responses["/player-props"] = {
            "data": [
                {"home_team": "A", "away_team": "B", "start_time": "2030-01-02T01:00:00Z",
                 "stat": "points", "sportsbook": "dk", "selection": "Over", "price": -110,
                 "line": 20.5, "player_name": "P"}
            ]
        }
        client = odds.OddsClient(provider="oddsjam")

        games = client.get_game_odds()
        props = client.get_player_props()

        assert list(games["bet_type"]) == ["moneyline", "spread", "total", "other"]
        assert props.iloc[0]["bet_type"] == "player_prop"


class TestOddsFacade:
    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            odds.OddsClient(provider="nope")

    def test_best_lines_and_implied_probabilities(self, api):
        api.responses["/odds"] = [EVENT]
        client = odds.OddsClient(provider="theodds")
        frame = client.get_game_odds(markets=["h2h"])

        best = client.get_best_lines(frame[frame["bet_type"] == "moneyline"])
        home = best[best["outcome_name"] == "Boston Celtics"].iloc[0]
        assert home["price"] == -140 and home["bookmaker"] == "fd"

        with_prob = client.get_implied_probabilities(frame)
        assert with_prob["implied_prob"].between(0, 1).all()
        assert client.get_best_lines(pd.DataFrame()).empty
        assert client.get_implied_probabilities(pd.DataFrame()).empty


# --------------------------------------------------------------------- nba stats


class FakeEndpoint:
    frames: list[pd.DataFrame] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_data_frames(self):
        return [frame.copy() for frame in self.frames]


def endpoint(frame: pd.DataFrame) -> type[FakeEndpoint]:
    return type("Endpoint", (FakeEndpoint,), {"frames": [frame]})


@pytest.fixture(autouse=False)
def fast_nba(monkeypatch):
    monkeypatch.setattr(nba._throttle, "wait", lambda: None)
    monkeypatch.setattr(nba.time, "sleep", lambda s: None)


class TestNbaClient:
    def test_throttle_sleeps_only_when_called_too_soon(self, monkeypatch):
        slept = []
        monkeypatch.setattr(nba.time, "sleep", slept.append)
        throttle = nba._ThrottledCaller(calls_per_minute=60)
        throttle.wait()
        throttle.wait()
        assert len(slept) == 1 and slept[0] <= 1.0

    def test_retry_recovers_from_timeouts_and_reraises_others(self, fast_nba):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise requests.exceptions.Timeout("slow")
            return "ok"

        assert nba._with_retry(flaky) == "ok"
        with pytest.raises(ValueError):
            nba._with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")))
        with pytest.raises(requests.exceptions.ConnectionError):
            nba._with_retry(lambda: (_ for _ in ()).throw(requests.exceptions.ConnectionError()))

    def test_team_and_player_tables(self, monkeypatch, fast_nba):
        teams = pd.DataFrame(
            {"TEAM_ID": [1610612738.0, 1610612748.0], "TEAM_ABBREVIATION": ["BOS", "MIA"],
             "GP": [10, 10], "PACE": [99.0, 97.0], "OFF_RATING": [118.0, 112.0],
             "DEF_RATING": [108.0, 110.0], "NET_RATING": [10.0, 2.0]}
        )
        players = pd.DataFrame(
            {"player_id": [1.0, 2.0], "player_name": ["A", "B"], "usg_pct": [0.3, 0.2]}
        )
        monkeypatch.setattr(nba, "LeagueDashTeamStats", endpoint(teams))
        monkeypatch.setattr(nba, "LeagueDashPlayerStats", endpoint(players))
        client = nba.NBAStatsClient()

        team_df = client.fetch_team_advanced_stats()
        player_df = client.fetch_player_advanced_stats()

        assert list(team_df.index) == [1610612738, 1610612748]
        assert list(player_df.index) == [1, 2]

    def test_player_game_log(self, monkeypatch, fast_nba):
        monkeypatch.setattr(
            nba, "PlayerDashboardByGeneralSplits", endpoint(pd.DataFrame({"pts": [10]}))
        )
        assert list(nba.NBAStatsClient().fetch_player_game_log(7).columns) == ["PTS"]

    def _game_log(self, team_id):
        dates = ["2030-01-01", "2030-01-02", "2030-01-05"]
        return pd.DataFrame(
            {
                "TEAM_ID": [team_id] * 3,
                "GAME_ID": ["g1", "g2", "g3"],
                "GAME_DATE": dates,
                "MATCHUP": ["BOS vs. MIA", "BOS @ MIA", "BOS vs. NYK"],
            }
        )

    def test_schedule_features_flag_back_to_backs(self, monkeypatch, fast_nba):
        monkeypatch.setattr(nba, "TeamGameLog", endpoint(self._game_log(1610612738)))
        client = nba.NBAStatsClient()

        sched = client.fetch_schedule_features(["BOS", "ZZZ"])

        rows = sched.reset_index()
        assert list(rows["REST_DAYS"]) == [7, 0, 2]
        assert rows["HOME"].tolist() == [1, 0, 1]
        assert rows["IS_B2B"].tolist() == [0, 0, 0]
        assert rows["IS_B2B_FIRST"].tolist() == [0, 0, 0]

    def test_schedule_features_skip_failures(self, monkeypatch, fast_nba):
        class Broken(FakeEndpoint):
            def get_data_frames(self):
                raise RuntimeError("down")

        monkeypatch.setattr(nba, "TeamGameLog", Broken)
        assert nba.NBAStatsClient().fetch_schedule_features(["BOS"]).empty

    def test_fast_schedule_and_load_all(self, monkeypatch, fast_nba):
        log = pd.concat([self._game_log(1), self._game_log(2)], ignore_index=True)
        monkeypatch.setattr(nba, "LeagueGameLog", endpoint(log))
        monkeypatch.setattr(
            nba, "LeagueDashTeamStats",
            endpoint(pd.DataFrame({"TEAM_ID": [1], "PACE": [1.0]})),
        )
        monkeypatch.setattr(
            nba, "LeagueDashPlayerStats",
            endpoint(pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["A"]})),
        )

        players, teams, sched = nba.NBAStatsClient().load_all()

        assert sched.index.names == ["TEAM_ID", "GAME_ID"]
        assert len(sched) == 6 and len(players) == 1 and len(teams) == 1


# ----------------------------------------------------------------- feature table


def _team_ids():
    return resolve_team_id("Boston Celtics"), resolve_team_id("Miami Heat")


class TestFeatureEngineering:
    def test_team_name_resolution(self):
        assert resolve_team_id("BOS") == resolve_team_id("Boston Celtics")
        assert resolve_team_id(" celtics ") == resolve_team_id("BOS")
        assert resolve_team_id("Nowhere Nobodies") is None

    @pytest.fixture
    def engineer(self):
        bos, mia = _team_ids()
        stats = pd.DataFrame(
            {f: [x, y] for f, (x, y) in {
                "PACE": (99.0, 97.0), "OFF_RATING": (118.0, 112.0), "DEF_RATING": (108.0, 110.0),
                "NET_RATING": (10.0, 2.0), "AST_PCT": (0.6, 0.5), "OREB_PCT": (0.25, 0.24),
                "DREB_PCT": (0.75, 0.74), "EFG_PCT": (0.56, 0.53), "TS_PCT": (0.6, 0.57),
            }.items()},
            index=pd.Index([bos, mia], name="TEAM_ID"),
        )
        sched = pd.DataFrame(
            {
                "TEAM_ID": [bos, mia],
                "GAME_ID": ["a", "b"],
                "GAME_DATE": [pd.Timestamp("2030-01-02")] * 2,
                "REST_DAYS": [2, 0],
                "IS_B2B": [0, 1],
                "IS_B2B_FIRST": [0, 0],
                "HOME": [1, 0],
            }
        ).set_index(["TEAM_ID", "GAME_ID"])
        players = pd.DataFrame(
            {"PLAYER_NAME": ["Jayson Tatum"], "USG_PCT": [0.3], "TS_PCT": [0.6],
             "NET_RATING": [9.0], "PIE": [0.15]},
            index=pd.Index([1], name="PLAYER_ID"),
        )
        return FeatureEngineer(stats, sched, players)

    @pytest.fixture
    def odds_frame(self, api):
        api.responses["/odds"] = [EVENT]
        frame = odds.OddsClient(provider="theodds").get_game_odds()
        # 01:00 UTC on the 3rd is the evening of the 2nd in US/Eastern.
        frame["commence_time"] = pd.Timestamp("2030-01-03 01:00", tz="UTC")
        return frame

    def test_game_features_merge_stats_schedule_and_market(self, engineer, odds_frame):
        features = engineer.build_game_features(odds_frame)

        row = features.iloc[0]
        assert row["DELTA_NET_RATING"] == 8.0
        assert row["HOME_REST_DAYS"] == 2 and row["AWAY_IS_B2B"] == 1
        assert row["CONSENSUS_TOTAL"] == 215.5
        assert row["CONSENSUS_SPREAD"] == -3.5
        assert 0.5 < row["ML_IMPLIED_HOME_PROB"] < 0.7
        assert np.isnan(row["HOME_WIN"])

    def test_unresolvable_teams_are_dropped_and_empty_inputs_degrade(self, odds_frame):
        blank = FeatureEngineer(pd.DataFrame(), pd.DataFrame())
        frame = odds_frame.copy()
        frame.loc[frame.index[:3], "home_team"] = "Nowhere Nobodies"

        features = blank.build_game_features(odds_frame)

        assert features["DELTA_PACE"].isna().all()
        assert features["HOME_REST_DAYS"].isna().all()
        assert blank.build_game_features(frame).empty

    def test_prop_features(self, engineer, odds_frame):
        props = pd.DataFrame(
            {
                "game_id": ["g"] * 2,
                "bet_type": ["player_prop"] * 2,
                "player_name": ["Jayson Tatum"] * 2,
                "prop_stat": ["points"] * 2,
                "outcome_name": ["Over", "Under"],
                "price": [-115.0, -105.0],
            }
        )
        result = engineer.build_prop_features(props)
        assert result.loc[("g", "Jayson Tatum", "points", "Over"), "USG_PCT"] == 0.3
        assert result["PROP_IMPLIED_OVER_PROB"].notna().sum() == 1
        assert engineer.build_prop_features(odds_frame).empty
        with pytest.raises(ValueError):
            FeatureEngineer(pd.DataFrame(), pd.DataFrame()).build_prop_features(props)


# ------------------------------------------------------------------ football data


def football_csvs():
    rng = np.random.default_rng(1)
    teams = ["Brazil", "Argentina", "Germany", "France", "USA", "Korea Republic"]
    tournaments = ["FIFA World Cup", "Friendly", "UEFA Euro qualification", "Copa America"]
    rows, scorers = [], []
    start = pd.Timestamp("2005-01-01")
    for i in range(260):
        home, away = rng.choice(teams, size=2, replace=False)
        hs, as_ = int(rng.poisson(1.6)), int(rng.poisson(1.1))
        date = start + pd.Timedelta(days=20 * i)
        rows.append(
            {"date": date.date(), "home_team": home, "away_team": away, "home_score": hs,
             "away_score": as_, "tournament": tournaments[i % 4], "city": "x", "country": "y",
             "neutral": bool(i % 2)}
        )
        for _ in range(hs):
            scorers.append(
                {"date": date.date(), "home_team": home, "away_team": away, "team": home,
                 "scorer": f"{home} {rng.integers(1, 4)}", "minute": 10, "own_goal": False,
                 "penalty": False}
            )
    rows.append(
        {"date": "2030-01-01", "home_team": "Brazil", "away_team": "France", "home_score": None,
         "away_score": None, "tournament": "FIFA World Cup", "city": "x", "country": "y",
         "neutral": True}
    )
    return pd.DataFrame(rows).to_csv(index=False), pd.DataFrame(scorers).to_csv(index=False)


@pytest.fixture
def football_http(monkeypatch):
    results, goals = football_csvs()

    def fake_get(url, timeout):
        return SimpleNamespace(
            text=goals if "goalscorers" in url else results, raise_for_status=lambda: None
        )

    monkeypatch.setattr(fb.requests, "get", fake_get)


class TestFootballClient:
    def test_name_and_k_factor_helpers(self):
        assert fb.normalize_name(" USA ") == "United States"
        assert fb.normalize_name("Brazil") == "Brazil"
        assert fb._k_factor("FIFA World Cup") == 60
        assert fb._k_factor("Friendly") == 10
        assert fb._k_factor("Minor Cup") == 30
        assert fb._elo_expected(1500, 1500) == 0.5

    def test_training_features_pipeline(self, football_http, tmp_path):
        out = tmp_path / "out" / "train.csv"

        frame = fb.build_training_features(
            "https://x/results.csv", "https://x/goalscorers.csv", min_year=2005,
            output_path=str(out),
        )

        assert out.exists()
        assert (~frame["tournament"].str.lower().str.contains("friendly")).all()
        for col in ("HOME_ELO", "HOME_FORM_LAST5", "HOME_ATK_RATING", "HOME_STAR_CONC"):
            assert col in frame.columns
        assert set(frame["RESULT"]) <= {0, 1, 2}
        assert "Korea Republic" not in set(frame["home_team"])
        assert "South Korea" in set(frame["home_team"]) | set(frame["away_team"])

    def test_competitive_filter_and_current_state(self, football_http):
        frame = fb.build_training_features(
            "https://x/results.csv", "https://x/goalscorers.csv", min_year=2005,
            filter_wc_and_competitive=True,
        )
        assert frame["tournament"].str.lower().str.contains("world cup|euro|copa").all()

        elo, stats = fb.get_current_state(frame)
        row = fb.build_scoring_row("Brazil", "Nobody", elo, stats, is_neutral=False)
        assert elo["Brazil"] != 1500.0
        assert row["IS_NEUTRAL"] == 0 and row["AWAY_ELO"] == 1500.0
        assert row["AWAY_STAR_CONC"] == 0.35
        assert set(row) >= set(
            ["HOME_ELO", "DELTA_ELO", "HOME_ATK_VS_AWAY_DEF", "IS_NEUTRAL"]
        )


# ----------------------------------------------------------------- sport history


class TestHistoryFetchers:
    def test_mlb_keeps_final_scored_games_only(self, monkeypatch):
        from sports import mlb

        def team(name, score=None):
            side = {"team": {"name": name}}
            if score is not None:
                side["score"] = score
            return side

        payload = {
            "dates": [
                {
                    "date": "2024-04-01",
                    "games": [
                        {"status": {"abstractGameState": "Final"},
                         "teams": {"home": team("A", 5), "away": team("B", 3)}},
                        {"status": {"abstractGameState": "Preview"},
                         "teams": {"home": team("A"), "away": team("B")}},
                        {"status": {"abstractGameState": "Final"},
                         "teams": {"home": team("A", 2), "away": team("B", 2)}},
                        {"status": {"abstractGameState": "Final"},
                         "teams": {"home": team("A"), "away": team("B")}},
                    ],
                }
            ]
        }
        monkeypatch.setattr(mlb, "_FIRST_SEASON", mlb.date.today().year)
        monkeypatch.setattr(
            mlb.requests, "get",
            lambda *a, **k: SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None),
        )

        games = mlb.PLUGIN.fetch_history()

        assert len(games) == 1 and games.iloc[0]["home_score"] == 5.0

    def test_epl_downloads_each_season_and_normalises_names(self, monkeypatch):
        from sports import epl

        csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n13/08/2022,Man United,Arsenal,2,1\n"

        def fake_get(url, timeout):
            if "1011" in url:
                raise requests.ConnectionError("gone")
            if "1112" in url:
                return SimpleNamespace(text="Date,Bad\n1,2\n", raise_for_status=lambda: None)
            return SimpleNamespace(text=csv, raise_for_status=lambda: None)

        monkeypatch.setattr(epl.requests, "get", fake_get)

        games = epl.PLUGIN.fetch_history()

        assert set(games["home_team"]) == {"Man United"}
        assert games["date"].iloc[0] == pd.Timestamp("2022-08-13")
        assert epl.PLUGIN.normalize_name("Manchester United") == "Man United"
        assert epl.PLUGIN.normalize_name(" Arsenal ") == "Arsenal"

    def test_basketball_pairs_team_rows_into_games(self, monkeypatch):
        from sports import _basketball

        log = pd.DataFrame(
            {
                "GAME_ID": ["1", "1", "2", "2"],
                "GAME_DATE": ["2024-01-02"] * 2 + ["2024-01-03"] * 2,
                "TEAM_NAME": ["Home A", "Away B", "Home C", "Away D"],
                "MATCHUP": ["HA vs. AB", "AB @ HA", "HC vs. AD", "AD @ HC"],
                "PTS": [100, 90, 80, 80],
            }
        )

        class FakeLog:
            def __init__(self, **kwargs):
                pass

            def get_data_frames(self):
                return [log]

        import nba_api.stats.endpoints as endpoints

        monkeypatch.setattr(endpoints, "leaguegamelog", SimpleNamespace(LeagueGameLog=FakeLog))
        monkeypatch.setattr(_basketball, "_CALL_GAP_SECONDS", 0)

        games = _basketball.fetch_basketball_history(["2023-24"], "00", include_playoffs=False)

        assert len(games) == 1  # the tied game is dropped
        assert games.iloc[0]["home_team"] == "Home A"

    def test_basketball_without_any_logs_raises(self, monkeypatch):
        from sports import _basketball

        class Broken:
            def __init__(self, **kwargs):
                raise RuntimeError("blocked")

        import nba_api.stats.endpoints as endpoints

        monkeypatch.setattr(endpoints, "leaguegamelog", SimpleNamespace(LeagueGameLog=Broken))
        monkeypatch.setattr(_basketball, "_CALL_GAP_SECONDS", 0)
        with pytest.raises(RuntimeError, match="No basketball"):
            _basketball.fetch_basketball_history(["2023-24"], "10")

    def test_nba_and_wnba_plugins_pick_their_league(self, monkeypatch):
        from sports import _basketball, nba, wnba

        seen = []
        monkeypatch.setattr(
            _basketball, "fetch_basketball_history",
            lambda seasons, league_id, **kw: seen.append((seasons[0], league_id)) or pd.DataFrame(),
        )
        nba.PLUGIN.fetch_history()
        wnba.PLUGIN.fetch_history()
        assert seen == [("2015-16", "00"), ("2015", "10")]
