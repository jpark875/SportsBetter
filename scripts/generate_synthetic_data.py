"""
scripts/generate_synthetic_data.py

Generates a realistic synthetic NBA game dataset for pipeline validation
and ML benchmarking.  Requires NO API keys and NO internet access.

The data-generating process (DGP) is calibrated to real NBA distributions:

  Team ratings  — drawn from Normal(0, 5) net-rating, matching the ~10-point
                  spread between best and worst teams in a typical season.
  Home advantage — adds ~3.2 pts to the home team's effective net rating,
                   consistent with empirical estimates since the 2020 bubble.
  Back-to-back  — deducts 2.5 pts from a team's effective net rating,
                   reflecting fatigue observed in pace-adjusted metrics.
  Noise          — residual game-level variance of ~8 pts (roughly the
                   std-dev of point differentials after controlling for quality).
  Market line    — set to 90% of the "true" spread plus Gaussian noise,
                   simulating an efficient but not perfectly accurate market.

The synthetic data is used exclusively to validate that the ML pipeline
produces sensible outputs. For a *production* model you must replace it
with real historical data (see README — "Training on Real Data").

Usage
-----
    from scripts.generate_synthetic_data import SyntheticNBADataset

    ds = SyntheticNBADataset(n_seasons=4, games_per_season=1_230, seed=42)
    X_train, X_test, y_train, y_test = ds.train_test_split()
    odds_like_df = ds.odds_dataframe()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants mirroring realistic NBA distributions
# ---------------------------------------------------------------------------

TEAM_NAMES = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

_HOME_ADVANTAGE_PTS: float = 1.5
_B2B_PENALTY_PTS: float = 2.5
_TEAM_RATING_STD: float = 5.0      # cross-team spread
_GAME_NOISE_STD: float = 8.0       # within-game variance
_MARKET_EFFICIENCY: float = 0.90   # fraction of true spread captured by the line
_MARKET_NOISE_STD: float = 1.5     # residual line noise


class SyntheticNBADataset:
    """
    Generates multiple NBA seasons of synthetic game-level data.

    Parameters
    ----------
    n_seasons : int
        Number of seasons to simulate.  Each season has ``games_per_season`` games.
    games_per_season : int
        Total games per season (real NBA: 1,230).
    seed : int
        Random seed for reproducibility.
    test_season_frac : float
        Fraction of seasons to hold out as a temporal test set.
        Default 0.25 → last season is the test set.
    """

    def __init__(
        self,
        n_seasons: int = 4,
        games_per_season: int = 1_230,
        seed: int = 42,
        test_season_frac: float = 0.25,
    ) -> None:
        self.n_seasons = n_seasons
        self.games_per_season = games_per_season
        self.seed = seed
        self.test_season_frac = test_season_frac
        self._rng = np.random.default_rng(seed)
        self._df: Optional[pd.DataFrame] = None
        self._generate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        """Full labelled dataset."""
        return self._df.copy()

    def train_test_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Temporal train/test split (last ``test_season_frac`` of seasons as test).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            ``(X_train, X_test, y_train, y_test)``
        """
        df = self._df
        cutoff_season = int(self.n_seasons * (1 - self.test_season_frac))
        train = df[df["season"] < cutoff_season]
        test = df[df["season"] >= cutoff_season]

        feature_cols = self._feature_columns()
        return (
            train[feature_cols],
            test[feature_cols],
            train["HOME_WIN"],
            test["HOME_WIN"],
        )

    def odds_dataframe(self) -> pd.DataFrame:
        """
        Return an odds-like DataFrame in the canonical OddsClient schema.

        Useful for end-to-end pipeline smoke-tests that need an odds feed
        without calling any external API.
        """
        df = self._df.copy()
        rows = []
        for _, row in df.iterrows():
            # Moneyline home
            rows.append({
                "game_id": row["game_id"],
                "commence_time": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "sport_key": "basketball_nba",
                "bet_type": "moneyline",
                "market_key": "h2h",
                "bookmaker": "synthetic_book",
                "outcome_name": row["home_team"],
                "price": _prob_to_american(row["ML_IMPLIED_HOME_PROB"]),
                "point": float("nan"),
                "player_name": "",
                "prop_stat": "",
                "implied_prob": row["ML_IMPLIED_HOME_PROB"],
            })
            # Moneyline away
            rows.append({
                "game_id": row["game_id"],
                "commence_time": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "sport_key": "basketball_nba",
                "bet_type": "moneyline",
                "market_key": "h2h",
                "bookmaker": "synthetic_book",
                "outcome_name": row["away_team"],
                "price": _prob_to_american(1.0 - row["ML_IMPLIED_HOME_PROB"]),
                "point": float("nan"),
                "player_name": "",
                "prop_stat": "",
                "implied_prob": 1.0 - row["ML_IMPLIED_HOME_PROB"],
            })
            # Spread
            rows.append({
                "game_id": row["game_id"],
                "commence_time": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "sport_key": "basketball_nba",
                "bet_type": "spread",
                "market_key": "spreads",
                "bookmaker": "synthetic_book",
                "outcome_name": row["home_team"],
                "price": -110.0,
                "point": round(row["CONSENSUS_SPREAD"] * 2) / 2,
                "player_name": "",
                "prop_stat": "",
                "implied_prob": 0.5238,
            })
        return pd.DataFrame(rows)

    def feature_names(self) -> list[str]:
        return self._feature_columns()

    def describe(self) -> pd.DataFrame:
        """Summary statistics for the feature matrix."""
        return self._df[self._feature_columns()].describe()

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Build the full synthetic DataFrame."""
        all_seasons = []
        for season_idx in range(self.n_seasons):
            team_ratings = self._draw_team_ratings()
            season_df = self._simulate_season(season_idx, team_ratings)
            all_seasons.append(season_df)

        self._df = pd.concat(all_seasons, ignore_index=True)

    def _draw_team_ratings(self) -> dict[str, float]:
        """Draw true net ratings for all 30 teams this season."""
        ratings = self._rng.normal(0, _TEAM_RATING_STD, len(TEAM_NAMES))
        return dict(zip(TEAM_NAMES, ratings))

    def _simulate_season(
        self, season_idx: int, team_ratings: dict[str, float]
    ) -> pd.DataFrame:
        """Simulate one full season of games."""
        rows = []
        base_date = pd.Timestamp(f"{2021 + season_idx}-10-19")

        # Track last game date per team for rest-day calculation
        last_game: dict[str, Optional[pd.Timestamp]] = {t: None for t in TEAM_NAMES}

        for game_num in range(self.games_per_season):
            # Sample two distinct teams
            home_idx, away_idx = self._rng.choice(len(TEAM_NAMES), size=2, replace=False)
            home = TEAM_NAMES[home_idx]
            away = TEAM_NAMES[away_idx]

            game_date = base_date + pd.Timedelta(days=int(game_num * 150 / self.games_per_season))
            game_id = f"{home}_{away}_{game_date.strftime('%Y%m%d')}_{game_num}"

            # Rest days
            home_rest = self._rest_days(last_game[home], game_date)
            away_rest = self._rest_days(last_game[away], game_date)

            # Effective team ratings with situational adjustments
            home_eff = (
                team_ratings[home]
                + _HOME_ADVANTAGE_PTS
                - (_B2B_PENALTY_PTS if home_rest == 1 else 0)
            )
            away_eff = (
                team_ratings[away]
                - (_B2B_PENALTY_PTS if away_rest == 1 else 0)
            )

            # Advanced stat features as deltas (home − away)
            delta_net = home_eff - away_eff
            delta_off = delta_net * 0.6 + self._rng.normal(0, 1.5)
            delta_def = delta_net * 0.4 + self._rng.normal(0, 1.5)
            delta_pace = self._rng.normal(0, 2.0)
            delta_ast_pct = delta_net * 0.02 + self._rng.normal(0, 0.03)
            delta_oreb_pct = self._rng.normal(0, 0.02)
            delta_dreb_pct = delta_def * 0.003 + self._rng.normal(0, 0.02)
            delta_efg_pct = delta_net * 0.006 + self._rng.normal(0, 0.02)
            delta_ts_pct = delta_net * 0.005 + self._rng.normal(0, 0.015)

            # True outcome
            point_diff = delta_net + self._rng.normal(0, _GAME_NOISE_STD)
            home_win = int(point_diff > 0)

            # Market-implied probability (efficient but noisy)
            true_spread = delta_net
            market_spread = (
                -(_MARKET_EFFICIENCY * true_spread)
                + self._rng.normal(0, _MARKET_NOISE_STD)
            )
            market_spread = round(market_spread * 2) / 2   # half-point granularity
            ml_implied_home_prob = float(np.clip(
                0.5 + true_spread * _MARKET_EFFICIENCY / 25.0
                + self._rng.normal(0, 0.03),
                0.15, 0.85,
            ))
            consensus_total = 220.0 + self._rng.normal(0, 8.0)

            rows.append({
                "season": season_idx,
                "game_id": game_id,
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                # Target
                "HOME_WIN": home_win,
                # Delta features
                "DELTA_PACE": delta_pace,
                "DELTA_OFF_RATING": delta_off,
                "DELTA_DEF_RATING": delta_def,
                "DELTA_NET_RATING": delta_net,
                "DELTA_AST_PCT": delta_ast_pct,
                "DELTA_OREB_PCT": delta_oreb_pct,
                "DELTA_DREB_PCT": delta_dreb_pct,
                "DELTA_EFG_PCT": delta_efg_pct,
                "DELTA_TS_PCT": delta_ts_pct,
                # Schedule features
                "HOME_REST_DAYS": home_rest,
                "AWAY_REST_DAYS": away_rest,
                "HOME_IS_B2B": int(home_rest == 1),
                "AWAY_IS_B2B": int(away_rest == 1),
                "HOME_IS_B2B_FIRST": int(self._rng.uniform() < 0.08),
                "AWAY_IS_B2B_FIRST": int(self._rng.uniform() < 0.08),
                # Market features
                "ML_IMPLIED_HOME_PROB": ml_implied_home_prob,
                "CONSENSUS_SPREAD": market_spread,
                "CONSENSUS_TOTAL": round(consensus_total * 2) / 2,
            })

            last_game[home] = game_date
            last_game[away] = game_date

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rest_days(
        last: Optional[pd.Timestamp], current: pd.Timestamp
    ) -> int:
        if last is None:
            return 7
        return min(int((current - last).days - 1), 7)

    @staticmethod
    def _feature_columns() -> list[str]:
        return [
            "DELTA_PACE", "DELTA_OFF_RATING", "DELTA_DEF_RATING",
            "DELTA_NET_RATING", "DELTA_AST_PCT", "DELTA_OREB_PCT",
            "DELTA_DREB_PCT", "DELTA_EFG_PCT", "DELTA_TS_PCT",
            "HOME_REST_DAYS", "AWAY_REST_DAYS",
            "HOME_IS_B2B", "AWAY_IS_B2B",
            "HOME_IS_B2B_FIRST", "AWAY_IS_B2B_FIRST",
            "ML_IMPLIED_HOME_PROB", "CONSENSUS_SPREAD", "CONSENSUS_TOTAL",
        ]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _prob_to_american(p: float) -> float:
    """Convert win probability to American moneyline odds (no vig)."""
    p = float(np.clip(p, 0.01, 0.99))
    if p >= 0.5:
        return round(-(p / (1 - p)) * 100)
    return round(((1 - p) / p) * 100)
