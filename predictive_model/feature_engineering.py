"""
predictive_model/feature_engineering.py

Transforms raw API payloads from the data_pipeline layer into a clean,
model-ready feature matrix.

The central artefact is ``FeatureMatrix`` — a labelled DataFrame where
every row represents one ``(game_id, team_side)`` observation and every
column is a numeric feature ready for LightGBM / XGBoost ingestion.

Key alignment contract
----------------------
* NBA stats tables are keyed on ``TEAM_ID`` or ``PLAYER_ID``.
* Odds tables are keyed on ``game_id`` (see odds_client.py canonical schema).
* This module joins them using the ``team_name → TEAM_ID`` mapping from
  ``nba_api.stats.static.teams``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as nba_teams_static

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static name → ID mapping
# ---------------------------------------------------------------------------

def _build_team_name_map() -> Dict[str, int]:
    """Return ``{full_name.lower(): team_id}`` and ``{abbreviation: team_id}``."""
    mapping: Dict[str, int] = {}
    for t in nba_teams_static.get_teams():
        mapping[t["full_name"].lower()] = t["id"]
        mapping[t["abbreviation"].lower()] = t["id"]
        mapping[t["nickname"].lower()] = t["id"]
    return mapping


_TEAM_NAME_MAP: Dict[str, int] = _build_team_name_map()


def resolve_team_id(name: str) -> Optional[int]:
    """
    Fuzzy-resolve a team name string to a canonical ``TEAM_ID``.

    Handles full names (``"Boston Celtics"``), nicknames (``"Celtics"``),
    and three-letter abbreviations (``"BOS"``).
    """
    return _TEAM_NAME_MAP.get(name.strip().lower())


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Builds the model feature matrix by merging stats and odds data.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Output of ``NBAStatsClient.fetch_team_advanced_stats()``.
        Indexed on ``TEAM_ID``.
    schedule_features : pd.DataFrame
        Output of ``NBAStatsClient.fetch_schedule_features()``.
        Multi-indexed on ``(TEAM_ID, GAME_ID)``.
    player_stats : pd.DataFrame, optional
        Output of ``NBAStatsClient.fetch_player_advanced_stats()``.
        Indexed on ``PLAYER_ID``.  Required for prop features.
    """

    # Features that will be delta'd: home_value − away_value
    _DELTA_FEATURES: List[str] = [
        "PACE",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "AST_PCT",
        "OREB_PCT",
        "DREB_PCT",
        "EFG_PCT",
        "TS_PCT",
    ]

    def __init__(
        self,
        team_stats: pd.DataFrame,
        schedule_features: pd.DataFrame,
        player_stats: Optional[pd.DataFrame] = None,
    ) -> None:
        self.team_stats = team_stats
        self.schedule_features = schedule_features
        self.player_stats = player_stats

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_game_features(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build one feature row per game from a canonical odds DataFrame.

        Each row represents a single game; the target variable ``HOME_WIN``
        is ``NaN`` for live (unlabelled) observations and must be populated
        from historical outcome data before training.

        Parameters
        ----------
        odds_df : pd.DataFrame
            Canonical odds schema from :class:`~data_pipeline.odds_client.OddsClient`.
            Must contain at least ``game_id``, ``home_team``, ``away_team``,
            ``bet_type``, ``price``, ``point`` columns.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed on ``game_id``.
        """
        games = (
            odds_df[["game_id", "home_team", "away_team", "commence_time"]]
            .drop_duplicates("game_id")
            .copy()
        )

        # Attach team IDs
        games["HOME_TEAM_ID"] = games["home_team"].apply(resolve_team_id)
        games["AWAY_TEAM_ID"] = games["away_team"].apply(resolve_team_id)

        unresolved = games[games["HOME_TEAM_ID"].isna() | games["AWAY_TEAM_ID"].isna()]
        if not unresolved.empty:
            logger.warning(
                "Could not resolve team IDs for %d game(s): %s",
                len(unresolved),
                unresolved[["home_team", "away_team"]].values.tolist(),
            )

        games = games.dropna(subset=["HOME_TEAM_ID", "AWAY_TEAM_ID"])
        games["HOME_TEAM_ID"] = games["HOME_TEAM_ID"].astype(int)
        games["AWAY_TEAM_ID"] = games["AWAY_TEAM_ID"].astype(int)

        # --- Merge team advanced stats ------------------------------------
        games = self._merge_team_stats(games)

        # --- Merge schedule / rest features --------------------------------
        games = self._merge_schedule(games, odds_df)

        # --- Merge market-implied features ---------------------------------
        games = self._merge_market_features(games, odds_df)

        # Placeholder target (populated later from historical outcomes)
        games["HOME_WIN"] = np.nan

        games.set_index("game_id", inplace=True)
        return games

    def build_prop_features(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build one feature row per player-prop wager.

        Requires ``player_stats`` to have been supplied at construction.

        Parameters
        ----------
        odds_df : pd.DataFrame
            Canonical odds schema filtered to ``bet_type == "player_prop"``.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed on ``(game_id, player_name, prop_stat, outcome_name)``.
        """
        if self.player_stats is None:
            raise ValueError(
                "player_stats must be provided to build prop features."
            )

        props = odds_df[odds_df["bet_type"] == "player_prop"].copy()
        if props.empty:
            return pd.DataFrame()

        props = self._merge_player_stats(props)
        props = self._merge_market_features_props(props)
        props.set_index(
            ["game_id", "player_name", "prop_stat", "outcome_name"], inplace=True
        )
        return props

    # ------------------------------------------------------------------
    # Private merge helpers
    # ------------------------------------------------------------------

    def _merge_team_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        """Attach home/away advanced stats and compute difference features."""
        ts = self.team_stats.copy()

        # Degrade gracefully when team stats are unavailable
        if ts.empty or not any(f in ts.columns for f in self._DELTA_FEATURES):
            for feat in self._DELTA_FEATURES:
                games[f"HOME_{feat}"] = np.nan
                games[f"AWAY_{feat}"] = np.nan
                games[f"DELTA_{feat}"] = np.nan
            return games

        home_cols = {c: f"HOME_{c}" for c in self._DELTA_FEATURES}
        away_cols = {c: f"AWAY_{c}" for c in self._DELTA_FEATURES}

        home_ts = ts[self._DELTA_FEATURES].rename(columns=home_cols)
        away_ts = ts[self._DELTA_FEATURES].rename(columns=away_cols)

        games = games.join(home_ts, on="HOME_TEAM_ID", how="left")
        games = games.join(away_ts, on="AWAY_TEAM_ID", how="left")

        # Delta = home advantage expressed as signed difference
        for feat in self._DELTA_FEATURES:
            games[f"DELTA_{feat}"] = (
                games[f"HOME_{feat}"] - games[f"AWAY_{feat}"]
            )

        return games

    def _merge_schedule(
        self, games: pd.DataFrame, odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach rest-day and back-to-back features for home and away teams."""
        if self.schedule_features.empty:
            for side in ("HOME", "AWAY"):
                for col in ("REST_DAYS", "IS_B2B", "IS_B2B_FIRST"):
                    games[f"{side}_{col}"] = np.nan
            return games

        sched = self.schedule_features.reset_index()

        # We need the nba_api GAME_ID, not our synthetic game_id.
        # The schedule features DataFrame holds GAME_DATE; we match by
        # TEAM_ID + GAME_DATE proximity to commence_time.
        # For a production system this would use the actual GAME_ID from
        # the LeagueGameLog endpoint.  Here we do a nearest-date merge.

        game_dates = (
            odds_df[["game_id", "commence_time"]]
            .drop_duplicates("game_id")
            .copy()
        )
        game_dates["GAME_DATE"] = pd.to_datetime(
            game_dates["commence_time"]
        ).dt.normalize()

        for side, id_col in (("HOME", "HOME_TEAM_ID"), ("AWAY", "AWAY_TEAM_ID")):
            merged = games[["game_id", id_col]].merge(
                game_dates[["game_id", "GAME_DATE"]], on="game_id", how="left"
            )
            merged = merged.merge(
                sched[["TEAM_ID", "GAME_DATE", "REST_DAYS", "IS_B2B", "IS_B2B_FIRST"]],
                left_on=[id_col, "GAME_DATE"],
                right_on=["TEAM_ID", "GAME_DATE"],
                how="left",
            )
            for col in ("REST_DAYS", "IS_B2B", "IS_B2B_FIRST"):
                games[f"{side}_{col}"] = merged[col].values

        return games

    def _merge_market_features(
        self, games: pd.DataFrame, odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract consensus market signals (best moneyline implied prob,
        consensus spread, total line) and attach them to the games table.
        """
        # Best moneyline for home team
        ml = odds_df[odds_df["bet_type"] == "moneyline"].copy()
        if not ml.empty:
            home_ml = (
                ml[ml.apply(lambda r: r["outcome_name"].lower() in
                    r["home_team"].lower(), axis=1)]
                .groupby("game_id")["price"]
                .mean()
                .rename("ML_HOME_PRICE_CONSENSUS")
            )
            away_ml = (
                ml[ml.apply(lambda r: r["outcome_name"].lower() in
                    r["away_team"].lower(), axis=1)]
                .groupby("game_id")["price"]
                .mean()
                .rename("ML_AWAY_PRICE_CONSENSUS")
            )
            games = games.join(home_ml, on="game_id", how="left")
            games = games.join(away_ml, on="game_id", how="left")
            games["ML_IMPLIED_HOME_PROB"] = games["ML_HOME_PRICE_CONSENSUS"].apply(
                lambda p: _american_implied(p) if pd.notna(p) else np.nan
            )

        # Consensus spread (home perspective, positive = home favourite)
        spread = odds_df[odds_df["bet_type"] == "spread"].copy()
        if not spread.empty:
            home_spread = (
                spread[spread.apply(
                    lambda r: r["outcome_name"].lower() in r["home_team"].lower(),
                    axis=1,
                )]
                .groupby("game_id")["point"]
                .mean()
                .rename("CONSENSUS_SPREAD")
            )
            games = games.join(home_spread, on="game_id", how="left")

        # Consensus total (O/U line)
        total = odds_df[odds_df["bet_type"] == "total"].copy()
        if not total.empty:
            over_line = (
                total[total["outcome_name"] == "Over"]
                .groupby("game_id")["point"]
                .mean()
                .rename("CONSENSUS_TOTAL")
            )
            games = games.join(over_line, on="game_id", how="left")

        return games

    def _merge_player_stats(self, props: pd.DataFrame) -> pd.DataFrame:
        """Join player advanced stats onto the props DataFrame by player name."""
        if self.player_stats is None:
            return props
        ps = self.player_stats.reset_index()[
            ["PLAYER_ID", "PLAYER_NAME", "USG_PCT", "TS_PCT", "NET_RATING", "PIE"]
        ].copy()
        ps["PLAYER_NAME_LOWER"] = ps["PLAYER_NAME"].str.lower()
        props["_name_lower"] = props["player_name"].str.lower()
        props = props.merge(
            ps.drop(columns="PLAYER_NAME"),
            left_on="_name_lower",
            right_on="PLAYER_NAME_LOWER",
            how="left",
        ).drop(columns=["_name_lower", "PLAYER_NAME_LOWER"])
        return props

    def _merge_market_features_props(self, props: pd.DataFrame) -> pd.DataFrame:
        """Attach market-implied probability for Over outcome of each prop."""
        def implied(row: pd.Series) -> float:
            if row["outcome_name"].lower() == "over" and pd.notna(row["price"]):
                return _american_implied(row["price"])
            return np.nan

        props["PROP_IMPLIED_OVER_PROB"] = props.apply(implied, axis=1)
        return props


# ---------------------------------------------------------------------------
# Module-level utility (also used by _merge_market_features)
# ---------------------------------------------------------------------------

def _american_implied(american: float) -> float:
    if american >= 100:
        return 100.0 / (american + 100.0)
    return -american / (-american + 100.0)
