"""
data_pipeline/nba_stats_client.py

Throttled NBA data client built on the `nba_api` package.
Pulls three categories of data required by the feature engineering layer:

  1. Player tracking / advanced stats  (Usage %, True Shooting %)
  2. Team advanced metrics             (Pace, OffRtg, DefRtg)
  3. Schedule impact features          (rest days, back-to-back flags)

All public methods return normalised pandas DataFrames keyed on
`PLAYER_ID` or `GAME_ID` so they can be joined cleanly downstream.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueGameLog,
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    PlayerDashboardByGeneralSplits,
    TeamGameLog,
)
from nba_api.stats.static import teams as nba_teams_static

from config.settings import NBA_API_CALLS_PER_MINUTE, NBA_API_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal throttle helper
# ---------------------------------------------------------------------------

class _ThrottledCaller:
    """Enforces a per-minute rate limit on nba_api calls to avoid 429s."""

    def __init__(self, calls_per_minute: int = NBA_API_CALLS_PER_MINUTE) -> None:
        self._min_interval: float = 60.0 / max(calls_per_minute, 1)
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


_throttle = _ThrottledCaller()


# ---------------------------------------------------------------------------
# Browser-like headers required by stats.nba.com to avoid request drops
# ---------------------------------------------------------------------------

_NBA_HEADERS: Dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

_RETRY_DELAYS: Tuple[int, ...] = (5, 15, 30)


def _with_retry(fn, *args, **kwargs):
    """
    Call ``fn(*args, **kwargs)`` up to ``len(_RETRY_DELAYS) + 1`` times.
    Retries only on timeout / connection errors; re-raises anything else.
    """
    last_exc: Optional[Exception] = None
    attempts = len(_RETRY_DELAYS) + 1
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            is_retryable = isinstance(
                exc,
                (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
            ) or "timeout" in str(exc).lower()
            if is_retryable and attempt < attempts:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "nba_api timeout (attempt %d/%d) — retrying in %ds …",
                    attempt, attempts, delay,
                )
                time.sleep(delay)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class NBAStatsClient:
    """
    Fetches historical NBA data via the unofficial nba_api library.

    Parameters
    ----------
    season : str
        NBA season string, e.g. ``"2024-25"``.
    season_type : str
        ``"Regular Season"`` or ``"Playoffs"``.
    """

    def __init__(
        self,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> None:
        self.season = season
        self.season_type = season_type
        self._team_id_map: Dict[str, int] = {
            t["abbreviation"]: t["id"]
            for t in nba_teams_static.get_teams()
        }

    # ------------------------------------------------------------------
    # Player advanced / tracking stats
    # ------------------------------------------------------------------

    def fetch_player_advanced_stats(self) -> pd.DataFrame:
        """
        Pull league-wide per-game advanced stats for all players.

        Returns a DataFrame with columns including::

            PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION,
            USG_PCT, TS_PCT, AST_PCT, REB_PCT, NET_RATING,
            PIE (Player Impact Estimate)

        Returns
        -------
        pd.DataFrame
            One row per player, indexed on ``PLAYER_ID``.
        """
        logger.info("Fetching player advanced stats for %s …", self.season)
        _throttle.wait()

        df: pd.DataFrame = _with_retry(
            lambda: LeagueDashPlayerStats(
                season=self.season,
                season_type_all_star=self.season_type,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=NBA_API_TIMEOUT,
                headers=_NBA_HEADERS,
            ).get_data_frames()[0]
        )

        # Normalise column names to lower-snake for downstream consistency
        df.columns = [c.upper() for c in df.columns]

        keep = [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "GP",
            "USG_PCT",
            "TS_PCT",
            "AST_PCT",
            "REB_PCT",
            "NET_RATING",
            "PIE",
        ]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()
        df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
        df.set_index("PLAYER_ID", inplace=True)

        logger.info("  → %d players fetched.", len(df))
        return df

    def fetch_player_game_log(self, player_id: int, last_n: int = 20) -> pd.DataFrame:
        """
        Return a player's last ``last_n`` game results with box-score columns.

        Parameters
        ----------
        player_id : int
        last_n : int
            Number of most-recent games to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``GAME_ID, GAME_DATE, MATCHUP, WL, PTS, REB, AST, …``
        """
        logger.debug("Fetching game log for player_id=%d", player_id)
        _throttle.wait()

        endpoint = PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season=self.season,
            season_type_playoffs=self.season_type,
            last_n_games=last_n,
            timeout=NBA_API_TIMEOUT,
        )
        # Index 0 is the overall dashboard; game-level splits live at index > 0
        frames = endpoint.get_data_frames()
        df = frames[0] if frames else pd.DataFrame()
        df.columns = [c.upper() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Team advanced metrics
    # ------------------------------------------------------------------

    def fetch_team_advanced_stats(self) -> pd.DataFrame:
        """
        Pull league-wide team advanced metrics (Pace, OffRtg, DefRtg, NetRtg).

        Returns
        -------
        pd.DataFrame
            One row per team, indexed on ``TEAM_ID``.  Key columns::

                TEAM_ID, TEAM_ABBREVIATION, PACE, OFF_RATING,
                DEF_RATING, NET_RATING, AST_PCT, AST_TO,
                OREB_PCT, DREB_PCT, EFG_PCT, TS_PCT
        """
        logger.info("Fetching team advanced stats for %s …", self.season)
        _throttle.wait()

        df: pd.DataFrame = _with_retry(
            lambda: LeagueDashTeamStats(
                season=self.season,
                season_type_all_star=self.season_type,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=NBA_API_TIMEOUT,
                headers=_NBA_HEADERS,
            ).get_data_frames()[0]
        )
        df.columns = [c.upper() for c in df.columns]

        keep = [
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "GP",
            "PACE",
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "AST_PCT",
            "AST_TO",
            "OREB_PCT",
            "DREB_PCT",
            "EFG_PCT",
            "TS_PCT",
        ]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        df.set_index("TEAM_ID", inplace=True)

        logger.info("  → %d teams fetched.", len(df))
        return df

    # ------------------------------------------------------------------
    # Schedule impact features
    # ------------------------------------------------------------------

    def fetch_schedule_features(
        self,
        team_abbreviations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build rest-day and back-to-back features for every team game.

        For each ``(TEAM_ID, GAME_ID)`` pair the method computes:

        - ``REST_DAYS``     — calendar days since last game (capped at 7).
        - ``IS_B2B``        — 1 if ``REST_DAYS == 1`` else 0.
        - ``IS_B2B_FIRST``  — 1 if today is the *first* leg of a back-to-back.
        - ``HOME``          — 1 if the team is the home side.

        Parameters
        ----------
        team_abbreviations : list[str], optional
            Subset of teams to process.  Defaults to all 30 franchises.

        Returns
        -------
        pd.DataFrame
            Keyed on ``(TEAM_ID, GAME_ID)``.
        """
        all_abbrevs = (
            team_abbreviations
            if team_abbreviations
            else list(self._team_id_map.keys())
        )
        records: List[Dict] = []

        for abbrev in all_abbrevs:
            team_id = self._team_id_map.get(abbrev)
            if team_id is None:
                logger.warning("Unknown abbreviation '%s', skipping.", abbrev)
                continue

            logger.debug("Fetching game log for team %s (id=%d)", abbrev, team_id)
            _throttle.wait()

            try:
                _tid = team_id  # capture current value for lambda closure
                df = _with_retry(
                    lambda: TeamGameLog(
                        team_id=_tid,
                        season=self.season,
                        season_type_all_star=self.season_type,
                        timeout=NBA_API_TIMEOUT,
                        headers=_NBA_HEADERS,
                    ).get_data_frames()[0]
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch log for %s: %s", abbrev, exc)
                continue

            df.columns = [c.upper() for c in df.columns]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df.sort_values("GAME_DATE", inplace=True)
            df["PREV_GAME_DATE"] = df["GAME_DATE"].shift(1)
            df["REST_DAYS"] = (
                (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days - 1
            ).clip(upper=7).fillna(7).astype(int)

            df["IS_B2B"] = (df["REST_DAYS"] == 1).astype(int)
            df["IS_B2B_FIRST"] = (
                df["REST_DAYS"].shift(-1) == 1
            ).fillna(False).astype(int)

            # MATCHUP format: "BOS vs. MIA"  or  "BOS @ MIA"
            df["HOME"] = df["MATCHUP"].apply(
                lambda m: 1 if "vs." in str(m) else 0
            )

            df["TEAM_ID"] = team_id
            records.append(
                df[["TEAM_ID", "GAME_ID", "GAME_DATE", "REST_DAYS",
                    "IS_B2B", "IS_B2B_FIRST", "HOME"]]
            )

        if not records:
            return pd.DataFrame()

        result = pd.concat(records, ignore_index=True)
        result.set_index(["TEAM_ID", "GAME_ID"], inplace=True)
        logger.info("Schedule features built for %d team-game rows.", len(result))
        return result

    def fetch_schedule_features_fast(self) -> pd.DataFrame:
        """
        Build schedule features from a **single** LeagueGameLog call instead
        of the 30-endpoint TeamGameLog loop in :meth:`fetch_schedule_features`.

        Produces the same schema (indexed on ``(TEAM_ID, GAME_ID)``) and is
        the preferred method for live-scoring runs where minimising API calls
        matters.

        Returns
        -------
        pd.DataFrame
            Keyed on ``(TEAM_ID, GAME_ID)`` with columns
            ``GAME_DATE, REST_DAYS, IS_B2B, IS_B2B_FIRST, HOME``.
        """
        logger.info(
            "Fetching schedule features via LeagueGameLog for %s …", self.season
        )
        _throttle.wait()

        df: pd.DataFrame = _with_retry(
            lambda: LeagueGameLog(
                season=self.season,
                season_type_all_star=self.season_type,
                timeout=NBA_API_TIMEOUT,
                headers=_NBA_HEADERS,
            ).get_data_frames()[0]
        )

        df.columns = [c.upper() for c in df.columns]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        df["HOME"] = df["MATCHUP"].apply(lambda m: 1 if "vs." in str(m) else 0)

        df.sort_values(["TEAM_ID", "GAME_DATE"], inplace=True)
        df["PREV_GAME_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
        df["REST_DAYS"] = (
            (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days - 1
        ).clip(upper=7).fillna(7).astype(int)
        df["IS_B2B"] = (df["REST_DAYS"] == 1).astype(int)
        df["IS_B2B_FIRST"] = (
            df.groupby("TEAM_ID")["REST_DAYS"].shift(-1) == 1
        ).fillna(False).astype(int)

        result = df[
            ["TEAM_ID", "GAME_ID", "GAME_DATE", "REST_DAYS", "IS_B2B", "IS_B2B_FIRST", "HOME"]
        ].copy()
        result.set_index(["TEAM_ID", "GAME_ID"], inplace=True)
        logger.info("  -> %d team-game schedule rows fetched.", len(result))
        return result

    # ------------------------------------------------------------------
    # Convenience batch loader
    # ------------------------------------------------------------------

    def load_all(
        self,
        team_abbreviations: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        One-shot loader that returns all three feature tables.

        Uses :meth:`fetch_schedule_features_fast` (single LeagueGameLog
        call) instead of the 30-TeamGameLog loop.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(player_advanced, team_advanced, schedule_features)``
        """
        player_df = self.fetch_player_advanced_stats()
        team_df = self.fetch_team_advanced_stats()
        sched_df = self.fetch_schedule_features_fast()
        return player_df, team_df, sched_df
