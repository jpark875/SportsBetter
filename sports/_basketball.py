"""
sports/_basketball.py

Shared history fetcher for NBA and WNBA — both live behind the same
stats.nba.com LeagueGameLog endpoint (league_id "00" vs "10"), so the
pairing logic (two team rows per GAME_ID → one home/away game row) is
written once here.

Reuses the browser headers and retry ladder already hardened in
data_pipeline/nba_stats_client.py.
"""

from __future__ import annotations

import logging
import time
from typing import List

import pandas as pd

from config.settings import NBA_API_TIMEOUT
from data_pipeline.nba_stats_client import _NBA_HEADERS, _with_retry

logger = logging.getLogger(__name__)

_CALL_GAP_SECONDS = 1.5  # be gentle: unofficial endpoint


def fetch_basketball_history(
    seasons: List[str],
    league_id: str,
    include_playoffs: bool = True,
) -> pd.DataFrame:
    """
    Fetch and pair game logs for the given seasons.

    Parameters
    ----------
    seasons : list[str]
        NBA format "2024-25"; WNBA format "2024".
    league_id : str
        "00" = NBA, "10" = WNBA.
    """
    from nba_api.stats.endpoints import leaguegamelog

    season_types = ["Regular Season"] + (["Playoffs"] if include_playoffs else [])
    frames: List[pd.DataFrame] = []

    for season in seasons:
        for stype in season_types:
            logger.info("LeagueGameLog league=%s season=%s (%s) …",
                        league_id, season, stype)
            try:
                log = _with_retry(
                    lambda: leaguegamelog.LeagueGameLog(
                        season=season,
                        league_id=league_id,
                        season_type_all_star=stype,
                        headers=_NBA_HEADERS,
                        timeout=NBA_API_TIMEOUT,
                    ).get_data_frames()[0]
                )
            except Exception as exc:  # noqa: BLE001 — one bad season shouldn't sink the rest
                logger.warning("  season %s (%s) failed: %s — skipping.",
                               season, stype, exc)
                continue
            if not log.empty:
                frames.append(log)
            time.sleep(_CALL_GAP_SECONDS)

    if not frames:
        raise RuntimeError("No basketball game logs could be fetched.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["PTS"])

    # Two rows per GAME_ID; "vs." in MATCHUP marks the home side.
    home = raw[raw["MATCHUP"].str.contains(" vs. ", na=False)][
        ["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS"]
    ].rename(columns={"TEAM_NAME": "home_team", "PTS": "home_score"})
    away = raw[raw["MATCHUP"].str.contains(" @ ", na=False)][
        ["GAME_ID", "TEAM_NAME", "PTS"]
    ].rename(columns={"TEAM_NAME": "away_team", "PTS": "away_score"})

    df = home.merge(away, on="GAME_ID", how="inner")
    df["date"] = pd.to_datetime(df["GAME_DATE"])
    df = df[["date", "home_team", "away_team", "home_score", "away_score"]]
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)
    df = df[df["home_score"] != df["away_score"]]  # no ties in basketball; guard anyway
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Basketball league=%s: %d games (%s – %s).",
                league_id, len(df), df["date"].min().date(), df["date"].max().date())
    return df
