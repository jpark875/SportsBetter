"""
sports/mlb.py

MLB plugin.  History comes from the official MLB Stats API
(statsapi.mlb.com) — free, no key, JSON schedule per season.

ELO tuning: K=6 and HFA=24 follow FiveThirtyEight's published MLB model;
a 162-game season needs a slow-moving rating, and home advantage in
baseball is the smallest of the major sports (~54% home win rate).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List

import pandas as pd
import requests

from sports.base import SportConfig, SportPlugin

logger = logging.getLogger(__name__)

_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
_FIRST_SEASON = 2015


class MLBPlugin(SportPlugin):
    config = SportConfig(
        key="mlb",
        display_name="MLB Baseball",
        odds_sport_key="baseball_mlb",
        has_draws=False,
        elo_k=6.0,
        hfa_elo=24.0,
        default_score=4.5,
        score_label="runs",
        ewm_span=10,   # high game-to-game variance → longer smoothing
    )

    def fetch_history(self) -> pd.DataFrame:
        rows: List[dict] = []
        current_year = date.today().year

        for season in range(_FIRST_SEASON, current_year + 1):
            params = {
                "sportId": 1,
                "season": season,
                "gameType": "R,F,D,L,W",  # regular season + all postseason rounds
            }
            logger.info("MLB schedule %d …", season)
            resp = requests.get(_SCHEDULE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for day in data.get("dates", []):
                for game in day.get("games", []):
                    status = game.get("status", {}).get("abstractGameState", "")
                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})
                    if status != "Final":
                        continue
                    if "score" not in home or "score" not in away:
                        continue
                    rows.append({
                        "date": day["date"],
                        "home_team": home["team"]["name"],
                        "away_team": away["team"]["name"],
                        "home_score": float(home["score"]),
                        "away_score": float(away["score"]),
                    })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        # Suspended games can post as tied finals — meaningless for a
        # 2-way market, so drop them.
        df = df[df["home_score"] != df["away_score"]].reset_index(drop=True)
        logger.info("MLB: %d final games (%s – %s).",
                    len(df), df["date"].min().date(), df["date"].max().date())
        return df


PLUGIN = MLBPlugin()
