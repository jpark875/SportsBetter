"""
sports/epl.py

English Premier League plugin.  History comes from football-data.co.uk's
free per-season CSVs (results + closing odds, maintained since the 90s).

The EPL is a 3-way market (draws ≈ 24% of matches), so the trainer
produces a 3-class model exactly like the World Cup engine.

Team names: football-data uses short names ("Man United", "Nott'm Forest")
while TheOddsAPI uses full names ("Manchester United") — normalize_name
maps odds names onto the football-data canon.
"""

from __future__ import annotations

import logging
from datetime import date
from io import StringIO
from typing import Dict, List

import pandas as pd
import requests

from sports.base import SportConfig, SportPlugin

logger = logging.getLogger(__name__)

_CSV_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
_FIRST_SEASON_START = 2010  # season code 1011

# TheOddsAPI name → football-data name
_NAME_MAP: Dict[str, str] = {
    "manchester united":          "Man United",
    "manchester city":            "Man City",
    "tottenham hotspur":          "Tottenham",
    "newcastle united":           "Newcastle",
    "wolverhampton wanderers":    "Wolves",
    "nottingham forest":          "Nott'm Forest",
    "brighton and hove albion":   "Brighton",
    "west ham united":            "West Ham",
    "leeds united":               "Leeds",
    "leicester city":             "Leicester",
    "luton town":                 "Luton",
    "ipswich town":               "Ipswich",
    "afc bournemouth":            "Bournemouth",
    "sheffield united":           "Sheffield United",
}


class EPLPlugin(SportPlugin):
    config = SportConfig(
        key="epl",
        display_name="Premier League",
        odds_sport_key="soccer_epl",
        has_draws=True,
        elo_k=20.0,
        hfa_elo=60.0,
        default_score=1.4,
        score_label="goals",
    )

    def normalize_name(self, name: str) -> str:
        return _NAME_MAP.get(name.strip().lower(), name.strip())

    def fetch_history(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        # Season code "1011" = 2010-11; the season starting in August of
        # year Y is fully played by June of Y+1.
        last_start = date.today().year if date.today().month >= 8 else date.today().year - 1

        for start in range(_FIRST_SEASON_START, last_start + 1):
            code = f"{start % 100:02d}{(start + 1) % 100:02d}"
            url = _CSV_URL.format(season=code)
            logger.info("EPL season %s …", code)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("  season %s unavailable (%s) — skipping.", code, exc)
                continue

            season_df = pd.read_csv(StringIO(resp.text), on_bad_lines="skip")
            needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            if not all(c in season_df.columns for c in needed):
                logger.warning("  season %s missing columns — skipping.", code)
                continue
            frames.append(season_df[needed])

        df = pd.concat(frames, ignore_index=True).dropna(subset=["FTHG", "FTAG"])
        df = df.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG": "home_score", "FTAG": "away_score",
        })
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        df = df.drop(columns=["Date"])
        logger.info("EPL: %d matches (%s – %s).",
                    len(df), df["date"].min().date(), df["date"].max().date())
        return df


PLUGIN = EPLPlugin()
