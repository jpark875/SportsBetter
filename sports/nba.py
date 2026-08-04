"""
sports/nba.py

NBA plugin.  History via stats.nba.com LeagueGameLog (league_id "00").

ELO tuning: K=20 with an ~80-point home bonus tracks the NBA's ~59%
historical home win rate.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from sports.base import SportConfig, SportPlugin

_FIRST_SEASON_START = 2015


class NBAPlugin(SportPlugin):
    config = SportConfig(
        key="nba",
        display_name="NBA Basketball",
        odds_sport_key="basketball_nba",
        has_draws=False,
        elo_k=20.0,
        hfa_elo=80.0,
        default_score=112.0,
        score_label="points",
    )

    def fetch_history(self) -> pd.DataFrame:
        from sports._basketball import fetch_basketball_history

        # Season "2024-25" runs Oct 2024 – Jun 2025.
        last_start = date.today().year if date.today().month >= 10 else date.today().year - 1
        seasons = [
            f"{y}-{(y + 1) % 100:02d}"
            for y in range(_FIRST_SEASON_START, last_start + 1)
        ]
        return fetch_basketball_history(seasons, league_id="00")


PLUGIN = NBAPlugin()
