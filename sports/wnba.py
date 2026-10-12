"""WNBA plugin."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sports.base import SportConfig, SportPlugin

_FIRST_SEASON = 2015


class WNBAPlugin(SportPlugin):
    config = SportConfig(
        key="wnba",
        display_name="WNBA Basketball",
        odds_sport_key="basketball_wnba",
        has_draws=False,
        elo_k=28.0,
        hfa_elo=80.0,
        default_score=82.0,
        score_label="points",
    )

    def fetch_history(self) -> pd.DataFrame:
        from sports._basketball import fetch_basketball_history

        # WNBA seasons are single calendar years (May–Oct).
        seasons = [str(y) for y in range(_FIRST_SEASON, date.today().year + 1)]
        return fetch_basketball_history(seasons, league_id="10")


PLUGIN = WNBAPlugin()
