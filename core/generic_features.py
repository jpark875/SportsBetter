"""
core/generic_features.py

Sport-agnostic feature engine.  Any sport whose games reduce to

    date | home_team | away_team | home_score | away_score

gets the full feature stack for free:

    ELO (with configurable K and home-advantage bonus)
    Rolling form over the last N games (points-per-game, scores for/against)
    Exponentially-weighted attack / defense ratings
    Attack-vs-defense matchup ratios

This is the same feature family validated on the World Cup engine
(data_pipeline/football_stats_client.py) — DELTA_ELO and the defense
ratings were its top-gain features — lifted out so new sports plug in
with ~100 lines of data-fetching code instead of a bespoke pipeline.

All rolling/EWM features are shift(1)-ed: a game's features only ever see
history strictly before that game.  ELO is computed in a single forward
chronological pass, so look-ahead is structurally impossible.

Result convention (matches the WC engine):
    RESULT = 0 → away win | 1 → draw | 2 → home win
For sports without draws, RESULT ∈ {0, 2} and the binary training target
is HOME_WIN = (RESULT == 2).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INITIAL_ELO: float = 1500.0

#: Feature columns produced by :func:`build_feature_table`, in model order.
GENERIC_FEATURE_COLS: list = [
    "HOME_ELO", "AWAY_ELO", "DELTA_ELO",
    "HOME_FORM_LAST5", "AWAY_FORM_LAST5",
    "HOME_SCORE_FOR_LAST5", "AWAY_SCORE_FOR_LAST5",
    "HOME_SCORE_AGAINST_LAST5", "AWAY_SCORE_AGAINST_LAST5",
    "HOME_ATK_RATING", "AWAY_ATK_RATING",
    "HOME_DEF_RATING", "AWAY_DEF_RATING",
    "HOME_ATK_VS_AWAY_DEF", "AWAY_ATK_VS_HOME_DEF",
]


def add_result_column(matches: pd.DataFrame) -> pd.DataFrame:
    """Add RESULT (0/1/2) and game_id, sort chronologically."""
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)

    df["RESULT"] = 1
    df.loc[df["home_score"] > df["away_score"], "RESULT"] = 2
    df.loc[df["home_score"] < df["away_score"], "RESULT"] = 0

    df = df.sort_values("date").reset_index(drop=True)
    df["game_id"] = (
        df["home_team"].str.replace(" ", "").str[:6].str.upper()
        + "_"
        + df["away_team"].str.replace(" ", "").str[:6].str.upper()
        + "_"
        + pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    )
    return df


def _elo_expected(rating_h: float, rating_a: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_a - rating_h - hfa) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    k_factor: float = 20.0,
    hfa_elo: float = 60.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Single chronological pass adding pre-match HOME_ELO / AWAY_ELO / DELTA_ELO.

    Parameters
    ----------
    k_factor : float
        Rating volatility. Lower for long seasons (MLB ~6), higher for
        short ones (WNBA ~24).
    hfa_elo : float
        Home-advantage bonus applied inside the expectation only (ratings
        themselves stay venue-neutral). 0 for neutral-site competitions.

    Returns
    -------
    (df, final_elo)
        ``df`` with the three ELO columns; ``final_elo`` maps every team to
        its post-history rating — exactly what live scoring needs, without
        a second pass.
    """
    elo: Dict[str, float] = {}
    home_elos, away_elos = [], []
    actual_map = {2: 1.0, 1: 0.5, 0: 0.0}

    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        he = elo.get(h, _INITIAL_ELO)
        ae = elo.get(a, _INITIAL_ELO)
        home_elos.append(he)
        away_elos.append(ae)

        exp_h = _elo_expected(he, ae, hfa_elo)
        actual_h = actual_map[row["RESULT"]]
        elo[h] = he + k_factor * (actual_h - exp_h)
        elo[a] = ae + k_factor * ((1 - actual_h) - (1 - exp_h))

    df = matches.copy()
    df["HOME_ELO"] = home_elos
    df["AWAY_ELO"] = away_elos
    df["DELTA_ELO"] = df["HOME_ELO"] - df["AWAY_ELO"]
    return df, elo


def _long_appearance_table(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): gf, ga, pts — the base for all rolling stats."""
    home = matches[["game_id", "date", "home_team", "home_score", "away_score", "RESULT"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
    )
    home["pts"] = home["RESULT"].map({2: 3, 1: 1, 0: 0})

    away = matches[["game_id", "date", "away_team", "away_score", "home_score", "RESULT"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
    )
    away["pts"] = away["RESULT"].map({2: 0, 1: 1, 0: 3})

    return pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)


def compute_form_and_ratings(
    matches: pd.DataFrame,
    form_window: int = 5,
    ewm_span: int = 7,
) -> pd.DataFrame:
    """
    Add rolling-form and EWM attack/defense columns (all shift(1)-ed).

    Form: mean pts (3/1/0) and scores for/against over the last
    ``form_window`` games.  Ratings: EWM (span=``ewm_span``) of scores
    for (attack) and against (defense), plus cross-matchup ratios.
    """
    long = _long_appearance_table(matches)

    grouped = long.groupby("team")
    long["form"] = grouped["pts"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    long["sf"] = grouped["gf"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    long["sa"] = grouped["ga"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    long["atk"] = grouped["gf"].transform(
        lambda s: s.shift(1).ewm(span=ewm_span, min_periods=1).mean())
    long["def_r"] = grouped["ga"].transform(
        lambda s: s.shift(1).ewm(span=ewm_span, min_periods=1).mean())

    stat_cols = ["form", "sf", "sa", "atk", "def_r"]
    home_stats = long.rename(columns={"team": "home_team"})[
        ["game_id", "home_team"] + stat_cols
    ].rename(columns={
        "form": "HOME_FORM_LAST5", "sf": "HOME_SCORE_FOR_LAST5",
        "sa": "HOME_SCORE_AGAINST_LAST5",
        "atk": "HOME_ATK_RATING", "def_r": "HOME_DEF_RATING",
    })
    away_stats = long.rename(columns={"team": "away_team"})[
        ["game_id", "away_team"] + stat_cols
    ].rename(columns={
        "form": "AWAY_FORM_LAST5", "sf": "AWAY_SCORE_FOR_LAST5",
        "sa": "AWAY_SCORE_AGAINST_LAST5",
        "atk": "AWAY_ATK_RATING", "def_r": "AWAY_DEF_RATING",
    })

    df = matches.merge(home_stats, on=["game_id", "home_team"], how="left")
    df = df.merge(away_stats, on=["game_id", "away_team"], how="left")

    # Matchup ratios; floor scaled to typical score magnitude so basketball
    # (~110 pts) and soccer (~1.4 goals) both avoid divide-by-near-zero.
    floor = max(df["HOME_DEF_RATING"].median() * 0.05, 0.1)
    df["HOME_ATK_VS_AWAY_DEF"] = df["HOME_ATK_RATING"] / (df["AWAY_DEF_RATING"] + floor)
    df["AWAY_ATK_VS_HOME_DEF"] = df["AWAY_ATK_RATING"] / (df["HOME_DEF_RATING"] + floor)
    return df


def build_feature_table(
    matches: pd.DataFrame,
    k_factor: float = 20.0,
    hfa_elo: float = 60.0,
    form_window: int = 5,
    ewm_span: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Full pipeline: RESULT/game_id → ELO → form + ratings.

    Returns ``(feature_df, final_elo)``.
    """
    df = add_result_column(matches)
    df, final_elo = compute_elo(df, k_factor=k_factor, hfa_elo=hfa_elo)
    df = compute_form_and_ratings(df, form_window=form_window, ewm_span=ewm_span)
    return df, final_elo


def latest_team_stats(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each team's most recent post-game rolling stats, for live scoring.

    Returns a DataFrame indexed on team with columns
    ``form, sf, sa, atk, def_r``.  Values come from the long appearance
    table *without* the shift — i.e. they include the team's latest game,
    which is exactly the history available before their next one.
    """
    long = _long_appearance_table(feature_df)
    grouped = long.groupby("team")
    # No shift here: for a FUTURE game, every played game is valid history.
    long["form"] = grouped["pts"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["sf"] = grouped["gf"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["sa"] = grouped["ga"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["atk"] = grouped["gf"].transform(lambda s: s.ewm(span=7, min_periods=1).mean())
    long["def_r"] = grouped["ga"].transform(lambda s: s.ewm(span=7, min_periods=1).mean())
    return (
        long.sort_values("date")
        .groupby("team")
        .last()[["form", "sf", "sa", "atk", "def_r"]]
    )


def build_scoring_row(
    home_team: str,
    away_team: str,
    current_elo: Dict[str, float],
    latest_stats: pd.DataFrame,
    default_score: float,
) -> Dict[str, float]:
    """
    Feature dict for one upcoming game, matching GENERIC_FEATURE_COLS.

    ``default_score`` is the sport's typical per-game score (runs, goals,
    points) used as prior for teams absent from history.
    """
    h_elo = current_elo.get(home_team, _INITIAL_ELO)
    a_elo = current_elo.get(away_team, _INITIAL_ELO)

    def _stat(team: str, col: str, default: float) -> float:
        if team in latest_stats.index:
            v = latest_stats.at[team, col]
            if not pd.isna(v):
                return float(v)
        return default

    h_atk = _stat(home_team, "atk", default_score)
    a_atk = _stat(away_team, "atk", default_score)
    h_def = _stat(home_team, "def_r", default_score)
    a_def = _stat(away_team, "def_r", default_score)
    floor = max(default_score * 0.05, 0.1)

    return {
        "HOME_ELO": h_elo,
        "AWAY_ELO": a_elo,
        "DELTA_ELO": h_elo - a_elo,
        "HOME_FORM_LAST5": _stat(home_team, "form", 1.5),
        "AWAY_FORM_LAST5": _stat(away_team, "form", 1.5),
        "HOME_SCORE_FOR_LAST5": _stat(home_team, "sf", default_score),
        "AWAY_SCORE_FOR_LAST5": _stat(away_team, "sf", default_score),
        "HOME_SCORE_AGAINST_LAST5": _stat(home_team, "sa", default_score),
        "AWAY_SCORE_AGAINST_LAST5": _stat(away_team, "sa", default_score),
        "HOME_ATK_RATING": h_atk,
        "AWAY_ATK_RATING": a_atk,
        "HOME_DEF_RATING": h_def,
        "AWAY_DEF_RATING": a_def,
        "HOME_ATK_VS_AWAY_DEF": h_atk / (a_def + floor),
        "AWAY_ATK_VS_HOME_DEF": a_atk / (h_def + floor),
    }
