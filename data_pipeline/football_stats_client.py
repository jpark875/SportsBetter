"""
data_pipeline/football_stats_client.py

Builds soccer / World Cup feature matrices from two public GitHub datasets
(martj42/international_results — no API key required):

    results.csv      ~49k international matches back to 1872
    goalscorers.csv  individual goal records tied to each match

Features computed (all with strict no-look-ahead):

    ELO
      HOME_ELO / AWAY_ELO / DELTA_ELO
          Classic ELO with variable K-factors per tournament type.

    Rolling form  (last 5 competitive matches, shift(1))
      HOME/AWAY_FORM_LAST5         avg points (3/1/0) per match
      HOME/AWAY_GOALS_FOR_LAST5    avg goals scored
      HOME/AWAY_GOALS_AGAINST_LAST5

    EWM attack / defense ratings  (span=7, shift(1))
      HOME/AWAY_ATK_RATING         exp-weighted avg goals scored
      HOME/AWAY_DEF_RATING         exp-weighted avg goals conceded

    Attack-vs-defense matchup scores
      HOME_ATK_VS_AWAY_DEF         home attack divided by away defense
      AWAY_ATK_VS_HOME_DEF         away attack divided by home defense
      (values >1 mean the attacking team scores more than opponent concedes)

    Player star concentration  (last 20 competitive matches, shift(1))
      HOME/AWAY_STAR_CONC          top scorer's share of team's goals
      0 = distributed scorer pool, 1 = one player scores everything

    Venue
      IS_NEUTRAL                   1 for all World Cup matches

Target: RESULT   0=away win  |  1=draw  |  2=home win
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ELO constants
# ---------------------------------------------------------------------------

_INITIAL_ELO: float = 1500.0

_K_MAP: Dict[str, int] = {
    "world cup":           60,
    "qualifier":           40,
    "qualification":       40,
    "euro":                40,
    "copa america":        40,
    "africa cup":          40,
    "african cup":         40,
    "asian cup":           40,
    "gold cup":            40,
    "nations league":      40,
    "confederations cup":  40,
    "olympics":            30,
}
_K_FRIENDLY = 10
_K_DEFAULT  = 30

# ---------------------------------------------------------------------------
# Team-name normalisation — TheOddsAPI <-> historical dataset
# ---------------------------------------------------------------------------

_NAME_MAP: Dict[str, str] = {
    "usa":                  "United States",
    "united states":        "United States",
    "south korea":          "South Korea",
    "korea republic":       "South Korea",
    "ivory coast":          "Ivory Coast",
    "cote d'ivoire":        "Ivory Coast",
    "republic of ireland":  "Republic of Ireland",
    "ir iran":              "Iran",
    "czech republic":       "Czech Republic",
    "turkiye":              "Turkey",
    "türkiye":              "Turkey",
    "north macedonia":      "North Macedonia",
    "cape verde":           "Cape Verde",
    "dr congo":             "DR Congo",
    "drc":                  "DR Congo",
}


def normalize_name(name: str) -> str:
    """Return a canonical team name, resolving known aliases."""
    return _NAME_MAP.get(name.strip().lower(), name.strip())


# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------

def _k_factor(tournament: str) -> int:
    t = tournament.lower()
    if "friendly" in t:
        return _K_FRIENDLY
    for keyword, k in _K_MAP.items():
        if keyword in t:
            return k
    return _K_DEFAULT


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(url: str) -> pd.DataFrame:
    """
    Download the international results CSV from GitHub.

    Returns a clean DataFrame with columns:
        date, home_team, away_team, home_score, away_score,
        tournament, neutral, RESULT (0=away 1=draw 2=home)
    """
    logger.info("Downloading international results from %s …", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"]    = df["neutral"].astype(bool)

    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)

    df["RESULT"] = 1
    df.loc[df["home_score"] > df["away_score"], "RESULT"] = 2
    df.loc[df["home_score"] < df["away_score"], "RESULT"] = 0

    df = df.sort_values("date").reset_index(drop=True)
    df["game_id"] = (
        df["home_team"].str.replace(" ", "").str[:6].str.upper()
        + "_"
        + df["away_team"].str.replace(" ", "").str[:6].str.upper()
        + "_"
        + df["date"].dt.strftime("%Y%m%d")
    )
    logger.info("  %d matches loaded (%s – %s).",
                len(df), df["date"].min().date(), df["date"].max().date())
    return df


def load_goalscorers(url: str) -> pd.DataFrame:
    """
    Download the goalscorers CSV from GitHub.

    Returns a DataFrame with columns:
        date, home_team, away_team, team, scorer, own_goal, penalty
    """
    logger.info("Downloading goalscorer data …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    df.columns = [c.lower() for c in df.columns]
    df["date"]      = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)
    df["team"]      = df["team"].apply(normalize_name)
    df["own_goal"]  = df.get("own_goal", pd.Series(False)).fillna(False).astype(bool)
    df = df[df["scorer"].notna()].copy()
    logger.info("  %d goal records loaded.", len(df))
    return df


# ---------------------------------------------------------------------------
# ELO computation
# ---------------------------------------------------------------------------

def compute_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add HOME_ELO, AWAY_ELO, DELTA_ELO (rating BEFORE each match).
    Iterates chronologically — future matches cannot contaminate ratings.
    """
    elo: Dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        he = elo.get(h, _INITIAL_ELO)
        ae = elo.get(a, _INITIAL_ELO)
        home_elos.append(he)
        away_elos.append(ae)

        k        = _k_factor(row.get("tournament", ""))
        exp_h    = _elo_expected(he, ae)
        actual_h = {2: 1.0, 1: 0.5, 0: 0.0}[row["RESULT"]]
        elo[h]   = he + k * (actual_h        - exp_h)
        elo[a]   = ae + k * ((1 - actual_h)  - (1 - exp_h))

    df = matches.copy()
    df["HOME_ELO"]   = home_elos
    df["AWAY_ELO"]   = away_elos
    df["DELTA_ELO"]  = df["HOME_ELO"] - df["AWAY_ELO"]
    return df


# ---------------------------------------------------------------------------
# Rolling form features
# ---------------------------------------------------------------------------

def compute_form_history(
    matches: pd.DataFrame,
    window: int = 5,
    competitive_only: bool = True,
) -> pd.DataFrame:
    """
    Add shift(1) rolling-average form columns from each team's last ``window``
    competitive matches.

    New columns: HOME/AWAY_FORM_LAST5,
                 HOME/AWAY_GOALS_FOR_LAST5, HOME/AWAY_GOALS_AGAINST_LAST5
    """
    src = matches.copy()
    if competitive_only:
        src = src[~src["tournament"].str.lower().str.contains("friendly")].copy()

    home_side = src[["game_id", "date", "home_team", "home_score", "away_score", "RESULT"]].copy()
    home_side = home_side.rename(columns={"home_team": "team",
                                          "home_score": "gf", "away_score": "ga"})
    home_side["pts"] = home_side["RESULT"].map({2: 3, 1: 1, 0: 0})

    away_side = src[["game_id", "date", "away_team", "away_score", "home_score", "RESULT"]].copy()
    away_side = away_side.rename(columns={"away_team": "team",
                                          "away_score": "gf", "home_score": "ga"})
    away_side["pts"] = away_side["RESULT"].map({2: 0, 1: 1, 0: 3})

    long = pd.concat([home_side, away_side]).sort_values(["team", "date"]).reset_index(drop=True)

    for col in ("pts", "gf", "ga"):
        long[f"form_{col}"] = long.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    form_home = long.rename(columns={"team": "home_team",
                                     "form_pts": "HOME_FORM_LAST5",
                                     "form_gf":  "HOME_GOALS_FOR_LAST5",
                                     "form_ga":  "HOME_GOALS_AGAINST_LAST5"})[
        ["game_id", "home_team",
         "HOME_FORM_LAST5", "HOME_GOALS_FOR_LAST5", "HOME_GOALS_AGAINST_LAST5"]
    ]
    form_away = long.rename(columns={"team": "away_team",
                                     "form_pts": "AWAY_FORM_LAST5",
                                     "form_gf":  "AWAY_GOALS_FOR_LAST5",
                                     "form_ga":  "AWAY_GOALS_AGAINST_LAST5"})[
        ["game_id", "away_team",
         "AWAY_FORM_LAST5", "AWAY_GOALS_FOR_LAST5", "AWAY_GOALS_AGAINST_LAST5"]
    ]

    df = matches.copy()
    df = df.merge(form_home, on=["game_id", "home_team"], how="left")
    df = df.merge(form_away, on=["game_id", "away_team"], how="left")
    return df


# ---------------------------------------------------------------------------
# EWM attack / defense ratings
# ---------------------------------------------------------------------------

def compute_attack_defense_ratings(
    matches: pd.DataFrame,
    span: int = 7,
) -> pd.DataFrame:
    """
    Add exponentially-weighted attack and defense ratings (span≈7 = ~4-match
    half-life so recent performances carry more weight than a simple average).

    All values are shifted by 1 so only pre-match history is used.

    New columns:
        HOME/AWAY_ATK_RATING         EWM goals scored per match
        HOME/AWAY_DEF_RATING         EWM goals conceded per match
        HOME_ATK_VS_AWAY_DEF         home attack / away defense  (>1 = mismatch)
        AWAY_ATK_VS_HOME_DEF         away attack / home defense
    """
    home_side = matches[["game_id", "date", "home_team", "home_score", "away_score"]].copy()
    home_side = home_side.rename(columns={"home_team": "team",
                                          "home_score": "gf", "away_score": "ga"})
    away_side = matches[["game_id", "date", "away_team", "away_score", "home_score"]].copy()
    away_side = away_side.rename(columns={"away_team": "team",
                                          "away_score": "gf", "home_score": "ga"})

    long = pd.concat([home_side, away_side]).sort_values(["team", "date"]).reset_index(drop=True)

    long["atk"] = long.groupby("team")["gf"].transform(
        lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
    )
    long["def_r"] = long.groupby("team")["ga"].transform(
        lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
    )

    atk_home = long.rename(columns={"team":  "home_team",
                                    "atk":   "HOME_ATK_RATING",
                                    "def_r": "HOME_DEF_RATING"})[
        ["game_id", "home_team", "HOME_ATK_RATING", "HOME_DEF_RATING"]
    ]
    atk_away = long.rename(columns={"team":  "away_team",
                                    "atk":   "AWAY_ATK_RATING",
                                    "def_r": "AWAY_DEF_RATING"})[
        ["game_id", "away_team", "AWAY_ATK_RATING", "AWAY_DEF_RATING"]
    ]

    df = matches.copy()
    df = df.merge(atk_home, on=["game_id", "home_team"], how="left")
    df = df.merge(atk_away, on=["game_id", "away_team"], how="left")

    # Matchup: attack vs opponent's defense (add 0.1 floor to avoid /0)
    df["HOME_ATK_VS_AWAY_DEF"] = df["HOME_ATK_RATING"] / (df["AWAY_DEF_RATING"] + 0.1)
    df["AWAY_ATK_VS_HOME_DEF"] = df["AWAY_ATK_RATING"] / (df["HOME_DEF_RATING"] + 0.1)

    return df


# ---------------------------------------------------------------------------
# Player star-concentration feature
# ---------------------------------------------------------------------------

def compute_player_concentration(
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    For each match, compute each team's goal-scoring concentration over their
    last ``window`` competitive appearances before that date.

    Concentration = goals by team's top scorer / total team goals.
    A value near 1.0 means one player carries the scoring load (high risk
    if that player is absent); near 0 means distributed scoring.

    New columns: HOME_STAR_CONC, AWAY_STAR_CONC
    """
    # Use only non-own-goal records in competitive matches
    gs = goalscorers[~goalscorers["own_goal"]].copy()
    competitive_ids = set(
        matches[~matches["tournament"].str.lower().str.contains("friendly")]["game_id"]
    )

    # Goals per (date, team, scorer)
    goal_counts = (
        gs.groupby(["date", "team", "scorer"])
        .size()
        .reset_index(name="n_goals")
    )

    # Per (date, team): total goals and top-scorer goals
    totals   = goal_counts.groupby(["date", "team"])["n_goals"].sum().reset_index(name="total")
    top_goal = goal_counts.groupby(["date", "team"])["n_goals"].max().reset_index(name="top")
    per_match = totals.merge(top_goal, on=["date", "team"])
    per_match["conc"] = per_match["top"] / per_match["total"].clip(lower=1)

    # Build team-match list to attach concentration
    home_tm = matches[["game_id", "date", "home_team", "tournament"]].rename(
        columns={"home_team": "team"}
    )
    away_tm = matches[["game_id", "date", "away_team", "tournament"]].rename(
        columns={"away_team": "team"}
    )
    team_matches = (
        pd.concat([home_tm, away_tm])
        .sort_values(["team", "date"])
        .drop_duplicates(["team", "game_id"])
        .reset_index(drop=True)
    )
    team_matches["_competitive"] = team_matches["game_id"].isin(competitive_ids)

    team_matches = team_matches.merge(
        per_match[["date", "team", "conc"]], on=["date", "team"], how="left"
    )
    # Matches where the team scored 0 goals: conc=0 (no concentration risk)
    team_matches["conc"] = team_matches["conc"].fillna(0.0)

    # Only roll over competitive appearances
    comp_matches = team_matches[team_matches["_competitive"]].copy()
    comp_matches["star_conc"] = comp_matches.groupby("team")["conc"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=5).mean()
    )

    star_conc_map = comp_matches.set_index(["game_id", "team"])["star_conc"]

    # Join home and away
    df = matches.copy()
    df["HOME_STAR_CONC"] = df.apply(
        lambda r: star_conc_map.get((r["game_id"], r["home_team"]), np.nan), axis=1
    )
    df["AWAY_STAR_CONC"] = df.apply(
        lambda r: star_conc_map.get((r["game_id"], r["away_team"]), np.nan), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# Full training feature builder
# ---------------------------------------------------------------------------

def build_training_features(
    url: str,
    goalscorers_url: str,
    min_year: int = 1980,
    filter_wc_and_competitive: bool = False,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download both datasets, compute all features, return training DataFrame.

    Parameters
    ----------
    url : str
        URL of the results CSV.
    goalscorers_url : str
        URL of the goalscorers CSV.
    min_year : int
        Drop rows before this year (keeps ELO warm-up period out of training).
    filter_wc_and_competitive : bool
        If True, keep only World Cup + qualifiers + continental tournaments.
    output_path : str, optional
        Write CSV here if provided.
    """
    import os

    raw = load_data(url)
    goalscorers = load_goalscorers(goalscorers_url)

    logger.info("Computing ELO history …")
    raw = compute_elo_history(raw)

    logger.info("Computing rolling form features …")
    raw = compute_form_history(raw, window=5, competitive_only=True)

    logger.info("Computing EWM attack/defense ratings …")
    raw = compute_attack_defense_ratings(raw, span=7)

    logger.info("Computing player star-concentration …")
    raw = compute_player_concentration(raw, goalscorers, window=20)

    # Training set: drop friendlies, apply year filter
    mask = (~raw["tournament"].str.lower().str.contains("friendly")) & \
           (raw["date"].dt.year >= min_year)
    df = raw[mask].copy()

    if filter_wc_and_competitive:
        keep_kws = ("world cup", "qualifier", "qualification", "euro",
                    "copa america", "africa cup", "asian cup", "gold cup",
                    "nations league", "confederations cup")
        df = df[df["tournament"].str.lower().apply(
            lambda t: any(k in t for k in keep_kws)
        )].copy()

    df["IS_NEUTRAL"] = df["neutral"].astype(int)

    logger.info(
        "Training set: %d matches | %d-%d | "
        "home win %.1f%% | draw %.1f%% | away win %.1f%%",
        len(df),
        df["date"].dt.year.min(), df["date"].dt.year.max(),
        (df["RESULT"] == 2).mean() * 100,
        (df["RESULT"] == 1).mean() * 100,
        (df["RESULT"] == 0).mean() * 100,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %d rows -> %s", len(df), output_path)

    return df


# ---------------------------------------------------------------------------
# Live-scoring helpers
# ---------------------------------------------------------------------------

def get_current_state(
    matches: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Extract the latest per-team snapshot from a fully-computed historical
    DataFrame (must already have ELO, ATK, DEF, form, and star-conc columns).

    Returns
    -------
    current_elo : dict[str, float]
        Post-match ELO for every team after their most recent game.
    latest_stats : pd.DataFrame
        Indexed on ``team``.  Columns used by :func:`build_scoring_row`:
        form, gf, ga, atk, def_r, atk_vs_def, star_conc
    """
    # Recompute final ELO for each team by walking the full history
    elo: Dict[str, float] = {}
    for _, row in matches.sort_values("date").iterrows():
        h, a  = row["home_team"], row["away_team"]
        he, ae = elo.get(h, _INITIAL_ELO), elo.get(a, _INITIAL_ELO)
        k        = _k_factor(row.get("tournament", ""))
        exp_h    = _elo_expected(he, ae)
        actual_h = {2: 1.0, 1: 0.5, 0: 0.0}[row["RESULT"]]
        elo[h]   = he + k * (actual_h       - exp_h)
        elo[a]   = ae + k * ((1 - actual_h) - (1 - exp_h))

    # Latest stats = each team's most-recent appearance in the dataset
    home_side = matches[["date", "home_team",
                          "HOME_FORM_LAST5",         "HOME_GOALS_FOR_LAST5",
                          "HOME_GOALS_AGAINST_LAST5",
                          "HOME_ATK_RATING",          "HOME_DEF_RATING",
                          "HOME_ATK_VS_AWAY_DEF",     "HOME_STAR_CONC"]].copy()
    home_side = home_side.rename(columns={
        "home_team":              "team",
        "HOME_FORM_LAST5":        "form",
        "HOME_GOALS_FOR_LAST5":   "gf",
        "HOME_GOALS_AGAINST_LAST5": "ga",
        "HOME_ATK_RATING":        "atk",
        "HOME_DEF_RATING":        "def_r",
        "HOME_ATK_VS_AWAY_DEF":   "atk_vs_def",
        "HOME_STAR_CONC":         "star_conc",
    })

    away_side = matches[["date", "away_team",
                          "AWAY_FORM_LAST5",         "AWAY_GOALS_FOR_LAST5",
                          "AWAY_GOALS_AGAINST_LAST5",
                          "AWAY_ATK_RATING",          "AWAY_DEF_RATING",
                          "AWAY_ATK_VS_HOME_DEF",     "AWAY_STAR_CONC"]].copy()
    away_side = away_side.rename(columns={
        "away_team":              "team",
        "AWAY_FORM_LAST5":        "form",
        "AWAY_GOALS_FOR_LAST5":   "gf",
        "AWAY_GOALS_AGAINST_LAST5": "ga",
        "AWAY_ATK_RATING":        "atk",
        "AWAY_DEF_RATING":        "def_r",
        "AWAY_ATK_VS_HOME_DEF":   "atk_vs_def",
        "AWAY_STAR_CONC":         "star_conc",
    })

    long = pd.concat([home_side, away_side]).sort_values("date")
    latest_stats = long.groupby("team").last()[
        ["form", "gf", "ga", "atk", "def_r", "atk_vs_def", "star_conc"]
    ]
    return elo, latest_stats


def build_scoring_row(
    home_team: str,
    away_team: str,
    current_elo: Dict[str, float],
    latest_stats: pd.DataFrame,
    is_neutral: bool = True,
) -> Dict[str, float]:
    """
    Build the complete feature dict for one live game.

    Parameters
    ----------
    home_team, away_team : str
        Normalised names (pass through :func:`normalize_name` first).
    current_elo : dict
        From :func:`get_current_state`.
    latest_stats : pd.DataFrame
        From :func:`get_current_state`, indexed on team name.
    """
    h_elo = current_elo.get(home_team, _INITIAL_ELO)
    a_elo = current_elo.get(away_team, _INITIAL_ELO)

    def _stat(team: str, col: str, default: float) -> float:
        if team in latest_stats.index:
            v = latest_stats.at[team, col]
            if not pd.isna(v):
                return float(v)
        return default

    h_atk = _stat(home_team, "atk",    1.2)
    a_atk = _stat(away_team, "atk",    1.2)
    h_def = _stat(home_team, "def_r",  1.2)
    a_def = _stat(away_team, "def_r",  1.2)

    return {
        "HOME_ELO":               h_elo,
        "AWAY_ELO":               a_elo,
        "DELTA_ELO":              h_elo - a_elo,
        "HOME_FORM_LAST5":        _stat(home_team, "form",      1.0),
        "AWAY_FORM_LAST5":        _stat(away_team, "form",      1.0),
        "HOME_GOALS_FOR_LAST5":   _stat(home_team, "gf",        1.2),
        "AWAY_GOALS_FOR_LAST5":   _stat(away_team, "gf",        1.2),
        "HOME_GOALS_AGAINST_LAST5": _stat(home_team, "ga",      1.2),
        "AWAY_GOALS_AGAINST_LAST5": _stat(away_team, "ga",      1.2),
        "HOME_ATK_RATING":        h_atk,
        "AWAY_ATK_RATING":        a_atk,
        "HOME_DEF_RATING":        h_def,
        "AWAY_DEF_RATING":        a_def,
        "HOME_ATK_VS_AWAY_DEF":   h_atk / (a_def + 0.1),
        "AWAY_ATK_VS_HOME_DEF":   a_atk / (h_def + 0.1),
        "HOME_STAR_CONC":         _stat(home_team, "star_conc", 0.35),
        "AWAY_STAR_CONC":         _stat(away_team, "star_conc", 0.35),
        "IS_NEUTRAL":             int(is_neutral),
    }
