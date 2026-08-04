"""
scripts/pull_training_data.py

Downloads real NBA game data from nba_api and builds a training-ready
feature CSV.  Requires NO odds API key -- everything here is free.

What it pulls
-------------
For each target season (default: 2022-23, 2023-24, 2024-25):
  - LeagueGameLog       -> one row per team per game (result, date, matchup)
  - LeagueDashTeamStats -> season-level advanced metrics (Pace, OffRtg, DefRtg …)

How it avoids look-ahead bias
------------------------------
For a game played in season N, the model uses team stats from season N-1.
This mirrors what you would actually know on game day -- you can look up last
season's ratings but you don't yet have this season's final numbers.

Output
------
  data/training_features.csv  -- one row per game, ready for train_model.py

Usage
-----
  python scripts/pull_training_data.py
  python scripts/pull_training_data.py --seasons 2021-22 2022-23 2023-24 2024-25
  python scripts/pull_training_data.py --dry-run   # show what would be pulled

Expected run time: ~4 minutes (nba_api rate-limit of 20 req/min).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueGameLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull")

# ---------------------------------------------------------------------------
# Rate-limit helper -- nba_api is an unofficial scraper; be polite
# ---------------------------------------------------------------------------

_LAST_CALL: float = 0.0
_MIN_INTERVAL: float = 60.0 / 20   # 20 calls per minute -> 3 s between calls


def _wait() -> None:
    global _LAST_CALL
    elapsed = time.monotonic() - _LAST_CALL
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL = time.monotonic()


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def _prev_season(season: str) -> str:
    """'2024-25' -> '2023-24'"""
    start = int(season[:4])
    return f"{start - 1}-{str(start)[-2:]}"


def _season_list(seasons: List[str]) -> List[str]:
    """Return seasons plus all their previous seasons (needed for lagged stats)."""
    needed = set(seasons)
    for s in seasons:
        needed.add(_prev_season(s))
    return sorted(needed)


# ---------------------------------------------------------------------------
# Step 1: Pull game logs
# ---------------------------------------------------------------------------

def pull_game_logs(season: str) -> pd.DataFrame:
    """
    Pull every regular-season game for *season* from LeagueGameLog.

    Returns one row per HOME team per game with columns:
        GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID,
        HOME_TEAM_ABB, AWAY_TEAM_ABB, HOME_WIN, SEASON
    """
    log.info("  Pulling game log: %s …", season)
    _wait()

    endpoint = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        timeout=30,
    )
    df = endpoint.get_data_frames()[0]
    df.columns = [c.upper() for c in df.columns]

    # Each game appears TWICE -- once per team.
    # "vs." in MATCHUP means the team is HOME.  "@ " means they are AWAY.
    home_rows = df[df["MATCHUP"].str.contains("vs\\.", regex=True)].copy()

    # Parse away team abbreviation from "BOS vs. MIA" -> "MIA"
    home_rows["HOME_TEAM_ABB"] = home_rows["MATCHUP"].str.split(" vs. ").str[0].str.strip()
    home_rows["AWAY_TEAM_ABB"] = home_rows["MATCHUP"].str.split(" vs. ").str[1].str.strip()

    # Build a TEAM_ID lookup from the full log
    team_id_map: Dict[str, int] = (
        df[["TEAM_ABBREVIATION", "TEAM_ID"]]
        .drop_duplicates()
        .set_index("TEAM_ABBREVIATION")["TEAM_ID"]
        .to_dict()
    )
    home_rows["AWAY_TEAM_ID"] = home_rows["AWAY_TEAM_ABB"].map(team_id_map)

    home_rows = home_rows.rename(columns={
        "TEAM_ID":           "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM_ABB_ORIG",
    })

    home_rows["HOME_WIN"] = (home_rows["WL"] == "W").astype(int)
    home_rows["GAME_DATE"] = pd.to_datetime(home_rows["GAME_DATE"])
    home_rows["SEASON"] = season

    cols = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM_ID", "AWAY_TEAM_ID",
        "HOME_TEAM_ABB", "AWAY_TEAM_ABB",
        "HOME_WIN",
    ]
    return home_rows[cols].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Pull team advanced stats
# ---------------------------------------------------------------------------

def pull_team_stats(season: str) -> pd.DataFrame:
    """
    Pull per-season team advanced metrics from LeagueDashTeamStats.

    Returns one row per team with TEAM_ID as index and columns:
        PACE, OFF_RATING, DEF_RATING, NET_RATING,
        AST_PCT, OREB_PCT, DREB_PCT, EFG_PCT, TS_PCT
    """
    log.info("  Pulling team stats: %s …", season)
    _wait()

    endpoint = LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        timeout=30,
    )
    df = endpoint.get_data_frames()[0]
    df.columns = [c.upper() for c in df.columns]

    keep = [
        "TEAM_ID", "TEAM_ABBREVIATION",
        "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_PCT", "OREB_PCT", "DREB_PCT", "EFG_PCT", "TS_PCT",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    df.set_index("TEAM_ID", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Step 3: Build rest-day / back-to-back features from the game log
# ---------------------------------------------------------------------------

def build_schedule_features(all_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute REST_DAYS, IS_B2B, IS_B2B_FIRST for home and away teams.

    Operates on the full multi-season game log so back-to-back detection
    works correctly across season boundaries.

    Parameters
    ----------
    all_games : pd.DataFrame
        Combined output of pull_game_logs() for all seasons.
        Must have GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID, GAME_ID.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 6 new schedule columns appended.
    """
    # Build a flat team -> game_date table to compute rest days
    home_side = all_games[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID"}
    )
    away_side = all_games[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ID"]].rename(
        columns={"AWAY_TEAM_ID": "TEAM_ID"}
    )
    all_appearances = pd.concat([home_side, away_side]).sort_values(
        ["TEAM_ID", "GAME_DATE"]
    )

    all_appearances["PREV_DATE"] = all_appearances.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    all_appearances["REST_DAYS"] = (
        (all_appearances["GAME_DATE"] - all_appearances["PREV_DATE"])
        .dt.days.sub(1)
        .clip(upper=7)
        .fillna(7)
        .astype(int)
    )
    all_appearances["IS_B2B"] = (all_appearances["REST_DAYS"] == 1).astype(int)
    # IS_B2B_FIRST: today's game is the FIRST leg -- tomorrow they play again
    all_appearances["NEXT_DATE"] = all_appearances.groupby("TEAM_ID")["GAME_DATE"].shift(-1)
    all_appearances["IS_B2B_FIRST"] = (
        (all_appearances["NEXT_DATE"] - all_appearances["GAME_DATE"]).dt.days == 1
    ).fillna(False).astype(int)

    sched = all_appearances[
        ["GAME_ID", "TEAM_ID", "REST_DAYS", "IS_B2B", "IS_B2B_FIRST"]
    ].copy()

    # Join home side
    df = all_games.merge(
        sched.rename(columns={
            "TEAM_ID":       "HOME_TEAM_ID",
            "REST_DAYS":     "HOME_REST_DAYS",
            "IS_B2B":        "HOME_IS_B2B",
            "IS_B2B_FIRST":  "HOME_IS_B2B_FIRST",
        }),
        on=["GAME_ID", "HOME_TEAM_ID"],
        how="left",
    )
    # Join away side
    df = df.merge(
        sched.rename(columns={
            "TEAM_ID":       "AWAY_TEAM_ID",
            "REST_DAYS":     "AWAY_REST_DAYS",
            "IS_B2B":        "AWAY_IS_B2B",
            "IS_B2B_FIRST":  "AWAY_IS_B2B_FIRST",
        }),
        on=["GAME_ID", "AWAY_TEAM_ID"],
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# Step 4: Merge lagged team stats onto game rows
# ---------------------------------------------------------------------------

_STAT_COLS = [
    "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "OREB_PCT", "DREB_PCT", "EFG_PCT", "TS_PCT",
]


def merge_lagged_stats(
    games: pd.DataFrame,
    all_team_stats: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    For each game in season N, attach team ratings from season N-1.

    If season N-1 stats are not available (e.g. expansion team or first
    season in the dataset), fall back to the league mean for that season.

    Parameters
    ----------
    games : pd.DataFrame
        Game log with SEASON, HOME_TEAM_ID, AWAY_TEAM_ID columns.
    all_team_stats : dict[str, pd.DataFrame]
        Map of season string -> team stats DataFrame (from pull_team_stats).

    Returns
    -------
    pd.DataFrame
        Games with DELTA_* columns and raw home/away stat columns appended.
    """
    df = games.copy()
    for col in _STAT_COLS:
        df[f"HOME_{col}"] = float("nan")
        df[f"AWAY_{col}"] = float("nan")

    for season, group in df.groupby("SEASON"):
        prev = _prev_season(season)
        stats = all_team_stats.get(prev)
        if stats is None:
            log.warning("  No lagged stats for %s (needed for %s) -- using zeros.", prev, season)
            continue

        league_mean = stats[_STAT_COLS].mean()

        for idx in group.index:
            for side, id_col in (("HOME", "HOME_TEAM_ID"), ("AWAY", "AWAY_TEAM_ID")):
                tid = df.at[idx, id_col]
                if tid in stats.index:
                    for col in _STAT_COLS:
                        df.at[idx, f"{side}_{col}"] = stats.at[tid, col]
                else:
                    # New/relocated team -- use league average
                    for col in _STAT_COLS:
                        df.at[idx, f"{side}_{col}"] = league_mean[col]

    for col in _STAT_COLS:
        df[f"DELTA_{col}"] = df[f"HOME_{col}"] - df[f"AWAY_{col}"]

    return df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_training_csv(
    seasons: List[str],
    output_path: str,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Full pipeline: pull -> schedule -> lagged stats -> save CSV.

    Parameters
    ----------
    seasons : list[str]
        Target seasons to include as training rows, e.g. ["2023-24", "2024-25"].
        The previous season of each is also pulled for lagged stats.
    output_path : str
        Where to write the CSV.
    dry_run : bool
        If True, print what would be pulled and exit without making API calls.

    Returns
    -------
    pd.DataFrame or None
    """
    all_seasons_needed = _season_list(seasons)

    if dry_run:
        log.info("DRY RUN -- would pull the following seasons:")
        log.info("  Game logs (target):   %s", seasons)
        log.info("  Team stats (all):     %s", all_seasons_needed)
        log.info("  Estimated API calls:  %d", len(seasons) + len(all_seasons_needed))
        log.info("  Estimated time:       ~%d seconds", (len(seasons) + len(all_seasons_needed)) * 4)
        return None

    # --- Pull game logs for target seasons ---
    log.info("Step 1/4: Pulling game logs …")
    game_frames = []
    for s in seasons:
        try:
            gf = pull_game_logs(s)
            game_frames.append(gf)
            log.info("    %s: %d games", s, len(gf))
        except Exception as exc:
            log.error("    FAILED %s: %s", s, exc)

    if not game_frames:
        log.error("No game data retrieved. Check your internet connection.")
        sys.exit(1)

    all_games = pd.concat(game_frames, ignore_index=True)
    log.info("  Total games across all seasons: %d", len(all_games))

    # --- Pull team stats for all seasons (including lagged) ---
    log.info("Step 2/4: Pulling team advanced stats …")
    all_team_stats: Dict[str, pd.DataFrame] = {}
    for s in all_seasons_needed:
        try:
            ts = pull_team_stats(s)
            all_team_stats[s] = ts
            log.info("    %s: %d teams", s, len(ts))
        except Exception as exc:
            log.error("    FAILED %s: %s", s, exc)

    # --- Schedule features ---
    log.info("Step 3/4: Computing rest-day / back-to-back features …")
    all_games = build_schedule_features(all_games)

    # --- Lagged team stats ---
    log.info("Step 4/4: Merging lagged team stats (season N-1 ratings) …")
    all_games = merge_lagged_stats(all_games, all_team_stats)

    # Drop rows where we couldn't get stats for either team
    before = len(all_games)
    all_games = all_games.dropna(subset=["DELTA_NET_RATING"])
    dropped = before - len(all_games)
    if dropped:
        log.warning("  Dropped %d rows with missing stats.", dropped)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    all_games.to_csv(output_path, index=False)
    log.info("Saved %d rows -> %s", len(all_games), output_path)

    # Quick sanity print
    log.info("Home-win rate: %.3f (expect 0.54-0.58)", all_games["HOME_WIN"].mean())
    log.info("Seasons:       %s", all_games["SEASON"].unique().tolist())
    log.info("Date range:    %s -> %s",
             all_games["GAME_DATE"].min().date(),
             all_games["GAME_DATE"].max().date())

    return all_games


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull real NBA data and build training_features.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25"],
        help="Target seasons to include as training rows.",
    )
    p.add_argument(
        "--output",
        default="data/training_features.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pulled without making any API calls.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_training_csv(
        seasons=args.seasons,
        output_path=args.output,
        dry_run=args.dry_run,
    )
