"""
portfolio_manager/arbitrage.py

True arbitrage scanner for soccer 1X2 (home / draw / away) markets.

An arbitrage exists when the sum of reciprocal decimal odds across all three
outcomes falls below 1.0.  Betting proportionally on every outcome then
guarantees a profit regardless of which team wins.

Example
-------
    Book A: Home win  +200  (decimal 3.00)   1/3.00 = 0.333
    Book B: Draw      +350  (decimal 4.50)   1/4.50 = 0.222
    Book C: Away win  +180  (decimal 2.80)   1/2.80 = 0.357
    Sum = 0.912  →  guaranteed profit of 8.8% of stakes

Usage
-----
    from portfolio_manager.arbitrage import scan_for_arbitrage, print_arb_report

    arb_df = scan_for_arbitrage(odds_df)
    print_arb_report(arb_df, budget=1000.0)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _american_to_decimal(american: float) -> float:
    """Convert American odds to decimal (European) odds."""
    if american >= 100:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def scan_for_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan a live odds DataFrame for guaranteed arbitrage opportunities.

    For each game, finds the *best* (highest) decimal price available
    across all bookmakers for each of the three 1X2 outcomes, then
    checks whether the implied probability sum is below 1.0.

    Parameters
    ----------
    odds_df : pd.DataFrame
        Must include columns: ``game_id, home_team, away_team, bet_type,
        outcome_name, bookmaker, price`` where ``price`` is in American format.
        Produced by :meth:`data_pipeline.odds_client.OddsClient.get_game_odds`.

    Returns
    -------
    pd.DataFrame
        One row per arbitrage opportunity with columns:
        ``game_id, home_team, away_team,
        home_book, home_american, home_decimal,
        draw_book, draw_american, draw_decimal,
        away_book, away_american, away_decimal,
        arb_margin, profit_pct``

        An empty DataFrame means no arbitrage was detected.
    """
    if odds_df.empty or "price" not in odds_df.columns:
        return pd.DataFrame()

    h2h = odds_df[odds_df["bet_type"] == "moneyline"].copy()
    if h2h.empty:
        return pd.DataFrame()

    h2h["decimal"] = h2h["price"].apply(_american_to_decimal)

    records = []
    for gid, group in h2h.groupby("game_id"):
        meta = group.iloc[0]
        home_name = str(meta.get("home_team", ""))
        away_name = str(meta.get("away_team", ""))

        # Identify outcome rows by fuzzy name match
        def _best(mask_fn) -> Optional[pd.Series]:
            sub = group[group["outcome_name"].apply(mask_fn)]
            if sub.empty:
                return None
            return sub.loc[sub["decimal"].idxmax()]

        home_row = _best(lambda n: (
            home_name.lower() in n.lower() or n.lower() in home_name.lower()
        ) if home_name else False)

        draw_row = _best(lambda n: "draw" in n.lower())

        away_row = _best(lambda n: (
            away_name.lower() in n.lower() or n.lower() in away_name.lower()
        ) if away_name else False)

        if home_row is None or draw_row is None or away_row is None:
            continue

        arb_margin = (
            1.0 / home_row["decimal"]
            + 1.0 / draw_row["decimal"]
            + 1.0 / away_row["decimal"]
        )

        if arb_margin < 1.0:
            records.append({
                "game_id":       gid,
                "home_team":     home_name,
                "away_team":     away_name,
                "home_book":     home_row["bookmaker"],
                "home_american": home_row["price"],
                "home_decimal":  home_row["decimal"],
                "draw_book":     draw_row["bookmaker"],
                "draw_american": draw_row["price"],
                "draw_decimal":  draw_row["decimal"],
                "away_book":     away_row["bookmaker"],
                "away_american": away_row["price"],
                "away_decimal":  away_row["decimal"],
                "arb_margin":    arb_margin,
                "profit_pct":    (1.0 / arb_margin - 1.0) * 100.0,
            })

    return pd.DataFrame(records)


def compute_arb_stakes(arb_row: pd.Series, budget: float) -> dict:
    """
    Given one arbitrage row and a total budget, return the exact stake
    for each outcome that guarantees equal profit regardless of result.

    Parameters
    ----------
    arb_row : pd.Series
        Single row from :func:`scan_for_arbitrage` output.
    budget : float
        Total amount to deploy across all three bets.

    Returns
    -------
    dict with keys ``home_stake, draw_stake, away_stake, guaranteed_return``
    """
    m = float(arb_row["arb_margin"])
    home_stake = budget / (m * float(arb_row["home_decimal"]))
    draw_stake = budget / (m * float(arb_row["draw_decimal"]))
    away_stake = budget / (m * float(arb_row["away_decimal"]))
    guaranteed_return = budget / m
    return {
        "home_stake":        round(home_stake, 2),
        "draw_stake":        round(draw_stake, 2),
        "away_stake":        round(away_stake, 2),
        "guaranteed_return": round(guaranteed_return, 2),
        "guaranteed_profit": round(guaranteed_return - budget, 2),
    }


def scan_h2h_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sport-agnostic arbitrage scanner for head-to-head markets with any
    number of outcomes (2-way US sports, 3-way soccer).

    For each game, takes the best available decimal price per distinct
    outcome across all books; if the reciprocals sum below 1.0, betting
    every outcome proportionally locks in profit.

    Parameters
    ----------
    odds_df : pd.DataFrame
        Canonical odds schema with ``game_id, home_team, away_team,
        bet_type, outcome_name, bookmaker, price`` (American odds).

    Returns
    -------
    pd.DataFrame
        One row per opportunity: ``game_id, home_team, away_team,
        arb_margin, profit_pct, legs`` where ``legs`` is a list of dicts
        ``{outcome, bookmaker, american, decimal, stake_fraction}``.
        ``stake_fraction`` is the share of total budget for that leg.
    """
    if odds_df.empty or "price" not in odds_df.columns:
        return pd.DataFrame()

    h2h = odds_df[odds_df["bet_type"] == "moneyline"].copy()
    if h2h.empty:
        return pd.DataFrame()
    h2h["decimal"] = h2h["price"].apply(_american_to_decimal)

    records = []
    for gid, group in h2h.groupby("game_id"):
        # The h2h market's own outcome set defines the legs (2 or 3).
        best = group.loc[group.groupby("outcome_name")["decimal"].idxmax()]
        if len(best) < 2:
            continue

        arb_margin = float((1.0 / best["decimal"]).sum())
        if arb_margin >= 1.0:
            continue

        meta = group.iloc[0]
        legs = [
            {
                "outcome": leg["outcome_name"],
                "bookmaker": leg["bookmaker"],
                "american": float(leg["price"]),
                "decimal": float(leg["decimal"]),
                "stake_fraction": 1.0 / (arb_margin * float(leg["decimal"])),
            }
            for _, leg in best.iterrows()
        ]
        records.append({
            "game_id": gid,
            "home_team": str(meta.get("home_team", "")),
            "away_team": str(meta.get("away_team", "")),
            "arb_margin": arb_margin,
            "profit_pct": (1.0 / arb_margin - 1.0) * 100.0,
            "legs": legs,
        })

    return pd.DataFrame(records)


def print_arb_report(arb_df: pd.DataFrame, budget: float = 100.0) -> None:
    """
    Pretty-print all detected arbitrage opportunities and per-outcome stakes.

    Parameters
    ----------
    arb_df : pd.DataFrame
        Output of :func:`scan_for_arbitrage`.
    budget : float
        Hypothetical per-game budget used to illustrate stake sizing.
    """
    sep = "=" * 72

    print(f"\n{sep}")
    print("  ARBITRAGE SCANNER — GUARANTEED PROFIT OPPORTUNITIES")
    print(sep)

    if arb_df.empty:
        print("  No true arbitrage detected across today's slate.")
        print(f"{sep}\n")
        return

    print(f"  {len(arb_df)} arb opportunity(ies) found  "
          f"(budget per game: ${budget:.0f})\n")

    for _, row in arb_df.iterrows():
        stakes = compute_arb_stakes(row, budget)
        print(f"  {row['home_team']} vs {row['away_team']}")
        print(f"  Margin: {row['arb_margin']:.4f}  |  "
              f"Guaranteed profit: {row['profit_pct']:.2f}%")
        print(f"    Home win  @ {row['home_american']:+.0f} ({row['home_book']})  "
              f"→ stake ${stakes['home_stake']:.2f}")
        print(f"    Draw      @ {row['draw_american']:+.0f} ({row['draw_book']})  "
              f"→ stake ${stakes['draw_stake']:.2f}")
        print(f"    Away win  @ {row['away_american']:+.0f} ({row['away_book']})  "
              f"→ stake ${stakes['away_stake']:.2f}")
        print(f"    Return: ${stakes['guaranteed_return']:.2f}  "
              f"(+${stakes['guaranteed_profit']:.2f})")
        print()

    print(sep + "\n")
