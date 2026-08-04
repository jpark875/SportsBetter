"""
core/parlay.py

Parlay expected-value and Kelly analysis.

The research finding this module encodes: a parlay multiplies both your
edge *and* the book's vig across legs.  A 2-leg parlay carries roughly a
20% hold versus ~4-5% on a straight bet, and it compounds with every
extra leg.  So a parlay is only worth making when *every* leg is
independently +EV by enough that the product still clears 1.0 — and even
then, betting the legs straight usually yields more expected profit at
far lower variance.

This module therefore never just prices a parlay; it prices the parlay
*and* the straight-bet alternative and returns a recommendation.

Independence caveat: combined probability is the product of leg
probabilities, valid only when legs are independent (different games).
Same-game legs are correlated — sportsbooks price that in on same-game
parlays, and our independence assumption will misestimate them — so
``analyze_parlay`` flags any parlay whose legs share a game_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ParlayLeg:
    label: str          # "Lakers ML", "Arsenal ML", …
    game_id: str
    true_prob: float    # model's calibrated P(win) for this leg
    decimal_odds: float # offered decimal price for this leg


@dataclass
class ParlayAnalysis:
    n_legs: int
    combined_true_prob: float
    combined_decimal: float          # what the book pays (product of leg decimals)
    fair_decimal: float              # 1 / combined_true_prob (zero-vig fair price)
    edge: float                      # EV per unit staked on the parlay
    kelly_fraction: float            # full-Kelly stake as a fraction of bankroll
    straight_total_edge: float       # summed EV if each leg were bet straight instead
    correlated: bool                 # any two legs share a game (independence broken)
    recommendation: str
    verdict: str = field(default="")  # "parlay" | "straight" | "pass"


def _kelly(prob: float, decimal: float) -> float:
    """Full-Kelly fraction for a single wager; 0 if non-positive edge."""
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    return max(0.0, (b * prob - q) / b)


def analyze_parlay(legs: List[ParlayLeg]) -> ParlayAnalysis:
    """
    Analyze a proposed parlay and recommend parlay vs. straight vs. pass.

    Parameters
    ----------
    legs : list[ParlayLeg]
        At least two legs.

    Returns
    -------
    ParlayAnalysis
    """
    if len(legs) < 2:
        raise ValueError("A parlay needs at least two legs.")

    combined_prob = float(np.prod([leg.true_prob for leg in legs]))
    combined_decimal = float(np.prod([leg.decimal_odds for leg in legs]))
    fair_decimal = 1.0 / combined_prob if combined_prob > 0 else float("inf")
    edge = combined_prob * combined_decimal - 1.0
    kelly = _kelly(combined_prob, combined_decimal)

    # Straight-bet alternative: sum of per-leg edges (each leg staked once).
    straight_total_edge = sum(
        leg.true_prob * leg.decimal_odds - 1.0 for leg in legs
    )

    game_ids = [leg.game_id for leg in legs]
    correlated = len(set(game_ids)) < len(game_ids)

    verdict, recommendation = _recommend(
        edge, straight_total_edge, kelly, correlated, len(legs)
    )

    return ParlayAnalysis(
        n_legs=len(legs),
        combined_true_prob=combined_prob,
        combined_decimal=combined_decimal,
        fair_decimal=fair_decimal,
        edge=edge,
        kelly_fraction=kelly,
        straight_total_edge=straight_total_edge,
        correlated=correlated,
        recommendation=recommendation,
        verdict=verdict,
    )


def _recommend(
    parlay_edge: float,
    straight_edge: float,
    kelly: float,
    correlated: bool,
    n_legs: int,
) -> tuple[str, str]:
    if correlated:
        return (
            "pass",
            "Two or more legs are from the same game, so they are correlated. "
            "This tool assumes independent legs and will misprice a same-game "
            "parlay — the sportsbook has already adjusted its payout for the "
            "correlation. Avoid unless you can model the joint probability.",
        )
    if parlay_edge <= 0:
        return (
            "pass",
            f"Negative expected value ({parlay_edge:+.1%}). The compounded vig "
            f"across {n_legs} legs outweighs the individual edges. Do not place "
            "this parlay.",
        )
    # Parlay is +EV. Straight bets capture edge at far lower variance, so we
    # only favor the parlay when its single-ticket edge genuinely exceeds the
    # combined straight edge (rare — happens with several strong legs).
    if parlay_edge > straight_edge:
        return (
            "parlay",
            f"Positive expected value ({parlay_edge:+.1%}) that even exceeds the "
            f"combined straight-bet edge. Stake at most the fractional-Kelly "
            f"amount ({kelly:.1%} of bankroll) — parlay variance is high.",
        )
    return (
        "straight",
        f"The parlay is +EV ({parlay_edge:+.1%}), but betting these {n_legs} legs "
        f"straight yields more expected profit ({straight_edge:+.1%} summed) at "
        "much lower variance. Prefer straight bets for steadier growth.",
    )
