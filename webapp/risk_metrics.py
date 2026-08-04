"""
webapp/risk_metrics.py

Turns a settled-bet ledger into the risk-adjusted performance metrics
that matter for a bettor optimising for *steady* growth rather than
raw hit rate.

Metrics (per the risk-analysis research):

    ROI              net profit / total staked
    Win rate         wins / settled bets
    Sharpe ratio     mean per-bet return / stdev of returns
                     — reward per unit of *total* volatility
    Sortino ratio    mean per-bet return / downside deviation
                     — reward per unit of *downside* volatility; ignores
                       upside swings, which a bettor doesn't mind
    Max drawdown     largest peak-to-trough drop of the bankroll curve,
                     as a fraction of the running peak — the single most
                     important survival metric
    Calmar ratio     total return / max drawdown — worst-case efficiency

Per-bet "return" is P&L divided by stake, so a won +150 bet returns
+1.5 and any loss returns -1.0.  These are unitless and comparable
across differently-sized wagers.

A ``stability_grade`` folds Sortino and max-drawdown into a single
plain-language rating so the UI can say something honest without
demanding the user read four ratios.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class RiskMetrics:
    n_bets: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_staked: float = 0.0
    net_profit: float = 0.0
    roi: float = 0.0
    avg_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0          # fraction in [0, 1]
    max_drawdown_dollars: float = 0.0
    calmar: float = 0.0
    stability_grade: str = "N/A"
    stability_note: str = "No settled bets yet."

    def as_dict(self) -> Dict:
        return asdict(self)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _downside_deviation(xs: List[float], target: float = 0.0) -> float:
    """RMS of returns that fall below ``target`` (the Sortino denominator)."""
    below = [min(0.0, x - target) for x in xs]
    if len(below) < 2:
        return 0.0
    return math.sqrt(sum(d ** 2 for d in below) / (len(below) - 1))


def _max_drawdown(bankroll_curve: List[float]) -> tuple[float, float]:
    """
    Largest peak-to-trough decline of a cumulative-P&L curve.

    Returns ``(fraction, dollars)``. The fraction is relative to the
    running peak *bankroll level*; we anchor the curve at an implied
    starting bankroll of the total staked so an all-losing start still
    yields a sane (<100%) percentage rather than dividing by ~zero.
    """
    if not bankroll_curve:
        return 0.0, 0.0
    peak = bankroll_curve[0]
    max_dd_frac = 0.0
    max_dd_dollars = 0.0
    for value in bankroll_curve:
        peak = max(peak, value)
        drop = peak - value
        if drop > max_dd_dollars:
            max_dd_dollars = drop
        if peak > 0:
            max_dd_frac = max(max_dd_frac, drop / peak)
    return max_dd_frac, max_dd_dollars


def _grade(sortino: float, max_dd: float, n: int) -> tuple[str, str]:
    """Fold Sortino + drawdown into a plain-language stability rating."""
    if n < 10:
        return "Provisional", (
            f"Only {n} settled bet(s) — too few to judge stability. "
            "Metrics stabilise after ~30 bets."
        )
    if max_dd >= 0.30:
        return "Volatile", (
            f"Peak drawdown hit {max_dd:.0%}. Above ~25% is where bankroll "
            "risk becomes dangerous — consider lowering your volatility tolerance."
        )
    if sortino >= 1.0 and max_dd < 0.15:
        return "Stable", (
            "Strong downside-risk-adjusted returns with a shallow drawdown. "
            "This is the profile you want for steady growth."
        )
    if sortino >= 0.3:
        return "Moderate", (
            "Positive risk-adjusted returns, but with meaningful swings. "
            "Sustainable if you stay disciplined on stake sizing."
        )
    return "Underwater", (
        "Downside-adjusted returns are weak or negative. Re-examine which "
        "edges you're actually taking before increasing exposure."
    )


def compute_metrics(settled_bets: List[Dict]) -> RiskMetrics:
    """
    Compute the full metric set from settled (won/lost) bets, oldest-first.

    Each bet dict needs ``stake``, ``pnl``, and ``status``.
    """
    if not settled_bets:
        return RiskMetrics()

    returns: List[float] = []      # per-bet P&L / stake
    bankroll_curve: List[float] = []
    cumulative = float(sum(b["stake"] for b in settled_bets))  # implied start
    starting = cumulative
    wins = 0
    net_profit = 0.0
    total_staked = 0.0

    for bet in settled_bets:
        stake = float(bet["stake"])
        pnl = float(bet.get("pnl") or 0.0)
        total_staked += stake
        net_profit += pnl
        if stake > 0:
            returns.append(pnl / stake)
        if bet["status"] == "won":
            wins += 1
        cumulative += pnl
        bankroll_curve.append(cumulative)

    n = len(settled_bets)
    losses = n - wins
    avg_return = _mean(returns)
    sd = _stdev(returns)
    dd_dev = _downside_deviation(returns)
    sharpe = avg_return / sd if sd > 0 else 0.0
    sortino = avg_return / dd_dev if dd_dev > 0 else (
        float("inf") if avg_return > 0 else 0.0
    )
    max_dd_frac, max_dd_dollars = _max_drawdown([starting] + bankroll_curve)
    total_return = net_profit / starting if starting > 0 else 0.0
    calmar = total_return / max_dd_frac if max_dd_frac > 0 else (
        float("inf") if total_return > 0 else 0.0
    )
    grade, note = _grade(sortino if math.isfinite(sortino) else 5.0, max_dd_frac, n)

    # Clamp non-finite sentinels to something JSON/round-friendly for the UI.
    def _clean(x: float) -> float:
        return round(x, 4) if math.isfinite(x) else 99.99

    return RiskMetrics(
        n_bets=n,
        wins=wins,
        losses=losses,
        win_rate=round(wins / n, 4),
        total_staked=round(total_staked, 2),
        net_profit=round(net_profit, 2),
        roi=round(net_profit / total_staked, 4) if total_staked > 0 else 0.0,
        avg_return=round(avg_return, 4),
        sharpe=_clean(sharpe),
        sortino=_clean(sortino),
        max_drawdown=round(max_dd_frac, 4),
        max_drawdown_dollars=round(max_dd_dollars, 2),
        calmar=_clean(calmar),
        stability_grade=grade,
        stability_note=note,
    )


def bankroll_curve(settled_bets: List[Dict]) -> List[Dict]:
    """
    Cumulative-P&L series for charting, oldest-first.

    Returns a list of ``{"n": index, "pnl": cumulative_pnl}`` points,
    starting at the zero point before any bet settled.
    """
    points = [{"n": 0, "pnl": 0.0}]
    cumulative = 0.0
    for i, bet in enumerate(settled_bets, start=1):
        cumulative += float(bet.get("pnl") or 0.0)
        points.append({"n": i, "pnl": round(cumulative, 2)})
    return points
