"""Trader scorecards: is this wallet good, or just big or lucky?

A position counts as a win if it made money. Win rate alone is a trap: someone
who only buys 95¢ favourites wins 95% of the time and barely profits, while a
40% hit rate on 20¢ longshots is excellent. So every scorecard also reports
the **edge**: win rate minus the average price paid (the win rate the market
expected of them). A trader is only worth copying if they beat the odds they
bought at, by a margin that survives a sample-size check (Wilson lower bound).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from copy_trader.config import CopyConfig
from copy_trader.polymarket import ClosedPosition, Position


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    """95% lower confidence bound on a win rate: 9/10 scores well below 90/100."""
    if n == 0:
        return 0.0
    p = wins / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


@dataclass
class Scorecard:
    wallet: str
    name: str = ""
    resolved: int = 0
    wins: int = 0
    win_rate: float = 0.0
    win_rate_low: float = 0.0
    implied_win_rate: float = 0.0
    edge: float = 0.0
    realized_pnl: float = 0.0
    roi: float = 0.0
    profit: float = 0.0
    open_positions: int = 0
    open_value: float = 0.0
    last_active: Optional[datetime] = None
    qualifies: bool = False
    failures: List[str] = field(default_factory=list)

    @property
    def conservative_edge(self) -> float:
        """Edge using the win-rate lower bound; what the ranking sorts on."""
        return self.win_rate_low - self.implied_win_rate

    def summary(self) -> str:
        return (
            f"{self.win_rate:.0%} win rate over {self.resolved} resolved, "
            f"{self.edge * 100:+.0f} pts vs prices paid, ${self.profit:,.0f} profit"
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["last_active"] = self.last_active.isoformat() if self.last_active else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Scorecard":
        data = dict(data)
        if data.get("last_active"):
            data["last_active"] = datetime.fromisoformat(data["last_active"])
        return cls(**data)


def score_trader(
    wallet: str,
    closed: Iterable[ClosedPosition],
    positions: Iterable[Position],
    config: CopyConfig,
    name: str = "",
    leaderboard_pnl: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Scorecard:
    now = now or datetime.now(timezone.utc)
    positions = list(positions)
    # Each record: (price paid, profit, cost)
    records = [(c.avg_price, c.realized_pnl, c.cost) for c in closed]
    # Resolved-but-unredeemed holdings are finished bets too.
    records += [(p.avg_price, p.cash_pnl, p.cost) for p in positions if p.resolved]
    open_positions = [p for p in positions if not p.resolved]

    card = Scorecard(wallet=wallet, name=name)
    card.resolved = len(records)
    card.wins = sum(1 for _, pnl, _ in records if pnl > 0)
    if records:
        card.win_rate = card.wins / card.resolved
        card.win_rate_low = wilson_lower(card.wins, card.resolved)
        card.implied_win_rate = sum(price for price, _, _ in records) / card.resolved
        card.edge = card.win_rate - card.implied_win_rate
    card.realized_pnl = sum(pnl for _, pnl, _ in records)
    cost = sum(c for _, _, c in records)
    card.roi = card.realized_pnl / cost if cost else 0.0
    card.profit = leaderboard_pnl if leaderboard_pnl is not None else card.realized_pnl
    card.open_positions = len(open_positions)
    card.open_value = sum(p.value for p in open_positions)
    stamps = [c.timestamp for c in closed if c.timestamp]
    card.last_active = max(stamps) if stamps else None
    if open_positions and (card.last_active is None or card.last_active < now - timedelta(days=1)):
        # Holding open positions counts as active even with no recent closes.
        card.last_active = now

    if card.resolved < config.min_resolved:
        card.failures.append(f"only {card.resolved} resolved positions (need {config.min_resolved})")
    if card.win_rate < config.min_win_rate:
        card.failures.append(f"win rate {card.win_rate:.0%} < {config.min_win_rate:.0%}")
    if card.edge < config.min_edge:
        card.failures.append(f"edge {card.edge * 100:+.1f} pts < {config.min_edge * 100:+.0f} (wins mostly priced in)")
    if card.profit < config.min_profit_usd:
        card.failures.append(f"profit ${card.profit:,.0f} < ${config.min_profit_usd:,.0f}")
    if card.last_active is None or card.last_active < now - timedelta(days=config.max_idle_days):
        card.failures.append(f"inactive for over {config.max_idle_days} days")
    card.qualifies = not card.failures
    return card


def rank(cards: Iterable[Scorecard]) -> List[Scorecard]:
    """Qualifying traders first, then by conservative edge, then profit."""
    return sorted(cards, key=lambda c: (not c.qualifies, -c.conservative_edge, -c.profit))
