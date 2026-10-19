"""Platform-neutral trade and alert records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

POLYMARKET = "polymarket"
KALSHI = "kalshi"
PLATFORMS = (POLYMARKET, KALSHI)


@dataclass(frozen=True)
class Trade:
    """One fill, normalised so the detector never sees platform-specific JSON.

    ``price`` is in dollars per share (0–1), so it doubles as the implied
    probability. ``trader`` is a wallet address on Polymarket and ``None`` on
    Kalshi, whose public trade feed is anonymous.
    """

    platform: str
    trade_id: str
    market_id: str
    market_title: str
    outcome: str
    side: str
    price: float
    size: float
    timestamp: datetime
    trader: Optional[str] = None
    trader_name: Optional[str] = None
    url: Optional[str] = None

    @property
    def notional(self) -> float:
        return self.size * self.price


LARGE_TRADE = "large_trade"
ACCUMULATION = "accumulation"
SWEEP = "sweep"
WATCHED_WALLET = "watched_wallet"


@dataclass
class Alert:
    kind: str
    platform: str
    market_id: str
    market_title: str
    outcome: str
    side: str
    notional: float
    avg_price: float
    trade_count: int
    first_ts: datetime
    last_ts: datetime
    trader: Optional[str] = None
    trader_name: Optional[str] = None
    url: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    trade_ids: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identity, so the same alert is never stored or sent twice."""
        digest = hashlib.sha1(",".join(sorted(self.trade_ids)).encode()).hexdigest()[:16]
        return f"{self.platform}:{self.kind}:{digest}"
