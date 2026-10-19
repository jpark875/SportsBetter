"""Polymarket's public Data API: leaderboard, open positions and closed positions.

Every Polymarket trader is a public wallet, so a trader's full record (what
they hold, what they paid, what they won) is readable without a key. Kalshi's
public API has no trader identities, which is why this tool is Polymarket only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DATA_API = "https://data-api.polymarket.com"
SITE = "https://polymarket.com"
POSITIONS_PAGE = 500
CLOSED_PAGE = 50
_RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Position:
    """An open (or resolved but unredeemed) holding of one outcome token."""

    asset: str
    condition_id: str
    title: str
    outcome: str
    size: float
    avg_price: float
    cur_price: float
    cash_pnl: float = 0.0
    redeemable: bool = False
    slug: str = ""
    end_date: str = ""

    @property
    def cost(self) -> float:
        return self.size * self.avg_price

    @property
    def value(self) -> float:
        return self.size * self.cur_price

    @property
    def url(self) -> str:
        return f"{SITE}/event/{self.slug}" if self.slug else SITE

    @property
    def resolved(self) -> bool:
        return self.redeemable or self.cur_price >= 0.995 or self.cur_price <= 0.005


@dataclass(frozen=True)
class ClosedPosition:
    title: str
    outcome: str
    avg_price: float
    shares: float
    realized_pnl: float
    timestamp: Optional[datetime]

    @property
    def cost(self) -> float:
        return self.avg_price * self.shares


def _f(raw: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return float(value)
    return default


def _ts(raw: Any) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    if isinstance(raw, str) and not raw.replace(".", "").isdigit():
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    value = float(raw)
    return datetime.fromtimestamp(value / 1000 if value > 1e12 else value, tz=timezone.utc)


def parse_position(raw: Dict[str, Any]) -> Optional[Position]:
    try:
        return Position(
            asset=str(raw["asset"]),
            condition_id=str(raw.get("conditionId") or ""),
            title=raw.get("title") or "",
            outcome=str(raw.get("outcome") or ""),
            size=float(raw["size"]),
            avg_price=_f(raw, "avgPrice"),
            cur_price=_f(raw, "curPrice"),
            cash_pnl=_f(raw, "cashPnl"),
            redeemable=bool(raw.get("redeemable")),
            slug=raw.get("eventSlug") or raw.get("slug") or "",
            end_date=str(raw.get("endDate") or ""),
        )
    except (KeyError, TypeError, ValueError):
        logger.debug("Skipping malformed position: %r", raw)
        return None


def parse_closed(raw: Dict[str, Any]) -> Optional[ClosedPosition]:
    try:
        return ClosedPosition(
            title=raw.get("title") or "",
            outcome=str(raw.get("outcome") or ""),
            avg_price=float(raw["avgPrice"]),
            shares=_f(raw, "totalBought", "size"),
            realized_pnl=float(raw["realizedPnl"]),
            timestamp=_ts(raw.get("timestamp") or raw.get("endDate")),
        )
    except (KeyError, TypeError, ValueError):
        logger.debug("Skipping malformed closed position: %r", raw)
        return None


def profile_url(wallet: str) -> str:
    return f"{SITE}/profile/{wallet}"


class PolymarketClient:
    def __init__(self, session: Optional[requests.Session] = None, base_url: str = DATA_API, sleep=time.sleep) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"Accept": "application/json", "User-Agent": "copy-trader/0.1"})
        self._base = base_url.rstrip("/")
        self._sleep = sleep

    def _get(self, path: str, params: Dict[str, Any], retries: int = 3) -> Any:
        url = f"{self._base}{path}"
        for attempt in range(retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=15)
            except requests.RequestException:
                if attempt == retries:
                    raise
                self._sleep(2 ** attempt)
                continue
            if response.status_code in _RETRY_STATUS and attempt < retries:
                self._sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("data", data)
        raise RuntimeError("unreachable")

    def leaderboard(self, period: str = "ALL", order_by: str = "PNL", limit: int = 50) -> List[Dict[str, Any]]:
        """``[{"wallet", "name", "pnl", "volume", "rank"}]`` best first."""
        rows = self._get("/v1/leaderboard", {
            "timePeriod": period.upper(), "orderBy": order_by.upper(), "category": "OVERALL", "limit": limit,
        })
        leaders = []
        for i, row in enumerate(rows or [], start=1):
            wallet = (row.get("proxyWallet") or row.get("user") or "").lower()
            if wallet:
                leaders.append({
                    "wallet": wallet,
                    "name": row.get("userName") or row.get("name") or "",
                    "pnl": _f(row, "pnl"),
                    "volume": _f(row, "vol", "volume"),
                    "rank": int(row.get("rank") or i),
                })
        return leaders

    def positions(self, wallet: str, max_pages: int = 4) -> List[Position]:
        """Everything ``wallet`` currently holds. Raises on any fetch error,
        because a partial list would look like the trader had sold."""
        out: List[Position] = []
        for page in range(max_pages):
            rows = self._get("/positions", {
                "user": wallet, "limit": POSITIONS_PAGE, "offset": page * POSITIONS_PAGE, "sizeThreshold": 0.1,
            })
            out.extend(p for p in (parse_position(r) for r in rows or []) if p)
            if len(rows or []) < POSITIONS_PAGE:
                return out
        raise RuntimeError(f"{wallet} holds more than {max_pages * POSITIONS_PAGE} positions; refusing a partial list")

    def closed_positions(self, wallet: str, limit: int = 500) -> List[ClosedPosition]:
        """The most recent ``limit`` closed positions.

        Sorted by time on purpose: the API can sort by profit, and scoring only
        someone's biggest wins would inflate their win rate.
        """
        out: List[ClosedPosition] = []
        offset = 0
        while offset < limit:
            rows = self._get("/closed-positions", {
                "user": wallet, "limit": CLOSED_PAGE, "offset": offset,
                "sortBy": "TIMESTAMP", "sortDirection": "DESC",
            }) or []
            out.extend(c for c in (parse_closed(r) for r in rows) if c)
            if len(rows) < CLOSED_PAGE:
                break
            offset += CLOSED_PAGE
        return out[:limit]
