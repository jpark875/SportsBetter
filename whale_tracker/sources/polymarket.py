"""Polymarket's public Data API (no key needed).

Trades come from ``GET /trades``, which can filter server-side by notional
(``filterType=CASH&filterAmount=…``) or by wallet (``user=…``). Wallets on
Polymarket are public, which is what makes per-trader tracking possible.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from whale_tracker.models import POLYMARKET, Trade
from whale_tracker.sources.http import get_json, make_session

logger = logging.getLogger(__name__)

DATA_API = "https://data-api.polymarket.com"
SITE = "https://polymarket.com"
PAGE_SIZE = 500


def _ts(raw: Any) -> datetime:
    value = float(raw)
    if value > 1e12:
        value /= 1000
    return datetime.fromtimestamp(value, tz=timezone.utc)


def parse_trade(raw: Dict[str, Any]) -> Optional[Trade]:
    try:
        wallet = (raw.get("proxyWallet") or "").lower()
        size = float(raw["size"])
        price = float(raw["price"])
        side = str(raw.get("side", "")).upper()
        tx = raw.get("transactionHash") or ""
        asset = raw.get("asset") or raw.get("conditionId") or ""
        timestamp = _ts(raw["timestamp"])
    except (KeyError, TypeError, ValueError):
        logger.debug("Skipping malformed Polymarket trade: %r", raw)
        return None

    slug = raw.get("eventSlug") or raw.get("slug")
    url = f"{SITE}/event/{slug}" if slug else None
    return Trade(
        platform=POLYMARKET,
        trade_id=f"{tx}:{asset}:{wallet}:{side}:{size}:{price}",
        market_id=raw.get("conditionId") or asset,
        market_title=raw.get("title") or "",
        outcome=str(raw.get("outcome") or ""),
        side=side,
        price=price,
        size=size,
        timestamp=timestamp,
        trader=wallet or None,
        trader_name=raw.get("name") or raw.get("pseudonym") or None,
        url=url,
    )


def profile_url(wallet: str) -> str:
    return f"{SITE}/profile/{wallet}"


class PolymarketClient:
    platform = POLYMARKET

    def __init__(self, session: Optional[requests.Session] = None, base_url: str = DATA_API) -> None:
        self._session = session or make_session()
        self._base = base_url.rstrip("/")
        self._history_cache: Dict[str, int] = {}

    def _trades(self, params: Dict[str, Any]) -> List[Trade]:
        raw = get_json(self._session, f"{self._base}/trades", params)
        rows = raw if isinstance(raw, list) else raw.get("data", [])
        return [t for t in (parse_trade(r) for r in rows) if t is not None]

    def recent_trades(self, min_usd: float, since: datetime, max_pages: int = 5) -> List[Trade]:
        """Every taker fill ≥ ``min_usd`` since ``since``, newest first.

        The feed is newest-first, so paging stops at the first page that reaches
        back past ``since``. Hitting ``max_pages`` first means trades were
        missed; that's logged so the floor or poll interval can be raised.
        """
        trades: List[Trade] = []
        for page in range(max_pages):
            batch = self._trades({
                "limit": PAGE_SIZE,
                "offset": page * PAGE_SIZE,
                "takerOnly": "true",
                "filterType": "CASH",
                "filterAmount": min_usd,
            })
            trades.extend(batch)
            if len(batch) < PAGE_SIZE or batch[-1].timestamp < since:
                break
        else:
            logger.warning(
                "Polymarket: %d pages of trades ≥ $%.0f since the last poll; some were "
                "skipped. Raise WHALE_POLYMARKET_FLOOR_USD or poll more often.",
                max_pages, min_usd,
            )
        return [t for t in trades if t.timestamp >= since]

    def wallet_trades(self, wallet: str, limit: int = 100) -> List[Trade]:
        # takerOnly=false so a followed wallet's resting (maker) fills show too.
        return self._trades({"user": wallet, "limit": limit, "takerOnly": "false"})

    def wallet_trade_count(self, wallet: str, cap: int = 100) -> int:
        """Trades on record for ``wallet``, capped at ``cap`` (cached per run)."""
        if wallet not in self._history_cache:
            self._history_cache[wallet] = len(self.wallet_trades(wallet, limit=cap))
        return self._history_cache[wallet]

    def leaderboard(self, period: str = "MONTH", order_by: str = "PNL", limit: int = 25) -> List[Dict[str, Any]]:
        """Top traders as ``{"wallet", "name", "pnl", "volume", "rank"}`` dicts."""
        raw = get_json(self._session, f"{self._base}/v1/leaderboard", {
            "timePeriod": period.upper(),
            "orderBy": order_by.upper(),
            "category": "OVERALL",
            "limit": limit,
        })
        rows = raw if isinstance(raw, list) else raw.get("data", [])
        leaders = []
        for i, row in enumerate(rows, start=1):
            wallet = (row.get("proxyWallet") or row.get("user") or "").lower()
            if not wallet:
                continue
            leaders.append({
                "wallet": wallet,
                "name": row.get("userName") or row.get("name") or row.get("pseudonym") or "",
                "pnl": float(row.get("pnl") or 0),
                "volume": float(row.get("vol") or row.get("volume") or 0),
                "rank": int(row.get("rank") or i),
            })
        return leaders
