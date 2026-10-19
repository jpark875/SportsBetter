"""Kalshi's public market-data API (no key needed for reads).

``GET /markets/trades`` is a global, cursor-paged feed of every fill. It
carries no trader identity, so Kalshi whales can only be spotted by size,
including bursts of fills that one large order leaves when it walks the book.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from whale_tracker.models import KALSHI, Trade
from whale_tracker.sources.http import get_json, make_session

logger = logging.getLogger(__name__)

API = "https://api.elections.kalshi.com/trade-api/v2"
SITE = "https://kalshi.com"
PAGE_SIZE = 1000


def _price(raw: Dict[str, Any], side: str) -> float:
    dollars = raw.get(f"{side}_price_dollars")
    if dollars not in (None, ""):
        return float(dollars)
    return float(raw[f"{side}_price"]) / 100


def _count(raw: Dict[str, Any]) -> float:
    fp = raw.get("count_fp")
    return float(fp) if fp not in (None, "") else float(raw["count"])


def parse_trade(raw: Dict[str, Any]) -> Optional[Trade]:
    try:
        side = str(raw["taker_side"]).lower()
        ticker = raw["ticker"]
        timestamp = datetime.fromisoformat(str(raw["created_time"]).replace("Z", "+00:00"))
        trade = Trade(
            platform=KALSHI,
            trade_id=str(raw["trade_id"]),
            market_id=ticker,
            market_title="",
            outcome=side.capitalize(),
            side="BUY",
            price=_price(raw, side),
            size=_count(raw),
            timestamp=timestamp,
            url=market_url(ticker),
        )
    except (KeyError, TypeError, ValueError):
        logger.debug("Skipping malformed Kalshi trade: %r", raw)
        return None
    return trade


def market_url(ticker: str) -> str:
    # The site routes by series ticker (the part before the first dash).
    return f"{SITE}/markets/{ticker.split('-')[0].lower()}"


class KalshiClient:
    platform = KALSHI

    def __init__(self, session: Optional[requests.Session] = None, base_url: str = API) -> None:
        self._session = session or make_session()
        self._base = base_url.rstrip("/")
        self._titles: Dict[str, str] = {}

    def recent_trades(self, since: datetime, max_pages: int = 20) -> List[Trade]:
        params: Dict[str, Any] = {"limit": PAGE_SIZE, "min_ts": int(since.timestamp())}
        trades: List[Trade] = []
        for _ in range(max_pages):
            raw = get_json(self._session, f"{self._base}/markets/trades", params)
            trades.extend(t for t in (parse_trade(r) for r in raw.get("trades", [])) if t)
            cursor = raw.get("cursor")
            if not cursor:
                break
            params["cursor"] = cursor
        else:
            logger.warning("Kalshi: stopped after %d pages; some trades were skipped. Poll more often.", max_pages)
        return [t for t in trades if t.timestamp >= since]

    def market_title(self, ticker: str) -> str:
        if ticker not in self._titles:
            try:
                market = get_json(self._session, f"{self._base}/markets/{ticker}").get("market", {})
            except requests.RequestException:
                logger.warning("Kalshi: could not look up market %s", ticker)
                return ticker
            title = market.get("title") or ticker
            subtitle = market.get("yes_sub_title") or market.get("subtitle")
            if subtitle and subtitle not in title:
                title = f"{title} — {subtitle}"
            self._titles[ticker] = title
        return self._titles[ticker]
