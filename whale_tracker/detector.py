"""Turns a stream of trades into whale alerts.

Four rules, checked per new trade in time order:

* **watched_wallet** – a followed Polymarket wallet traded at least
  ``watch_min_usd``.
* **large_trade** – one fill at or above the platform's whale threshold.
* **accumulation** – (Polymarket) one wallet's smaller fills on the same market
  and side add up to the threshold within ``accumulation_window_seconds``.
  This catches whales who split orders to stay under the radar.
* **sweep** – (Kalshi) anonymous fills on the same market and side within a
  few seconds add up to the threshold: most likely one big order walking the
  book. That's an inference, and the alert says so.

A fill only counts toward one alert; once a window fires it is emptied, so the
next accumulation alert needs another full threshold's worth.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

from whale_tracker.config import TrackerConfig
from whale_tracker.models import (
    ACCUMULATION, KALSHI, LARGE_TRADE, POLYMARKET, SWEEP, WATCHED_WALLET, Alert, Trade,
)

GroupKey = Tuple[str, str, str, str, str]


def _usd(amount: float) -> str:
    return f"${amount:,.0f}"


class WhaleDetector:
    def __init__(
        self,
        config: TrackerConfig,
        watchlist: Optional[Dict[str, Dict[str, str]]] = None,
        wallet_trade_count: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.config = config
        self.watchlist = watchlist or {}
        self._wallet_trade_count = wallet_trade_count
        self._windows: Dict[GroupKey, Deque[Trade]] = defaultdict(deque)

    def process(self, trades: Iterable[Trade]) -> List[Alert]:
        """Alerts for ``trades``, which must already be de-duplicated."""
        alerts: List[Alert] = []
        for trade in sorted(trades, key=lambda t: t.timestamp):
            alert = self._single(trade)
            if alert is None:
                alert = self._grouped(trade)
            if alert is not None:
                self._annotate(alert)
                alerts.append(alert)
        return alerts

    def _single(self, trade: Trade) -> Optional[Alert]:
        watched = self.watchlist.get(trade.trader or "")
        if watched and trade.notional >= self.config.watch_min_usd:
            alert = self._alert(WATCHED_WALLET, [trade])
            label = watched.get("label") or "on your watchlist"
            alert.reasons.append(f"followed wallet ({label})")
            return alert
        threshold = self.config.min_usd_for(trade.platform)
        if trade.notional >= threshold:
            alert = self._alert(LARGE_TRADE, [trade])
            alert.reasons.append(f"single fill ≥ {_usd(threshold)}")
            return alert
        return None

    def _grouped(self, trade: Trade) -> Optional[Alert]:
        if trade.platform == POLYMARKET and trade.trader:
            kind, window_s = ACCUMULATION, self.config.accumulation_window_seconds
            who = trade.trader
        elif trade.platform == KALSHI:
            kind, window_s = SWEEP, self.config.kalshi_sweep_window_seconds
            who = ""
        else:
            return None

        key: GroupKey = (trade.platform, who, trade.market_id, trade.outcome, trade.side)
        window = self._windows[key]
        window.append(trade)
        horizon = trade.timestamp - timedelta(seconds=window_s)
        while window and window[0].timestamp < horizon:
            window.popleft()

        threshold = self.config.min_usd_for(trade.platform)
        total = sum(t.notional for t in window)
        if total < threshold:
            return None

        fills = list(window)
        window.clear()
        alert = self._alert(kind, fills)
        span = (fills[-1].timestamp - fills[0].timestamp).total_seconds()
        if kind == ACCUMULATION:
            alert.reasons.append(f"{len(fills)} smaller fills from one wallet totalling {_usd(total)} in {_span(span)}")
        else:
            alert.reasons.append(
                f"{len(fills)} fills totalling {_usd(total)} within {_span(span)}, "
                "likely one large order (Kalshi doesn't publish trader identities)"
            )
        return alert

    def _alert(self, kind: str, fills: List[Trade]) -> Alert:
        first, last = fills[0], fills[-1]
        notional = sum(t.notional for t in fills)
        shares = sum(t.size for t in fills)
        return Alert(
            kind=kind,
            platform=first.platform,
            market_id=first.market_id,
            market_title=first.market_title,
            outcome=first.outcome,
            side=first.side,
            notional=notional,
            avg_price=notional / shares if shares else first.price,
            trade_count=len(fills),
            first_ts=first.timestamp,
            last_ts=last.timestamp,
            trader=first.trader,
            trader_name=next((t.trader_name for t in fills if t.trader_name), None),
            url=first.url,
            trade_ids=[t.trade_id for t in fills],
        )

    def _annotate(self, alert: Alert) -> None:
        if alert.side == "BUY":
            if alert.avg_price <= self.config.longshot_price and alert.avg_price > 0:
                alert.reasons.append(
                    f"longshot at {alert.avg_price * 100:.0f}¢, pays {1 / alert.avg_price:.1f}x if right"
                )
            elif alert.avg_price >= self.config.favorite_price:
                alert.reasons.append(f"heavy favourite at {alert.avg_price * 100:.0f}¢")
        if alert.trader and alert.kind != WATCHED_WALLET and self._wallet_trade_count:
            count = self._wallet_trade_count(alert.trader)
            if count < self.config.fresh_wallet_trades:
                alert.reasons.append(f"new wallet: only {count} trade{'s' if count != 1 else ''} on record")


def _span(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
