"""The polling loop: fetch → de-duplicate → detect → store → notify."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

import requests

from whale_tracker.config import TrackerConfig
from whale_tracker.detector import WhaleDetector
from whale_tracker.models import KALSHI, POLYMARKET, Alert, Trade
from whale_tracker.notify import Notifier, dispatch
from whale_tracker.sources import KalshiClient, PolymarketClient
from whale_tracker.store import WhaleStore

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WhaleWatcher:
    def __init__(
        self,
        config: TrackerConfig,
        store: WhaleStore,
        notifiers: List[Notifier],
        polymarket: Optional[PolymarketClient] = None,
        kalshi: Optional[KalshiClient] = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.config = config
        self.store = store
        self.notifiers = notifiers
        self.polymarket = polymarket or (PolymarketClient() if POLYMARKET in config.platforms else None)
        self.kalshi = kalshi or (KalshiClient() if KALSHI in config.platforms else None)
        self.clock = clock
        self.detector = WhaleDetector(config, wallet_trade_count=self._wallet_trade_count)

    def _wallet_trade_count(self, wallet: str) -> int:
        try:
            return self.polymarket.wallet_trade_count(wallet) if self.polymarket else 10 ** 6
        except requests.RequestException as exc:
            logger.debug("Wallet history lookup failed for %s: %s", wallet, exc)
            return 10 ** 6

    def _since(self, platform: str, now: datetime) -> datetime:
        floor = now - timedelta(seconds=self.config.lookback_seconds)
        cursor = self.store.get_cursor(platform)
        return max(cursor, floor) if cursor else floor

    def _fetch(self, platform: str, since: datetime) -> List[Trade]:
        if platform == POLYMARKET and self.polymarket:
            trades = self.polymarket.recent_trades(self.config.polymarket_floor_usd, since)
            for wallet, info in self.store.watchlist(POLYMARKET).items():
                trades.extend(t for t in self.polymarket.wallet_trades(wallet, limit=50) if t.timestamp >= since)
            return trades
        if platform == KALSHI and self.kalshi:
            return self.kalshi.recent_trades(since)
        return []

    def poll_once(self) -> List[Alert]:
        """One pass over every enabled platform; returns the alerts sent."""
        now = self.clock()
        self.detector.watchlist = self.store.watchlist()
        sent: List[Alert] = []
        for platform in self.config.platforms:
            since = self._since(platform, now)
            try:
                trades = self._fetch(platform, since)
            except requests.RequestException as exc:
                logger.warning("%s: fetch failed, will retry next poll (%s)", platform, exc)
                continue

            fresh_ids = self.store.filter_unseen(platform, (t.trade_id for t in trades))
            unique = {t.trade_id: t for t in trades if t.trade_id in fresh_ids}
            alerts = self.detector.process(unique.values())
            for alert in alerts:
                if platform == KALSHI and self.kalshi and not alert.market_title:
                    alert.market_title = self.kalshi.market_title(alert.market_id)
                if self.store.save_alert(alert):
                    dispatch(alert, self.notifiers)
                    sent.append(alert)

            if trades:
                # Back off a little so fills that land late at the API are not skipped.
                newest = max(t.timestamp for t in trades)
                self.store.set_cursor(platform, newest - timedelta(seconds=5))
            logger.info("%s: %d trades fetched, %d new, %d alerts", platform, len(trades), len(unique), len(alerts))
        return sent

    def run(self, iterations: Optional[int] = None, sleep: Callable[[float], None] = time.sleep) -> None:
        count = 0
        while iterations is None or count < iterations:
            started = time.monotonic()
            self.poll_once()
            count += 1
            if count % 100 == 0:
                self.store.prune()
            if iterations is None or count < iterations:
                sleep(max(0.0, self.config.poll_seconds - (time.monotonic() - started)))
