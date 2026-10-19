"""The polling loop: snapshot each followed trader, diff, size, alert."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import requests

from copy_trader.config import CopyConfig
from copy_trader.notify import Notifier, dispatch
from copy_trader.polymarket import PolymarketClient, Position
from copy_trader.scoring import Scorecard, score_trader
from copy_trader.signals import BUYS, SELLS, WAIT, Signal, consensus_notes, diff_positions, size_buy, size_sell
from copy_trader.store import CopyStore

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CopyWatcher:
    def __init__(
        self,
        config: CopyConfig,
        store: CopyStore,
        notifiers: List[Notifier],
        client: Optional[PolymarketClient] = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.config = config
        self.store = store
        self.notifiers = notifiers
        self.client = client or PolymarketClient()
        self.clock = clock
        self._leaderboard: Dict[str, float] = {}
        self._leaderboard_at: Optional[datetime] = None

    def _leaderboard_pnl(self, wallet: str) -> Optional[float]:
        now = self.clock()
        if self._leaderboard_at is None or now - self._leaderboard_at > timedelta(hours=self.config.rescore_hours):
            try:
                self._leaderboard = {r["wallet"]: r["pnl"] for r in self.client.leaderboard("ALL", "PNL", 50)}
                self._leaderboard_at = now
            except requests.RequestException as exc:
                logger.warning("Leaderboard fetch failed: %s", exc)
        return self._leaderboard.get(wallet)

    def rescore(self, wallet: str, name: str, positions: List[Position]) -> Scorecard:
        closed = self.client.closed_positions(wallet, self.config.history_limit)
        card = score_trader(
            wallet, closed, positions, self.config, name=name,
            leaderboard_pnl=self._leaderboard_pnl(wallet), now=self.clock(),
        )
        self.store.save_scorecard(card)
        return card

    def poll_once(self) -> List[Signal]:
        traders = self.store.traders()
        holdings: Dict[str, Dict[str, Position]] = {}
        pending: List[tuple] = []

        for trader in traders:
            wallet, name = trader["wallet"], trader["name"]
            try:
                positions = self.client.positions(wallet)
            except (requests.RequestException, RuntimeError) as exc:
                # Keep the old snapshot: a failed fetch must never look like a sell-off.
                logger.warning("%s: positions fetch failed (%s); skipping this poll", wallet, exc)
                old = self.store.snapshot(wallet)
                if old:
                    holdings[wallet] = old
                continue
            new = {p.asset: p for p in positions}
            holdings[wallet] = new

            card = trader["card"]
            stale = trader["scored_at"] is None or self.clock() - trader["scored_at"] > timedelta(hours=self.config.rescore_hours)
            if stale:
                try:
                    card = self.rescore(wallet, name, positions)
                except requests.RequestException as exc:
                    logger.warning("%s: rescoring failed (%s)", wallet, exc)

            old = self.store.snapshot(wallet)
            self.store.save_snapshot(wallet, new)
            if old is None:
                logger.info("%s: baseline taken (%d positions); signals start from the next change", wallet, len(new))
                continue
            if card and not card.qualifies and trader["source"] != "manual":
                logger.info("%s no longer meets the bar (%s); not copying", wallet, "; ".join(card.failures))
                continue
            for signal in diff_positions(wallet, old, new, self.config, trader_name=name or (card.name if card else "")):
                pending.append((signal, card))

        sent: List[Signal] = []
        for signal, card in pending:
            if card:
                signal.trader_summary = f"{signal.trader_name or signal.trader[:10]}: {card.summary()}"
                if not card.qualifies:
                    signal.notes.append("note: below your copy bar (" + "; ".join(card.failures) + ")")
            consensus_notes(signal, (h for w, h in holdings.items() if w != signal.trader))
            if signal.action in BUYS:
                size_buy(signal, self.config)
            elif signal.action in SELLS:
                size_sell(signal, self.store.copied_shares(signal.trader, signal.asset), self.config)
            if signal.action != WAIT:
                self.store.record_copy(signal)
            self.store.log_signal(signal)
            dispatch(signal, self.notifiers)
            sent.append(signal)
        logger.info("Polled %d traders, %d signals", len(traders), len(sent))
        return sent

    def run(self, iterations: Optional[int] = None, sleep: Callable[[float], None] = time.sleep) -> None:
        count = 0
        while iterations is None or count < iterations:
            started = time.monotonic()
            self.poll_once()
            count += 1
            if iterations is None or count < iterations:
                sleep(max(0.0, self.config.poll_seconds - (time.monotonic() - started)))
