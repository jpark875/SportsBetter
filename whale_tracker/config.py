"""Tracker settings, read from ``WHALE_*`` environment variables (or ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

from whale_tracker.models import KALSHI, PLATFORMS

load_dotenv()


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class TrackerConfig:
    platforms: List[str] = field(default_factory=lambda: list(PLATFORMS))
    poll_seconds: int = 30
    # On a cold start (or after a long gap) only trades this recent can alert,
    # so the first poll doesn't replay hours of history.
    lookback_seconds: int = 900

    # A single fill at or above this notional is a whale trade.
    min_usd: float = 10_000.0
    kalshi_min_usd: float = 10_000.0
    # Smaller Polymarket fills are still pulled so split orders can be summed.
    polymarket_floor_usd: float = 1_000.0
    # Same wallet, same market and side, within this window → one position.
    accumulation_window_seconds: int = 3600
    # Kalshi has no trader ids: fills on one market and side this close together
    # are treated as one order walking the book.
    kalshi_sweep_window_seconds: int = 10

    # Followed wallets alert at a much lower bar.
    watch_min_usd: float = 1_000.0
    # A Polymarket wallet with fewer trades than this on record is "new".
    fresh_wallet_trades: int = 5
    longshot_price: float = 0.15
    favorite_price: float = 0.90

    db_path: str = os.path.join("data", "whales.db")

    discord_webhook_url: str = ""
    slack_webhook_url: str = ""
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls) -> "TrackerConfig":
        d = cls()
        return cls(
            platforms=[p for p in _list("WHALE_PLATFORMS", d.platforms) if p in PLATFORMS],
            poll_seconds=_int("WHALE_POLL_SECONDS", d.poll_seconds),
            lookback_seconds=_int("WHALE_LOOKBACK_SECONDS", d.lookback_seconds),
            min_usd=_float("WHALE_MIN_USD", d.min_usd),
            kalshi_min_usd=_float("WHALE_KALSHI_MIN_USD", _float("WHALE_MIN_USD", d.kalshi_min_usd)),
            polymarket_floor_usd=_float("WHALE_POLYMARKET_FLOOR_USD", d.polymarket_floor_usd),
            accumulation_window_seconds=_int("WHALE_ACCUMULATION_WINDOW_SECONDS", d.accumulation_window_seconds),
            kalshi_sweep_window_seconds=_int("WHALE_KALSHI_SWEEP_WINDOW_SECONDS", d.kalshi_sweep_window_seconds),
            watch_min_usd=_float("WHALE_WATCH_MIN_USD", d.watch_min_usd),
            fresh_wallet_trades=_int("WHALE_FRESH_WALLET_TRADES", d.fresh_wallet_trades),
            longshot_price=_float("WHALE_LONGSHOT_PRICE", d.longshot_price),
            favorite_price=_float("WHALE_FAVORITE_PRICE", d.favorite_price),
            db_path=os.getenv("WHALE_DB_PATH", d.db_path),
            discord_webhook_url=os.getenv("WHALE_DISCORD_WEBHOOK_URL", ""),
            slack_webhook_url=os.getenv("WHALE_SLACK_WEBHOOK_URL", ""),
            ntfy_topic=os.getenv("WHALE_NTFY_TOPIC", ""),
            ntfy_server=os.getenv("WHALE_NTFY_SERVER", d.ntfy_server),
            telegram_bot_token=os.getenv("WHALE_TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("WHALE_TELEGRAM_CHAT_ID", ""),
        )

    def min_usd_for(self, platform: str) -> float:
        return self.kalshi_min_usd if platform == KALSHI else self.min_usd
