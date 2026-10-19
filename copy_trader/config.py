"""Settings, read from ``COPY_*`` environment variables (or ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass
class CopyConfig:
    # --- Who is worth copying ---
    min_resolved: int = 30           # resolved positions needed before a win rate means anything
    min_win_rate: float = 0.60
    min_profit_usd: float = 50_000   # all-time profit
    # Win rate minus the average price paid. Buying 95¢ favourites wins 95% of
    # the time and earns nothing; this demands they beat the odds they bought at.
    min_edge: float = 0.05
    max_idle_days: int = 30
    history_limit: int = 500         # most recent resolved positions scored per trader

    # --- What to copy ---
    bankroll_usd: float = 1_000.0
    copy_scale: float = 1.0          # 1.0 = same share of your bankroll as their trade is of their portfolio
    max_position_pct: float = 0.05   # never put more than this share of bankroll into one copy
    min_order_usd: float = 1.0
    min_signal_usd: float = 500.0    # ignore the trader's moves smaller than this
    min_trim_fraction: float = 0.20  # ignore trims smaller than this share of their position
    max_chase: float = 0.05          # skip if the price has run more than 5¢ past their entry
    slippage: float = 0.01           # limit = current price + 1¢ (buys) / − 1¢ (sells)
    max_entry_price: float = 0.97    # little upside left above this

    poll_seconds: int = 60
    rescore_hours: int = 24
    db_path: str = os.path.join("data", "copytrader.db")

    discord_webhook_url: str = ""
    slack_webhook_url: str = ""
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls) -> "CopyConfig":
        d = cls()
        return cls(
            min_resolved=_int("COPY_MIN_RESOLVED", d.min_resolved),
            min_win_rate=_float("COPY_MIN_WIN_RATE", d.min_win_rate),
            min_profit_usd=_float("COPY_MIN_PROFIT_USD", d.min_profit_usd),
            min_edge=_float("COPY_MIN_EDGE", d.min_edge),
            max_idle_days=_int("COPY_MAX_IDLE_DAYS", d.max_idle_days),
            history_limit=_int("COPY_HISTORY_LIMIT", d.history_limit),
            bankroll_usd=_float("COPY_BANKROLL_USD", d.bankroll_usd),
            copy_scale=_float("COPY_SCALE", d.copy_scale),
            max_position_pct=_float("COPY_MAX_POSITION_PCT", d.max_position_pct),
            min_order_usd=_float("COPY_MIN_ORDER_USD", d.min_order_usd),
            min_signal_usd=_float("COPY_MIN_SIGNAL_USD", d.min_signal_usd),
            min_trim_fraction=_float("COPY_MIN_TRIM_FRACTION", d.min_trim_fraction),
            max_chase=_float("COPY_MAX_CHASE", d.max_chase),
            slippage=_float("COPY_SLIPPAGE", d.slippage),
            max_entry_price=_float("COPY_MAX_ENTRY_PRICE", d.max_entry_price),
            poll_seconds=_int("COPY_POLL_SECONDS", d.poll_seconds),
            rescore_hours=_int("COPY_RESCORE_HOURS", d.rescore_hours),
            db_path=os.getenv("COPY_DB_PATH", d.db_path),
            discord_webhook_url=os.getenv("COPY_DISCORD_WEBHOOK_URL", ""),
            slack_webhook_url=os.getenv("COPY_SLACK_WEBHOOK_URL", ""),
            ntfy_topic=os.getenv("COPY_NTFY_TOPIC", ""),
            ntfy_server=os.getenv("COPY_NTFY_SERVER", d.ntfy_server),
            telegram_bot_token=os.getenv("COPY_TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("COPY_TELEGRAM_CHAT_ID", ""),
        )
