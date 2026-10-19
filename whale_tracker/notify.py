"""Alert delivery: terminal, Discord, Slack, ntfy (phone push) and Telegram."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional, TextIO, Tuple

import requests

from whale_tracker.config import TrackerConfig
from whale_tracker.models import ACCUMULATION, LARGE_TRADE, SWEEP, WATCHED_WALLET, Alert
from whale_tracker.sources.polymarket import profile_url

logger = logging.getLogger(__name__)

_HEADLINE = {
    LARGE_TRADE: "Whale trade",
    ACCUMULATION: "Whale accumulating",
    SWEEP: "Large order (sweep)",
    WATCHED_WALLET: "Followed wallet traded",
}


def _short(wallet: str) -> str:
    return f"{wallet[:6]}…{wallet[-4:]}" if len(wallet) > 12 else wallet


def format_alert(alert: Alert) -> Tuple[str, str]:
    """``(title, body)`` as plain text, shared by every channel."""
    platform = alert.platform.capitalize()
    title = (
        f"🐋 {_HEADLINE.get(alert.kind, 'Whale')} on {platform}: ${alert.notional:,.0f} "
        f"{alert.side} {alert.outcome} @ {alert.avg_price * 100:.0f}¢"
    )
    lines = [alert.market_title or alert.market_id]
    if alert.trader:
        who = f"{alert.trader_name} ({_short(alert.trader)})" if alert.trader_name else alert.trader
        lines.append(f"Trader: {who} · {profile_url(alert.trader)}")
    when = alert.last_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    fills = f"{alert.trade_count} fills" if alert.trade_count > 1 else "1 fill"
    lines.append(f"{fills}, last at {when}")
    if alert.reasons:
        lines.append("Why: " + "; ".join(alert.reasons))
    if alert.url:
        lines.append(alert.url)
    return title, "\n".join(lines)


class Notifier:
    name = "notifier"

    def send(self, alert: Alert) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    name = "console"

    def __init__(self, stream: Optional[TextIO] = None, bell: bool = False) -> None:
        self.stream = stream or sys.stdout
        self.bell = bell

    def send(self, alert: Alert) -> None:
        title, body = format_alert(alert)
        bell = "\a" if self.bell else ""
        indented = "\n".join("   " + line for line in body.splitlines())
        print(f"{bell}{title}\n{indented}\n", file=self.stream, flush=True)


class _HttpNotifier(Notifier):
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _post(self, url: str, **kwargs) -> None:
        response = self.session.post(url, timeout=10, **kwargs)
        response.raise_for_status()


class DiscordNotifier(_HttpNotifier):
    name = "discord"

    def __init__(self, webhook_url: str, session: Optional[requests.Session] = None) -> None:
        super().__init__(session)
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> None:
        title, body = format_alert(alert)
        self._post(self.webhook_url, json={"content": f"**{title}**\n{body}"[:2000]})


class SlackNotifier(_HttpNotifier):
    name = "slack"

    def __init__(self, webhook_url: str, session: Optional[requests.Session] = None) -> None:
        super().__init__(session)
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> None:
        title, body = format_alert(alert)
        self._post(self.webhook_url, json={"text": f"*{title}*\n{body}"})


class NtfyNotifier(_HttpNotifier):
    """Push to a phone via https://ntfy.sh: install the app, subscribe to the topic."""

    name = "ntfy"

    def __init__(self, topic: str, server: str = "https://ntfy.sh", session: Optional[requests.Session] = None) -> None:
        super().__init__(session)
        self.url = f"{server.rstrip('/')}/{topic}"

    def send(self, alert: Alert) -> None:
        title, body = format_alert(alert)
        headers = {
            # HTTP headers must be latin-1; the emoji goes in as a tag instead.
            "Title": title.replace("🐋 ", "").encode("latin-1", "replace").decode("latin-1"),
            "Tags": "whale",
            "Priority": "high" if alert.notional >= 50_000 else "default",
        }
        if alert.url:
            headers["Click"] = alert.url
        self._post(self.url, data=body.encode("utf-8"), headers=headers)


class TelegramNotifier(_HttpNotifier):
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, session: Optional[requests.Session] = None) -> None:
        super().__init__(session)
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id

    def send(self, alert: Alert) -> None:
        title, body = format_alert(alert)
        self._post(self.url, json={
            "chat_id": self.chat_id,
            "text": f"{title}\n{body}",
            "disable_web_page_preview": True,
        })


def build_notifiers(config: TrackerConfig, console: bool = True, bell: bool = False) -> List[Notifier]:
    notifiers: List[Notifier] = [ConsoleNotifier(bell=bell)] if console else []
    if config.discord_webhook_url:
        notifiers.append(DiscordNotifier(config.discord_webhook_url))
    if config.slack_webhook_url:
        notifiers.append(SlackNotifier(config.slack_webhook_url))
    if config.ntfy_topic:
        notifiers.append(NtfyNotifier(config.ntfy_topic, config.ntfy_server))
    if config.telegram_bot_token and config.telegram_chat_id:
        notifiers.append(TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id))
    return notifiers


def dispatch(alert: Alert, notifiers: List[Notifier]) -> None:
    """Send to every channel; one failing channel never blocks the others."""
    for notifier in notifiers:
        try:
            notifier.send(alert)
        except Exception as exc:  # noqa: BLE001 — a dead webhook must not stop the tracker
            logger.warning("%s notification failed: %s", notifier.name, exc)
