"""Signal delivery: terminal, Discord, Slack, ntfy (phone push) and Telegram."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional, TextIO

import requests

from copy_trader.config import CopyConfig
from copy_trader.signals import Signal, format_signal

logger = logging.getLogger(__name__)


class Notifier:
    name = "notifier"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def send(self, signal: Signal) -> None:
        raise NotImplementedError

    def _post(self, url: str, **kwargs) -> None:
        self.session.post(url, timeout=10, **kwargs).raise_for_status()


class ConsoleNotifier(Notifier):
    name = "console"

    def __init__(self, stream: Optional[TextIO] = None, bell: bool = False) -> None:
        self.stream = stream or sys.stdout
        self.bell = bell

    def send(self, signal: Signal) -> None:
        title, body = format_signal(signal)
        indented = "\n".join("   " + line for line in body.splitlines())
        print(f"{chr(7) if self.bell else ''}{title}\n{indented}\n", file=self.stream, flush=True)


class DiscordNotifier(Notifier):
    name = "discord"

    def __init__(self, url: str, session=None) -> None:
        super().__init__(session)
        self.url = url

    def send(self, signal: Signal) -> None:
        title, body = format_signal(signal)
        self._post(self.url, json={"content": f"**{title}**\n{body}"[:2000]})


class SlackNotifier(Notifier):
    name = "slack"

    def __init__(self, url: str, session=None) -> None:
        super().__init__(session)
        self.url = url

    def send(self, signal: Signal) -> None:
        title, body = format_signal(signal)
        self._post(self.url, json={"text": f"*{title}*\n{body}"})


class NtfyNotifier(Notifier):
    name = "ntfy"

    def __init__(self, topic: str, server: str = "https://ntfy.sh", session=None) -> None:
        super().__init__(session)
        self.url = f"{server.rstrip('/')}/{topic}"

    def send(self, signal: Signal) -> None:
        title, body = format_signal(signal)
        headers = {
            # Header values must be latin-1.
            "Title": title.replace("▸", "-").replace("≈", "~").encode("latin-1", "replace").decode("latin-1"),
            "Tags": "chart_with_upwards_trend" if signal.side == "BUY" else "chart_with_downwards_trend",
            "Priority": "high" if signal.action != "WAIT" else "low",
            "Click": signal.url,
        }
        self._post(self.url, data=body.encode("utf-8"), headers=headers)


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, token: str, chat_id: str, session=None) -> None:
        super().__init__(session)
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, signal: Signal) -> None:
        title, body = format_signal(signal)
        self._post(self.url, json={"chat_id": self.chat_id, "text": f"{title}\n{body}", "disable_web_page_preview": True})


def build_notifiers(config: CopyConfig, console: bool = True, bell: bool = False) -> List[Notifier]:
    out: List[Notifier] = [ConsoleNotifier(bell=bell)] if console else []
    if config.discord_webhook_url:
        out.append(DiscordNotifier(config.discord_webhook_url))
    if config.slack_webhook_url:
        out.append(SlackNotifier(config.slack_webhook_url))
    if config.ntfy_topic:
        out.append(NtfyNotifier(config.ntfy_topic, config.ntfy_server))
    if config.telegram_bot_token and config.telegram_chat_id:
        out.append(TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id))
    return out


def dispatch(signal: Signal, notifiers: List[Notifier]) -> None:
    for notifier in notifiers:
        try:
            notifier.send(signal)
        except Exception as exc:  # noqa: BLE001 — one dead channel must not stop the others
            logger.warning("%s notification failed: %s", notifier.name, exc)
