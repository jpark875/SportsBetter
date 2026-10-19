"""Command line: ``python -m whale_tracker <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import List, Optional

import requests

from whale_tracker.config import TrackerConfig
from whale_tracker.models import LARGE_TRADE, PLATFORMS, POLYMARKET, Alert
from whale_tracker.notify import ConsoleNotifier, build_notifiers, dispatch
from whale_tracker.sources import PolymarketClient
from whale_tracker.store import WhaleStore
from whale_tracker.watcher import WhaleWatcher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whale-tracker",
        description="Watch Polymarket and Kalshi for whale trades and alert on them.",
    )
    parser.add_argument("--db", help="SQLite file (default: WHALE_DB_PATH or data/whales.db)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="poll continuously and alert on whale trades")
    watch.add_argument("--platform", choices=PLATFORMS, action="append", help="limit to one platform (repeatable)")
    watch.add_argument("--min-usd", type=float, help="whale threshold for a single trade")
    watch.add_argument("--interval", type=int, help="seconds between polls")
    watch.add_argument("--once", action="store_true", help="poll once and exit")
    watch.add_argument("--bell", action="store_true", help="ring the terminal bell on each alert")

    follow = sub.add_parser("follow", help="follow a Polymarket wallet")
    follow.add_argument("wallet")
    follow.add_argument("--label", default="")

    unfollow = sub.add_parser("unfollow", help="stop following a Polymarket wallet")
    unfollow.add_argument("wallet")

    sub.add_parser("following", help="list followed wallets")

    board = sub.add_parser("import-leaderboard", help="follow Polymarket's top traders")
    board.add_argument("--top", type=int, default=20)
    board.add_argument("--period", default="MONTH", choices=["DAY", "WEEK", "MONTH", "ALL"])
    board.add_argument("--by", default="PNL", choices=["PNL", "VOL"])

    recent = sub.add_parser("recent", help="show recent alerts")
    recent.add_argument("--limit", type=int, default=20)
    recent.add_argument("--platform", choices=PLATFORMS)

    sub.add_parser("test-alert", help="send a sample alert to every configured channel")
    return parser


def _sample_alert() -> Alert:
    now = datetime.now(timezone.utc)
    return Alert(
        kind=LARGE_TRADE, platform=POLYMARKET, market_id="test", market_title="Test alert: is the tracker wired up?",
        outcome="Yes", side="BUY", notional=25_000, avg_price=0.12, trade_count=1, first_ts=now, last_ts=now,
        trader="0x0000000000000000000000000000000000000000", trader_name="test-whale",
        reasons=["this is a test"], trade_ids=[f"test-{now.timestamp()}"],
    )


def _is_wallet(value: str) -> bool:
    return value.startswith("0x") and len(value) == 42


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    config = TrackerConfig.from_env()
    if args.db:
        config.db_path = args.db
    store = WhaleStore(config.db_path)

    if args.command == "watch":
        if args.platform:
            config.platforms = list(dict.fromkeys(args.platform))
        if args.min_usd is not None:
            config.min_usd = config.kalshi_min_usd = args.min_usd
        if args.interval:
            config.poll_seconds = args.interval
        notifiers = build_notifiers(config, bell=args.bell)
        channels = ", ".join(n.name for n in notifiers)
        print(f"Watching {', '.join(config.platforms)} for trades ≥ ${config.min_usd:,.0f} → {channels}. Ctrl+C to stop.")
        watcher = WhaleWatcher(config, store, notifiers)
        try:
            watcher.run(iterations=1 if args.once else None)
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0

    if args.command == "follow":
        if not _is_wallet(args.wallet):
            print("Expected a 0x… Polymarket wallet address (42 characters).", file=sys.stderr)
            return 2
        store.follow(POLYMARKET, args.wallet, args.label)
        print(f"Following {args.wallet}" + (f" ({args.label})" if args.label else ""))
        return 0

    if args.command == "unfollow":
        if store.unfollow(POLYMARKET, args.wallet):
            print(f"Unfollowed {args.wallet}")
            return 0
        print(f"{args.wallet} wasn't on the watchlist.", file=sys.stderr)
        return 1

    if args.command == "following":
        watchlist = store.watchlist()
        if not watchlist:
            print("No wallets followed yet. Try `follow 0x…` or `import-leaderboard`.")
        for wallet, info in watchlist.items():
            print(f"{wallet}  {info['label'] or '-'}  [{info['source']}]")
        return 0

    if args.command == "import-leaderboard":
        try:
            leaders = PolymarketClient().leaderboard(args.period, args.by, args.top)
        except requests.RequestException as exc:
            print(f"Could not fetch the Polymarket leaderboard: {exc}", file=sys.stderr)
            return 1
        for leader in leaders:
            label = f"#{leader['rank']} {args.period.lower()} {args.by.lower()}"
            if leader["name"]:
                label = f"{leader['name']}, {label}"
            store.follow(POLYMARKET, leader["wallet"], label, source="leaderboard")
            print(f"{leader['wallet']}  {label}  pnl ${leader['pnl']:,.0f}  vol ${leader['volume']:,.0f}")
        print(f"Following {len(leaders)} leaderboard wallets.")
        return 0

    if args.command == "recent":
        alerts = store.recent_alerts(args.limit, args.platform)
        if not alerts:
            print("No alerts yet.")
        console = ConsoleNotifier()
        for alert in alerts:
            console.send(alert)
        return 0

    if args.command == "test-alert":
        notifiers = build_notifiers(config)
        dispatch(_sample_alert(), notifiers)
        print(f"Sent a test alert to: {', '.join(n.name for n in notifiers)}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
