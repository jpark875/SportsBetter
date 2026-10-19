"""Command line: ``python -m copy_trader <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import requests

from copy_trader.config import CopyConfig
from copy_trader.notify import ConsoleNotifier, build_notifiers, dispatch
from copy_trader.polymarket import PolymarketClient, profile_url
from copy_trader.scoring import Scorecard, rank, score_trader
from copy_trader.signals import OPEN, Signal, size_buy
from copy_trader.store import CopyStore
from copy_trader.watcher import CopyWatcher


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="copy-trader",
        description="Follow Polymarket traders with high win rates and big profits, and get the orders to match their trades.",
    )
    p.add_argument("--db", help="SQLite file (default: COPY_DB_PATH or data/copytrader.db)")
    p.add_argument("--bankroll", type=float, help="your bankroll in USD (default: COPY_BANKROLL_USD or 1000)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="score the leaderboard and find traders worth copying")
    d.add_argument("--candidates", type=int, default=50, help="leaderboard traders to score")
    d.add_argument("--period", default="ALL", choices=["DAY", "WEEK", "MONTH", "ALL"])
    d.add_argument("--follow", type=int, default=0, metavar="N", help="follow the top N that qualify")

    s = sub.add_parser("score", help="show one wallet's scorecard")
    s.add_argument("wallet")

    f = sub.add_parser("follow", help="copy a wallet even if it doesn't meet the bar")
    f.add_argument("wallet")
    f.add_argument("--name", default="")

    u = sub.add_parser("unfollow")
    u.add_argument("wallet")

    sub.add_parser("traders", help="list followed traders and their scorecards")

    w = sub.add_parser("watch", help="poll followed traders and send copy orders")
    w.add_argument("--interval", type=int, help="seconds between polls")
    w.add_argument("--once", action="store_true")
    w.add_argument("--bell", action="store_true")

    sub.add_parser("open", help="followed traders' open positions you can still enter near their price")
    r = sub.add_parser("signals", help="recent copy signals")
    r.add_argument("--limit", type=int, default=20)
    sub.add_parser("test-alert", help="send a sample signal to every configured channel")
    return p


def _is_wallet(value: str) -> bool:
    return value.startswith("0x") and len(value) == 42


def _print_card(card: Scorecard, out=None) -> None:
    verdict = "✔ copyable" if card.qualifies else "✘ " + "; ".join(card.failures)
    print(
        f"{card.name or card.wallet}\n"
        f"   {card.wins}/{card.resolved} won ({card.win_rate:.0%}, at least {card.win_rate_low:.0%} at 95% confidence)\n"
        f"   paid {card.implied_win_rate * 100:.0f}¢ on average → edge {card.edge * 100:+.1f} pts\n"
        f"   profit ${card.profit:,.0f} · ROI on resolved {card.roi:+.0%} · {card.open_positions} open (${card.open_value:,.0f})\n"
        f"   {verdict}\n"
        f"   {profile_url(card.wallet)}",
        file=out or sys.stdout,
    )


def _score(client: PolymarketClient, config: CopyConfig, wallet: str, name: str = "", pnl=None) -> Scorecard:
    return score_trader(
        wallet, client.closed_positions(wallet, config.history_limit), client.positions(wallet),
        config, name=name, leaderboard_pnl=pnl,
    )


def main(argv: Optional[List[str]] = None, client: Optional[PolymarketClient] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    config = CopyConfig.from_env()
    if args.db:
        config.db_path = args.db
    if args.bankroll:
        config.bankroll_usd = args.bankroll
    store = CopyStore(config.db_path)
    client = client or PolymarketClient()

    try:
        return _run(args, config, store, client)
    except requests.RequestException as exc:
        print(f"Polymarket API error: {exc}", file=sys.stderr)
        return 1


def _run(args, config: CopyConfig, store: CopyStore, client: PolymarketClient) -> int:
    if args.command == "discover":
        leaders = client.leaderboard(args.period, "PNL", args.candidates)
        print(f"Scoring {len(leaders)} leaderboard traders (most recent {config.history_limit} resolved positions each)…\n")
        cards = []
        for leader in leaders:
            try:
                cards.append(_score(client, config, leader["wallet"], leader["name"], leader["pnl"]))
            except (requests.RequestException, RuntimeError) as exc:
                logging.getLogger(__name__).warning("%s: %s", leader["wallet"], exc)
        ranked = rank(cards)
        for card in ranked:
            _print_card(card)
        good = [c for c in ranked if c.qualifies]
        print(f"\n{len(good)} of {len(cards)} meet the bar (≥{config.min_resolved} resolved, ≥{config.min_win_rate:.0%} wins, "
              f"≥{config.min_edge * 100:.0f} pts edge, ≥${config.min_profit_usd:,.0f} profit).")
        for card in good[: args.follow]:
            store.follow(card.wallet, card.name, source="discover", card=card)
            print(f"Following {card.name or card.wallet}")
        if good and not args.follow:
            print("Run again with --follow N to copy the top N.")
        return 0

    if args.command == "score":
        _print_card(_score(client, config, args.wallet.lower()))
        return 0

    if args.command == "follow":
        if not _is_wallet(args.wallet):
            print("Expected a 0x… Polymarket wallet address (42 characters).", file=sys.stderr)
            return 2
        card = _score(client, config, args.wallet.lower(), args.name)
        store.follow(args.wallet, args.name, card=card)
        _print_card(card)
        if not card.qualifies:
            print("Following anyway, as you asked. Signals will carry a warning.")
        return 0

    if args.command == "unfollow":
        if store.unfollow(args.wallet):
            print(f"Unfollowed {args.wallet}")
            return 0
        print(f"{args.wallet} isn't followed.", file=sys.stderr)
        return 1

    if args.command == "traders":
        traders = store.traders()
        if not traders:
            print("Not following anyone. Start with `discover`.")
        for t in traders:
            if t["card"]:
                _print_card(t["card"])
            else:
                print(f"{t['name'] or t['wallet']} (not scored yet)")
        return 0

    if args.command == "watch":
        if args.interval:
            config.poll_seconds = args.interval
        if not store.traders():
            print("Not following anyone yet. Run `discover --follow 10` first.", file=sys.stderr)
            return 1
        notifiers = build_notifiers(config, bell=args.bell)
        print(f"Copying {len(store.traders())} traders with a ${config.bankroll_usd:,.0f} bankroll → "
              f"{', '.join(n.name for n in notifiers)}. Ctrl+C to stop.")
        try:
            CopyWatcher(config, store, notifiers, client=client).run(iterations=1 if args.once else None)
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0

    if args.command == "open":
        console = ConsoleNotifier()
        shown = 0
        for t in store.traders():
            positions = [p for p in client.positions(t["wallet"]) if not p.resolved]
            portfolio = sum(p.value for p in positions) or 1.0
            for p in sorted(positions, key=lambda p: -p.value):
                signal = Signal(
                    action=OPEN, trader=t["wallet"], trader_name=t["name"], asset=p.asset,
                    condition_id=p.condition_id, title=p.title, outcome=p.outcome, their_price=p.avg_price,
                    market_price=p.cur_price, their_usd=p.cost, their_weight=p.value / portfolio, url=p.url,
                    trader_summary=f"{t['name'] or t['wallet'][:10]}: {t['card'].summary()}" if t["card"] else "",
                )
                if p.cost < config.min_signal_usd or size_buy(signal, config).action != OPEN:
                    continue
                signal.notes.append("existing position: they entered earlier, so this is not a fresh signal")
                console.send(signal)
                shown += 1
        if not shown:
            print("Nothing open is still near the price the traders paid.")
        return 0

    if args.command == "signals":
        signals = store.recent_signals(args.limit)
        if not signals:
            print("No signals yet.")
        console = ConsoleNotifier()
        for signal in signals:
            console.send(signal)
        return 0

    if args.command == "test-alert":
        sample = Signal(
            action=OPEN, trader="0x" + "0" * 40, trader_name="test-trader", asset="test", condition_id="test",
            title="Test signal: is the copy trader wired up?", outcome="Yes", their_price=0.41, market_price=0.42,
            their_usd=38_400, their_weight=0.06, url="https://polymarket.com",
            trader_summary="test-trader: 68% win rate over 214 resolved, +11 pts vs prices paid, $1,200,000 profit",
        )
        notifiers = build_notifiers(config)
        dispatch(size_buy(sample, config), notifiers)
        print(f"Sent a test signal to: {', '.join(n.name for n in notifiers)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
