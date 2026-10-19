"""Copy trader: parsing, trader scoring, position diffs, order sizing, the poll loop and CLI, all offline."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest
import requests

from copy_trader import cli
from copy_trader.config import CopyConfig
from copy_trader.notify import ConsoleNotifier, NtfyNotifier, dispatch
from copy_trader.polymarket import ClosedPosition, PolymarketClient, Position, parse_closed, parse_position
from copy_trader.scoring import rank, score_trader, wilson_lower
from copy_trader.signals import (
    ADD, EXIT, OPEN, TRIM, WAIT, Signal, consensus_notes, diff_positions, format_signal, size_buy, size_sell,
)
from copy_trader.store import CopyStore
from copy_trader.watcher import CopyWatcher

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
SHARP = "0x" + "ab" * 20
OTHER = "0x" + "cd" * 20


def pos(asset="yes1", size=10_000, avg=0.40, cur=0.41, cond="c1", outcome="Yes", **kw):
    return Position(asset=asset, condition_id=cond, title="Will it rain?", outcome=outcome,
                    size=size, avg_price=avg, cur_price=cur, slug="rain", **kw)


def closed(avg, pnl, shares=1_000, days_ago=1):
    return ClosedPosition(title="m", outcome="Yes", avg_price=avg, shares=shares, realized_pnl=pnl,
                          timestamp=NOW - timedelta(days=days_ago))


def raw_pos(asset="yes1", size=10_000, avg=0.40, cur=0.41, cond="c1", outcome="Yes", redeemable=False):
    return {"proxyWallet": SHARP, "asset": asset, "conditionId": cond, "size": size, "avgPrice": avg,
            "curPrice": cur, "initialValue": size * avg, "currentValue": size * cur, "cashPnl": size * (cur - avg),
            "title": "Will it rain?", "outcome": outcome, "eventSlug": "rain", "redeemable": redeemable,
            "endDate": "2026-12-31"}


def raw_closed(avg, pnl, ts=None):
    return {"avgPrice": avg, "totalBought": 1_000, "realizedPnl": pnl, "title": "m", "outcome": "Yes",
            "timestamp": int((ts or NOW - timedelta(days=1)).timestamp())}


@pytest.fixture
def config(tmp_path):
    return CopyConfig(db_path=str(tmp_path / "copy.db"), bankroll_usd=2_000, min_resolved=10)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, route):
        self.route = route
        self.headers = {}
        self.gets = []
        self.posts = []

    def get(self, url, params=None, timeout=None):
        self.gets.append((url, dict(params or {})))
        result = self.route(url, params or {})
        return result if isinstance(result, FakeResponse) else FakeResponse(result)

    def post(self, url, timeout=None, **kw):
        self.posts.append((url, kw))
        return FakeResponse({})


# --------------------------------------------------------------------- parsing

def test_parse_position_and_closed_position():
    p = parse_position(raw_pos(redeemable=True))
    assert p.cost == pytest.approx(4_000) and p.value == pytest.approx(4_100)
    assert p.resolved and p.url == "https://polymarket.com/event/rain"
    c = parse_closed(raw_closed(0.3, 700))
    assert c.cost == pytest.approx(300) and c.timestamp == NOW - timedelta(days=1)
    assert parse_position({"size": 1}) is None and parse_closed({}) is None


def test_closed_positions_are_fetched_newest_first_not_by_profit():
    pages = {0: [raw_closed(0.5, 1)] * 50, 50: [raw_closed(0.5, -1)] * 10}
    session = FakeSession(lambda url, p: pages[p["offset"]])
    out = PolymarketClient(session=session).closed_positions(SHARP, limit=500)
    assert len(out) == 60
    assert session.gets[0][1]["sortBy"] == "TIMESTAMP"


def test_positions_refuses_a_truncated_list():
    session = FakeSession(lambda url, p: [raw_pos(asset=str(i)) for i in range(500)])
    with pytest.raises(RuntimeError):
        PolymarketClient(session=session).positions(SHARP, max_pages=1)


def test_client_retries_rate_limits():
    replies = iter([FakeResponse({}, 429), FakeResponse([raw_pos()])])
    client = PolymarketClient(session=FakeSession(lambda url, p: next(replies)), sleep=lambda s: None)
    assert len(client.positions(SHARP)) == 1


# --------------------------------------------------------------------- scoring

def test_wilson_lower_penalises_small_samples():
    assert wilson_lower(9, 10) < wilson_lower(90, 100) < 0.9
    assert wilson_lower(0, 0) == 0


def test_sharp_trader_qualifies(config):
    history = [closed(0.40, 600) for _ in range(14)] + [closed(0.40, -400) for _ in range(6)]
    card = score_trader(SHARP, history, [pos()], config, leaderboard_pnl=250_000, now=NOW)
    assert card.resolved == 20 and card.win_rate == pytest.approx(0.70)
    assert card.edge == pytest.approx(0.30)
    assert card.qualifies, card.failures


def test_favourite_buyer_with_high_win_rate_is_rejected(config):
    """96% wins at 95¢ is no edge at all."""
    history = [closed(0.95, 50) for _ in range(48)] + [closed(0.95, -950) for _ in range(2)]
    card = score_trader(SHARP, history, [], config, leaderboard_pnl=100_000, now=NOW)
    assert card.win_rate == pytest.approx(0.96)
    assert not card.qualifies
    assert any("edge" in f for f in card.failures)


def test_scoring_counts_unredeemed_resolved_positions_and_checks_activity(config):
    history = [closed(0.3, 700, days_ago=90) for _ in range(12)]
    won = pos(cur=1.0, redeemable=True, cash_pnl=6_000)
    card = score_trader(SHARP, history, [won], config, now=NOW)
    assert card.resolved == 13 and card.open_positions == 0
    assert card.profit == pytest.approx(12 * 700 + 6_000)
    assert any("inactive" in f for f in card.failures)


def test_rank_puts_qualifiers_first(config):
    good = score_trader(SHARP, [closed(0.4, 1) for _ in range(20)], [pos()], config, leaderboard_pnl=1e6, now=NOW)
    bad = score_trader(OTHER, [closed(0.4, 1)], [], config, now=NOW)
    assert [c.wallet for c in rank([bad, good])] == [SHARP, OTHER]


# ---------------------------------------------------------------------- diffs

def _snap(*positions):
    return {p.asset: p for p in positions}


def test_new_position_is_an_open_signal(config):
    other = pos(asset="x", size=10_000, avg=0.5, cur=0.5)
    [s] = diff_positions(SHARP, _snap(other), _snap(other, pos(size=10_000, avg=0.40)), config)
    assert s.action == OPEN and s.their_price == pytest.approx(0.40)
    assert s.their_usd == pytest.approx(4_000)
    assert s.their_weight == pytest.approx(4_000 / (5_000 + 4_100))


def test_bigger_position_recovers_their_fill_price(config):
    before = pos(size=10_000, avg=0.40)
    after = pos(size=20_000, avg=0.45, cur=0.50)  # bought 10k more at 50¢
    [s] = diff_positions(SHARP, _snap(before), _snap(after), config)
    assert s.action == ADD and s.their_price == pytest.approx(0.50)


def test_trims_exits_and_resolutions(config):
    a, b, c = pos(asset="a"), pos(asset="b"), pos(asset="c", cur=1.0)
    small_trim = pos(asset="d", size=9_500)
    old = _snap(a, b, c, pos(asset="d"))
    new = _snap(pos(asset="a", size=4_000), small_trim)
    signals = {s.asset: s for s in diff_positions(SHARP, old, new, config)}
    assert signals["a"].action == TRIM and signals["a"].sell_fraction == pytest.approx(0.6)
    assert signals["b"].action == EXIT and signals["b"].sell_fraction == 1.0
    assert "c" not in signals  # market resolved: a payout, not a sale
    assert "d" not in signals  # 5% trim is noise


def test_tiny_moves_are_ignored(config):
    assert diff_positions(SHARP, {}, _snap(pos(size=100)), config) == []


# --------------------------------------------------------------------- sizing

def _buy(their=0.40, now=0.41, weight=0.02):
    return Signal(action=OPEN, trader=SHARP, trader_name="sharp", asset="yes1", condition_id="c1",
                  title="Will it rain?", outcome="Yes", their_price=their, market_price=now,
                  their_usd=40_000, their_weight=weight)


def test_buy_mirrors_their_portfolio_weight(config):
    s = size_buy(_buy(weight=0.02), config)
    assert s.action == OPEN and s.limit_price == pytest.approx(0.42)
    assert s.stake == pytest.approx(40, abs=0.01)  # 2% of $2,000
    assert s.shares == pytest.approx(95.23)


def test_buy_is_capped(config):
    s = size_buy(_buy(weight=0.30), config)
    assert s.stake == pytest.approx(100, abs=0.01)
    assert any("capped at 5%" in n for n in s.notes)


def test_dont_chase_a_price_that_ran(config):
    s = size_buy(_buy(their=0.40, now=0.48), config)
    assert s.action == WAIT and s.shares is None
    assert "don't chase" in s.notes[0] and "45¢" in s.notes[0]


def test_dont_buy_near_certainties(config):
    assert size_buy(_buy(their=0.96, now=0.985), config).action == WAIT


def test_order_too_small_for_bankroll(config):
    config.bankroll_usd = 20
    s = size_buy(_buy(weight=0.01), config)
    assert s.action == WAIT and "minimum" in s.notes[0]


def test_sell_uses_copied_shares(config):
    s = Signal(action=TRIM, trader=SHARP, trader_name="", asset="yes1", condition_id="c1", title="t",
               outcome="Yes", their_price=0.6, market_price=0.6, their_usd=6_000, sell_fraction=0.5)
    size_sell(s, 200, config)
    assert s.shares == 100 and s.limit_price == pytest.approx(0.59)
    blind = size_sell(Signal(**{**s.to_dict(), "created_at": s.created_at, "shares": None, "notes": []}), 0, config)
    assert blind.shares is None and "no copied shares" in blind.notes[0]


def test_consensus_notes(config):
    s = _buy()
    consensus_notes(s, [_snap(pos()), _snap(pos(asset="no1", outcome="No")), _snap(pos(cond="other"))])
    assert any("1 other followed trader also holds Yes" in n for n in s.notes)
    assert any("other side" in n for n in s.notes)


def test_format_signal_reads_like_an_order(config):
    title, body = format_signal(size_buy(_buy(), config))
    assert title == "COPY ▸ BUY 95.23 Yes @ limit 42¢ (≈$40.00)"
    assert "opened $40,000 at ~40¢" in body
    wait_title, _ = format_signal(size_buy(_buy(now=0.6), config))
    assert wait_title.startswith("WAIT ▸")


def test_ntfy_title_is_latin1_safe(config):
    session = FakeSession(lambda u, p: {})
    NtfyNotifier("t", session=session).send(size_buy(_buy(), config))
    session.posts[0][1]["headers"]["Title"].encode("latin-1")


def test_dispatch_survives_broken_channel(config):
    class Broken(ConsoleNotifier):
        name = "broken"

        def send(self, signal):
            raise RuntimeError("down")

    out = io.StringIO()
    dispatch(size_buy(_buy(), config), [Broken(), ConsoleNotifier(stream=out)])
    assert "COPY ▸ BUY" in out.getvalue()


# -------------------------------------------------------------------- watcher

class World:
    """A fake Polymarket whose positions can change between polls."""

    def __init__(self):
        self.positions = {SHARP: [raw_pos(asset="base", size=50_000, avg=0.5, cur=0.5, cond="c0")]}
        self.closed = {SHARP: [raw_closed(0.4, 600) for _ in range(14)] + [raw_closed(0.4, -400) for _ in range(6)]}
        self.down = set()

    def route(self, url, params):
        user = params.get("user")
        if user in self.down:
            return FakeResponse({}, 503)
        if url.endswith("/positions"):
            return self.positions.get(user, [])
        if url.endswith("/closed-positions"):
            return self.closed.get(user, []) if params["offset"] == 0 else []
        if url.endswith("/v1/leaderboard"):
            return [{"rank": 1, "proxyWallet": SHARP, "userName": "sharp", "pnl": 900_000, "vol": 5e6}]
        return FakeResponse({}, 404)


def _watcher(config, world, source="discover"):
    store = CopyStore(config.db_path)
    store.follow(SHARP, "sharp", source=source)
    out = io.StringIO()
    client = PolymarketClient(session=FakeSession(world.route), sleep=lambda s: None)
    return CopyWatcher(config, store, [ConsoleNotifier(stream=out)], client=client, clock=lambda: NOW), out


def test_watcher_baseline_then_copy_then_exit(config):
    world = World()
    watcher, out = _watcher(config, world)
    assert watcher.poll_once() == []  # first look is a baseline, not a buy signal

    world.positions[SHARP].append(raw_pos(asset="yes1", size=10_000, avg=0.40, cur=0.41))
    [buy] = watcher.poll_once()
    assert buy.action == OPEN and buy.shares > 0
    assert "70% win rate over 20 resolved" in buy.trader_summary
    assert watcher.store.copied_shares(SHARP, "yes1") == pytest.approx(buy.shares)

    world.positions[SHARP] = world.positions[SHARP][:1]
    [sell] = watcher.poll_once()
    assert sell.action == EXIT and sell.shares == pytest.approx(buy.shares)
    assert watcher.store.copied_shares(SHARP, "yes1") == 0
    assert [s.action for s in watcher.store.recent_signals()] == [EXIT, OPEN]
    assert out.getvalue().count("COPY ▸") == 2


def test_failed_fetch_is_not_a_sell_off(config):
    world = World()
    watcher, _ = _watcher(config, world)
    watcher.poll_once()
    world.down.add(SHARP)
    assert watcher.poll_once() == []
    world.down.clear()
    assert watcher.poll_once() == []


def test_traders_who_stop_qualifying_are_not_copied(config):
    world = World()
    world.closed[SHARP] = [raw_closed(0.4, -100) for _ in range(20)]
    watcher, _ = _watcher(config, world)
    watcher.poll_once()
    world.positions[SHARP].append(raw_pos(asset="yes1"))
    assert watcher.poll_once() == []


def test_manual_follows_are_copied_with_a_warning(config):
    world = World()
    world.closed[SHARP] = [raw_closed(0.4, -100) for _ in range(20)]
    watcher, _ = _watcher(config, world, source="manual")
    watcher.poll_once()
    world.positions[SHARP].append(raw_pos(asset="yes1"))
    [s] = watcher.poll_once()
    assert any("below your copy bar" in n for n in s.notes)


# ------------------------------------------------------------------------ cli

def test_cli_discover_follows_qualifiers(config, capsys, monkeypatch):
    monkeypatch.setenv("COPY_MIN_RESOLVED", "10")
    client = PolymarketClient(session=FakeSession(World().route), sleep=lambda s: None)
    assert cli.main(["--db", config.db_path, "discover", "--follow", "5"], client=client) == 0
    out = capsys.readouterr().out
    assert "✔ copyable" in out and "Following sharp" in out
    assert [t["wallet"] for t in CopyStore(config.db_path).traders()] == [SHARP]


def test_cli_follow_validates_and_open_lists_positions(config, capsys, monkeypatch):
    monkeypatch.setenv("COPY_MIN_RESOLVED", "10")
    world = World()
    world.positions[SHARP].append(raw_pos(asset="yes1", size=10_000, avg=0.40, cur=0.41))
    client = PolymarketClient(session=FakeSession(world.route), sleep=lambda s: None)
    assert cli.main(["--db", config.db_path, "follow", "nope"], client=client) == 2
    assert cli.main(["--db", config.db_path, "follow", SHARP, "--name", "sharp"], client=client) == 0
    capsys.readouterr()
    assert cli.main(["--db", config.db_path, "--bankroll", "2000", "open"], client=client) == 0
    out = capsys.readouterr().out
    assert out.count("COPY ▸ BUY") == 2 and "existing position" in out
