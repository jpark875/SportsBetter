"""Whale tracker: API parsing, detection rules, storage, notifiers and the poll loop, all offline."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest
import requests

from whale_tracker import cli
from whale_tracker.config import TrackerConfig
from whale_tracker.detector import WhaleDetector
from whale_tracker.models import (
    ACCUMULATION, KALSHI, LARGE_TRADE, POLYMARKET, SWEEP, WATCHED_WALLET, Trade,
)
from whale_tracker.notify import (
    ConsoleNotifier, DiscordNotifier, NtfyNotifier, SlackNotifier, TelegramNotifier, dispatch, format_alert,
)
from whale_tracker.sources import KalshiClient, PolymarketClient
from whale_tracker.sources import kalshi as kalshi_src
from whale_tracker.sources import polymarket as poly_src
from whale_tracker.sources.http import get_json
from whale_tracker.store import WhaleStore
from whale_tracker.watcher import WhaleWatcher

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
WHALE = "0x" + "ab" * 20
MINNOW = "0x" + "cd" * 20


def poly_raw(size, price, wallet=WHALE, seconds_ago=0, tx="0xtx1", side="BUY", outcome="Yes", **extra):
    row = {
        "proxyWallet": wallet,
        "side": side,
        "asset": "1234",
        "conditionId": "0xcond",
        "size": size,
        "price": price,
        "timestamp": int((NOW - timedelta(seconds=seconds_ago)).timestamp()),
        "title": "Will it rain in NYC tomorrow?",
        "slug": "rain-nyc",
        "eventSlug": "weather-nyc",
        "outcome": outcome,
        "outcomeIndex": 0,
        "name": "bigfish",
        "pseudonym": "Grand-Wombat",
        "transactionHash": tx,
    }
    row.update(extra)
    return row


def kalshi_raw(trade_id, count, yes_cents, taker="yes", seconds_ago=0, ticker="KXFED-26DEC-T4.00", dollars=True):
    row = {
        "trade_id": trade_id,
        "ticker": ticker,
        "taker_side": taker,
        "created_time": (NOW - timedelta(seconds=seconds_ago)).isoformat().replace("+00:00", "Z"),
    }
    if dollars:
        row.update({
            "count_fp": f"{count:.2f}",
            "yes_price_dollars": f"{yes_cents / 100:.4f}",
            "no_price_dollars": f"{(100 - yes_cents) / 100:.4f}",
        })
    else:
        row.update({"count": count, "yes_price": yes_cents, "no_price": 100 - yes_cents})
    return row


def trade(notional, price=0.5, platform=POLYMARKET, trader=WHALE, seconds=0, tid=None, outcome="Yes", market="m1"):
    return Trade(
        platform=platform,
        trade_id=tid or f"t{notional}-{seconds}-{trader}",
        market_id=market,
        market_title="Market",
        outcome=outcome,
        side="BUY",
        price=price,
        size=notional / price,
        timestamp=NOW + timedelta(seconds=seconds),
        trader=trader if platform == POLYMARKET else None,
    )


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Routes GETs by path to canned payloads (or callables of params) and records POSTs."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.gets = []
        self.posts = []

    def get(self, url, params=None, timeout=None):
        self.gets.append((url, dict(params or {})))
        for suffix, payload in self.routes.items():
            if url.endswith(suffix):
                value = payload(params or {}) if callable(payload) else payload
                return value if isinstance(value, FakeResponse) else FakeResponse(value)
        return FakeResponse({}, status=404)

    def post(self, url, timeout=None, **kwargs):
        self.posts.append((url, kwargs))
        return FakeResponse({})


@pytest.fixture
def config(tmp_path):
    return TrackerConfig(db_path=str(tmp_path / "whales.db"), platforms=[POLYMARKET, KALSHI])


# --------------------------------------------------------------------- parsing

def test_polymarket_trade_parses_documented_fields():
    t = poly_src.parse_trade(poly_raw(20_000, 0.6))
    assert t.platform == POLYMARKET
    assert t.notional == pytest.approx(12_000)
    assert t.trader == WHALE
    assert t.trader_name == "bigfish"
    assert t.outcome == "Yes" and t.side == "BUY"
    assert t.url == "https://polymarket.com/event/weather-nyc"
    assert t.timestamp == NOW


def test_polymarket_parse_handles_millisecond_timestamps_and_junk():
    raw = poly_raw(10, 0.5)
    raw["timestamp"] = int(NOW.timestamp() * 1000)
    assert poly_src.parse_trade(raw).timestamp == NOW
    assert poly_src.parse_trade({"size": "abc"}) is None


def test_kalshi_trade_parses_dollar_and_legacy_cent_fields():
    yes = kalshi_src.parse_trade(kalshi_raw("a", 1000, 40))
    assert yes.outcome == "Yes" and yes.price == pytest.approx(0.40)
    assert yes.notional == pytest.approx(400)
    assert yes.trader is None
    assert yes.url == "https://kalshi.com/markets/kxfed"

    no = kalshi_src.parse_trade(kalshi_raw("b", 500, 40, taker="no", dollars=False))
    assert no.outcome == "No" and no.price == pytest.approx(0.60)
    assert no.notional == pytest.approx(300)
    assert kalshi_src.parse_trade({"ticker": "X"}) is None


def test_get_json_retries_on_rate_limit():
    responses = iter([FakeResponse({}, status=429, headers={"Retry-After": "1"}), FakeResponse({"ok": True})])
    session = FakeSession({"/x": lambda p: next(responses)})
    waits = []
    assert get_json(session, "https://h/x", sleep=waits.append) == {"ok": True}
    assert waits == [1.0]


def test_polymarket_recent_trades_sends_cash_filter_and_drops_old_rows():
    session = FakeSession({"/trades": [poly_raw(30_000, 0.5), poly_raw(30_000, 0.5, tx="old", seconds_ago=3600)]})
    trades = PolymarketClient(session=session).recent_trades(1_000, since=NOW - timedelta(minutes=5))
    assert [t.trade_id.split(":")[0] for t in trades] == ["0xtx1"]
    params = session.gets[0][1]
    assert params["filterType"] == "CASH" and params["filterAmount"] == 1_000
    assert params["takerOnly"] == "true"


def test_polymarket_recent_trades_pages_until_it_reaches_since(monkeypatch):
    monkeypatch.setattr(poly_src, "PAGE_SIZE", 2)
    pages = {
        0: [poly_raw(1, 1, tx="a"), poly_raw(1, 1, tx="b", seconds_ago=10)],
        2: [poly_raw(1, 1, tx="c", seconds_ago=20), poly_raw(1, 1, tx="d", seconds_ago=999)],
    }
    session = FakeSession({"/trades": lambda p: pages.get(p["offset"], [])})
    trades = PolymarketClient(session=session).recent_trades(1, since=NOW - timedelta(seconds=60))
    assert [t.trade_id[0] for t in trades] == ["a", "b", "c"]
    assert len(session.gets) == 2


def test_polymarket_leaderboard_normalises_rows():
    session = FakeSession({"/v1/leaderboard": [
        {"rank": "1", "proxyWallet": WHALE.upper().replace("0X", "0x"), "userName": "bigfish", "pnl": 1e6, "vol": 5e6},
        {"rank": "2", "userName": "no-wallet"},
    ]})
    leaders = PolymarketClient(session=session).leaderboard("week", "vol", 10)
    assert leaders == [{"wallet": WHALE, "name": "bigfish", "pnl": 1e6, "volume": 5e6, "rank": 1}]
    assert session.gets[0][1]["timePeriod"] == "WEEK"


def test_kalshi_follows_cursor_and_caches_titles():
    pages = {None: {"trades": [kalshi_raw("1", 10, 50)], "cursor": "next"}, "next": {"trades": [kalshi_raw("2", 10, 50)], "cursor": ""}}
    session = FakeSession({
        "/markets/trades": lambda p: pages[p.get("cursor")],
        "/markets/KXFED-26DEC-T4.00": {"market": {"title": "Fed rate in Dec?", "yes_sub_title": "Above 4.00%"}},
    })
    client = KalshiClient(session=session)
    assert [t.trade_id for t in client.recent_trades(NOW - timedelta(minutes=1))] == ["1", "2"]
    assert client.market_title("KXFED-26DEC-T4.00") == "Fed rate in Dec? — Above 4.00%"
    client.market_title("KXFED-26DEC-T4.00")
    assert sum(1 for url, _ in session.gets if url.endswith("T4.00")) == 1


# ------------------------------------------------------------------ detection

def test_large_single_trade_alerts(config):
    alerts = WhaleDetector(config).process([trade(9_999), trade(10_000, seconds=1)])
    assert [a.kind for a in alerts] == [LARGE_TRADE]
    assert alerts[0].notional == pytest.approx(10_000)


def test_split_orders_from_one_wallet_accumulate(config):
    fills = [trade(3_000, seconds=s * 60) for s in range(4)]
    alerts = WhaleDetector(config).process(fills)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.kind == ACCUMULATION and a.trade_count == 4 and a.notional == pytest.approx(12_000)
    assert "4 smaller fills" in a.reasons[0]


def test_accumulation_ignores_fills_outside_window_and_other_wallets(config):
    config.accumulation_window_seconds = 600
    fills = [trade(6_000, seconds=0), trade(6_000, seconds=601), trade(6_000, trader=MINNOW, seconds=602)]
    assert WhaleDetector(config).process(fills) == []


def test_accumulation_window_resets_after_firing(config):
    fills = [trade(5_000, seconds=s) for s in range(5)]
    alerts = WhaleDetector(config).process(fills)
    # Five $5k fills: two $10k alerts, and the fifth fill waits for more.
    assert [a.trade_count for a in alerts] == [2, 2]
    assert set(alerts[0].trade_ids).isdisjoint(alerts[1].trade_ids)


def test_kalshi_sweep_needs_fills_close_together(config):
    fast = [trade(4_000, platform=KALSHI, seconds=s) for s in (0, 1, 2)]
    slow = [trade(4_000, platform=KALSHI, seconds=s, market="m2") for s in (0, 30, 60)]
    alerts = WhaleDetector(config).process(fast + slow)
    assert [(a.kind, a.market_id) for a in alerts] == [(SWEEP, "m1")]
    assert "likely one large order" in alerts[0].reasons[0]


def test_kalshi_has_its_own_threshold(config):
    config.kalshi_min_usd = 2_000
    alerts = WhaleDetector(config).process([trade(2_500, platform=KALSHI), trade(2_500)])
    assert [a.platform for a in alerts] == [KALSHI]


def test_followed_wallet_alerts_at_lower_bar(config):
    detector = WhaleDetector(config, watchlist={WHALE: {"label": "leaderboard #1"}})
    alerts = detector.process([trade(1_500), trade(1_500, trader=MINNOW)])
    assert [(a.kind, a.trader) for a in alerts] == [(WATCHED_WALLET, WHALE)]
    assert "leaderboard #1" in alerts[0].reasons[0]


def test_reasons_flag_longshots_and_fresh_wallets(config):
    detector = WhaleDetector(config, wallet_trade_count=lambda w: 2)
    alert = detector.process([trade(20_000, price=0.08)])[0]
    assert any("longshot at 8¢" in r and "12.5x" in r for r in alert.reasons)
    assert any("new wallet: only 2 trades" in r for r in alert.reasons)

    seasoned = WhaleDetector(config, wallet_trade_count=lambda w: 100).process([trade(20_000, price=0.95)])[0]
    assert any("heavy favourite" in r for r in seasoned.reasons)
    assert not any("new wallet" in r for r in seasoned.reasons)


# ---------------------------------------------------------------------- store

def test_store_dedupes_trades_and_alerts(tmp_path):
    store = WhaleStore(str(tmp_path / "w.db"))
    assert store.filter_unseen(POLYMARKET, ["a", "b"]) == {"a", "b"}
    assert store.filter_unseen(POLYMARKET, ["b", "c"]) == {"c"}
    assert store.filter_unseen(KALSHI, ["a"]) == {"a"}

    alert = WhaleDetector(TrackerConfig()).process([trade(50_000)])[0]
    assert store.save_alert(alert) is True
    assert store.save_alert(alert) is False
    [loaded] = store.recent_alerts()
    assert loaded.key == alert.key and loaded.first_ts == alert.first_ts


def test_store_watchlist_and_cursor(tmp_path):
    store = WhaleStore(str(tmp_path / "w.db"))
    store.follow(POLYMARKET, WHALE.upper().replace("0X", "0x"), "big")
    store.follow(POLYMARKET, WHALE, "renamed")
    assert store.watchlist() == {WHALE: {"platform": POLYMARKET, "label": "renamed", "source": "manual"}}
    assert store.unfollow(POLYMARKET, WHALE) and not store.unfollow(POLYMARKET, WHALE)
    assert store.get_cursor(KALSHI) is None
    store.set_cursor(KALSHI, NOW)
    assert store.get_cursor(KALSHI) == NOW


# ------------------------------------------------------------------ notifiers

def _alert():
    return WhaleDetector(TrackerConfig()).process([trade(48_200, price=0.62)])[0]


def test_format_alert_reads_well():
    title, body = format_alert(_alert())
    assert title == "🐋 Whale trade on Polymarket: $48,200 BUY Yes @ 62¢"
    assert "https://polymarket.com/profile/" + WHALE in body
    assert "Why: single fill ≥ $10,000" in body


def test_http_notifiers_post_expected_payloads():
    session = FakeSession()
    alert = _alert()
    DiscordNotifier("https://discord/hook", session).send(alert)
    SlackNotifier("https://slack/hook", session).send(alert)
    NtfyNotifier("my-whales", session=session).send(alert)
    TelegramNotifier("TOKEN", "42", session).send(alert)
    (d_url, d), (s_url, s), (n_url, n), (t_url, t) = session.posts
    assert d["json"]["content"].startswith("**🐋 Whale trade")
    assert s["json"]["text"].startswith("*🐋 Whale trade")
    assert n_url == "https://ntfy.sh/my-whales"
    n["headers"]["Title"].encode("latin-1")
    assert t_url == "https://api.telegram.org/botTOKEN/sendMessage" and t["json"]["chat_id"] == "42"


def test_dispatch_survives_a_broken_channel():
    class Broken(ConsoleNotifier):
        name = "broken"

        def send(self, alert):
            raise RuntimeError("webhook down")

    out = io.StringIO()
    dispatch(_alert(), [Broken(), ConsoleNotifier(stream=out)])
    assert "Whale trade" in out.getvalue()


# -------------------------------------------------------------------- watcher

def _watcher(config, poly_rows, kalshi_rows, clock=lambda: NOW):
    poly = PolymarketClient(session=FakeSession({"/trades": lambda p: poly_rows() if callable(poly_rows) else poly_rows}))
    kal = KalshiClient(session=FakeSession({
        "/markets/trades": {"trades": kalshi_rows, "cursor": ""},
        "/markets/KXFED-26DEC-T4.00": {"market": {"title": "Fed rate in Dec?"}},
    }))
    out = io.StringIO()
    store = WhaleStore(config.db_path)
    return WhaleWatcher(config, store, [ConsoleNotifier(stream=out)], polymarket=poly, kalshi=kal, clock=clock), out


def test_watcher_alerts_once_per_trade_and_fills_kalshi_titles(config):
    poly_rows = [poly_raw(40_000, 0.5)]
    kalshi_rows = [kalshi_raw(str(i), 10_000, 50, seconds_ago=1) for i in range(3)]
    watcher, out = _watcher(config, poly_rows, kalshi_rows)

    first = watcher.poll_once()
    assert sorted(a.kind for a in first) == [LARGE_TRADE, SWEEP]
    assert next(a for a in first if a.platform == KALSHI).market_title == "Fed rate in Dec?"
    assert watcher.poll_once() == []
    assert out.getvalue().count("🐋") == 2
    assert len(watcher.store.recent_alerts()) == 2


def test_watcher_cold_start_ignores_history_older_than_lookback(config):
    config.lookback_seconds = 60
    watcher, _ = _watcher(config, [poly_raw(40_000, 0.5, seconds_ago=3600)], [])
    assert watcher.poll_once() == []


def test_watcher_keeps_going_when_a_platform_is_down(config):
    def down():
        raise requests.ConnectionError("no route")

    watcher, _ = _watcher(config, down, [kalshi_raw("1", 30_000, 50)])
    assert [a.platform for a in watcher.poll_once()] == [KALSHI]


def test_watcher_polls_followed_wallets(config):
    config.platforms = [POLYMARKET]
    session = FakeSession({"/trades": lambda p: [poly_raw(2_000, 0.5, tx="small")] if p.get("user") == WHALE else []})
    store = WhaleStore(config.db_path)
    store.follow(POLYMARKET, WHALE, "friend")
    watcher = WhaleWatcher(config, store, [], polymarket=PolymarketClient(session=session), clock=lambda: NOW)
    [alert] = watcher.poll_once()
    assert alert.kind == WATCHED_WALLET and alert.notional == pytest.approx(1_000)


# ------------------------------------------------------------------------ cli

def test_cli_follow_list_unfollow(tmp_path, capsys):
    db = str(tmp_path / "w.db")
    assert cli.main(["--db", db, "follow", "not-a-wallet"]) == 2
    assert cli.main(["--db", db, "follow", WHALE, "--label", "big"]) == 0
    cli.main(["--db", db, "following"])
    assert f"{WHALE}  big  [manual]" in capsys.readouterr().out
    assert cli.main(["--db", db, "unfollow", WHALE]) == 0
    assert cli.main(["--db", db, "unfollow", WHALE]) == 1


def test_cli_recent_prints_stored_alerts(tmp_path, capsys):
    db = str(tmp_path / "w.db")
    WhaleStore(db).save_alert(_alert())
    assert cli.main(["--db", db, "recent"]) == 0
    assert "$48,200" in capsys.readouterr().out
