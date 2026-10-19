"""SQLite persistence: trades already seen, alerts sent, followed wallets, poll cursors."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set

from whale_tracker.models import Alert


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WhaleStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS seen_trades (
                    platform  TEXT NOT NULL,
                    trade_id  TEXT NOT NULL,
                    seen_at   TEXT NOT NULL,
                    PRIMARY KEY (platform, trade_id)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    key        TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    platform   TEXT NOT NULL,
                    kind       TEXT NOT NULL,
                    notional   REAL NOT NULL,
                    payload    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS watchlist (
                    platform  TEXT NOT NULL,
                    trader    TEXT NOT NULL,
                    label     TEXT NOT NULL DEFAULT '',
                    source    TEXT NOT NULL DEFAULT 'manual',
                    added_at  TEXT NOT NULL,
                    PRIMARY KEY (platform, trader)
                );
                CREATE TABLE IF NOT EXISTS cursors (
                    platform  TEXT PRIMARY KEY,
                    last_ts   TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def filter_unseen(self, platform: str, trade_ids: Iterable[str]) -> Set[str]:
        """Record ``trade_ids`` as seen and return the ones that were new."""
        ids = list(dict.fromkeys(trade_ids))
        if not ids:
            return set()
        now = _now().isoformat()
        new: Set[str] = set()
        with self._connect() as conn:
            for trade_id in ids:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO seen_trades (platform, trade_id, seen_at) VALUES (?, ?, ?)",
                    (platform, trade_id, now),
                )
                if cur.rowcount:
                    new.add(trade_id)
        return new

    def prune(self, keep_days: int = 7) -> None:
        cutoff = (_now() - timedelta(days=keep_days)).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM seen_trades WHERE seen_at < ?", (cutoff,))

    def save_alert(self, alert: Alert) -> bool:
        """Store ``alert``; False if an identical one was already stored."""
        payload = asdict(alert)
        payload["first_ts"] = alert.first_ts.isoformat()
        payload["last_ts"] = alert.last_ts.isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO alerts (key, created_at, platform, kind, notional, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (alert.key, _now().isoformat(), alert.platform, alert.kind, alert.notional, json.dumps(payload)),
            )
            return bool(cur.rowcount)

    def recent_alerts(self, limit: int = 20, platform: Optional[str] = None) -> List[Alert]:
        query = "SELECT payload FROM alerts"
        args: list = []
        if platform:
            query += " WHERE platform = ?"
            args.append(platform)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        alerts = []
        for row in rows:
            data = json.loads(row["payload"])
            data["first_ts"] = datetime.fromisoformat(data["first_ts"])
            data["last_ts"] = datetime.fromisoformat(data["last_ts"])
            alerts.append(Alert(**data))
        return alerts

    def follow(self, platform: str, trader: str, label: str = "", source: str = "manual") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO watchlist (platform, trader, label, source, added_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(platform, trader) DO UPDATE SET label = excluded.label, source = excluded.source",
                (platform, trader.lower(), label, source, _now().isoformat()),
            )

    def unfollow(self, platform: str, trader: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM watchlist WHERE platform = ? AND trader = ?", (platform, trader.lower()))
            return bool(cur.rowcount)

    def watchlist(self, platform: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """``{trader: {"platform", "label", "source"}}``."""
        query = "SELECT platform, trader, label, source FROM watchlist"
        args: list = []
        if platform:
            query += " WHERE platform = ?"
            args.append(platform)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY added_at", args).fetchall()
        return {r["trader"]: {"platform": r["platform"], "label": r["label"], "source": r["source"]} for r in rows}

    def get_cursor(self, platform: str) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute("SELECT last_ts FROM cursors WHERE platform = ?", (platform,)).fetchone()
        return datetime.fromisoformat(row["last_ts"]) if row else None

    def set_cursor(self, platform: str, ts: datetime) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cursors (platform, last_ts) VALUES (?, ?) "
                "ON CONFLICT(platform) DO UPDATE SET last_ts = excluded.last_ts",
                (platform, ts.isoformat()),
            )
