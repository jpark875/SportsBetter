"""SQLite: followed traders + scorecards, position snapshots, your copied book, signal log."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from copy_trader.polymarket import Position
from copy_trader.scoring import Scorecard
from copy_trader.signals import Signal


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CopyStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traders (
                    wallet     TEXT PRIMARY KEY,
                    name       TEXT NOT NULL DEFAULT '',
                    source     TEXT NOT NULL DEFAULT 'manual',
                    scorecard  TEXT,
                    scored_at  TEXT,
                    added_at   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    wallet     TEXT PRIMARY KEY,
                    positions  TEXT NOT NULL,
                    taken_at   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS copies (
                    wallet     TEXT NOT NULL,
                    asset      TEXT NOT NULL,
                    shares     REAL NOT NULL,
                    cost       REAL NOT NULL,
                    PRIMARY KEY (wallet, asset)
                );
                CREATE TABLE IF NOT EXISTS signals (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    wallet     TEXT NOT NULL,
                    action     TEXT NOT NULL,
                    payload    TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- traders -----------------------------------------------------------

    def follow(self, wallet: str, name: str = "", source: str = "manual", card: Optional[Scorecard] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO traders (wallet, name, source, added_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(wallet) DO UPDATE SET name = CASE WHEN excluded.name != '' THEN excluded.name ELSE name END",
                (wallet.lower(), name, source, _now()),
            )
        if card:
            self.save_scorecard(card)

    def unfollow(self, wallet: str) -> bool:
        wallet = wallet.lower()
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM traders WHERE wallet = ?", (wallet,))
            conn.execute("DELETE FROM snapshots WHERE wallet = ?", (wallet,))
            return bool(cur.rowcount)

    def save_scorecard(self, card: Scorecard) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE traders SET scorecard = ?, scored_at = ? WHERE wallet = ?",
                (json.dumps(card.to_dict()), _now(), card.wallet.lower()),
            )

    def traders(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM traders ORDER BY added_at").fetchall()
        out = []
        for r in rows:
            card = Scorecard.from_dict(json.loads(r["scorecard"])) if r["scorecard"] else None
            scored = datetime.fromisoformat(r["scored_at"]) if r["scored_at"] else None
            out.append({"wallet": r["wallet"], "name": r["name"], "source": r["source"], "card": card, "scored_at": scored})
        return out

    # --- snapshots ---------------------------------------------------------

    def snapshot(self, wallet: str) -> Optional[Dict[str, Position]]:
        with self._connect() as conn:
            row = conn.execute("SELECT positions FROM snapshots WHERE wallet = ?", (wallet,)).fetchone()
        if row is None:
            return None
        return {asset: Position(**data) for asset, data in json.loads(row["positions"]).items()}

    def save_snapshot(self, wallet: str, positions: Dict[str, Position]) -> None:
        payload = json.dumps({asset: asdict(p) for asset, p in positions.items()})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots (wallet, positions, taken_at) VALUES (?, ?, ?) "
                "ON CONFLICT(wallet) DO UPDATE SET positions = excluded.positions, taken_at = excluded.taken_at",
                (wallet, payload, _now()),
            )

    # --- your copied book --------------------------------------------------

    def copied_shares(self, wallet: str, asset: str) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT shares FROM copies WHERE wallet = ? AND asset = ?", (wallet, asset)).fetchone()
        return row["shares"] if row else 0.0

    def record_copy(self, signal: Signal) -> None:
        """Track the shares each signal told you to buy or sell, so exits can be sized."""
        if not signal.shares:
            return
        with self._connect() as conn:
            if signal.side == "BUY":
                conn.execute(
                    "INSERT INTO copies (wallet, asset, shares, cost) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(wallet, asset) DO UPDATE SET shares = shares + excluded.shares, cost = cost + excluded.cost",
                    (signal.trader, signal.asset, signal.shares, signal.stake or 0.0),
                )
            else:
                conn.execute(
                    "UPDATE copies SET shares = MAX(shares - ?, 0) WHERE wallet = ? AND asset = ?",
                    (signal.shares, signal.trader, signal.asset),
                )
                conn.execute("DELETE FROM copies WHERE shares <= 0.005")

    # --- signal log --------------------------------------------------------

    def log_signal(self, signal: Signal) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO signals (created_at, wallet, action, payload) VALUES (?, ?, ?, ?)",
                (signal.created_at.isoformat(), signal.trader, signal.action, json.dumps(signal.to_dict())),
            )

    def recent_signals(self, limit: int = 20) -> List[Signal]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM signals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Signal.from_dict(json.loads(r["payload"])) for r in rows]
