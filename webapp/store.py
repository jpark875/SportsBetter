"""
webapp/store.py

Persistence layer — a thin, dependency-free wrapper over SQLite (stdlib).

Two things are stored:

    profile   a single row of the user's financial parameters, kept in
              sync with :class:`config.settings.UserRiskProfile`.
    bets      the user's placed-bet ledger, which the risk-metrics module
              turns into Sharpe / Sortino / drawdown analytics.

SQLite was chosen over an ORM so the app runs with zero extra
dependencies and the database is a single portable file.  The path is
configurable via ``BETS_DB_PATH`` so tests can point at a temp file.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "data", "app.db")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """SQLite-backed store for the risk profile and bet ledger."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.getenv("BETS_DB_PATH", _DEFAULT_DB)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS profile (
                    id                  INTEGER PRIMARY KEY CHECK (id = 1),
                    liquid_bankroll     REAL NOT NULL,
                    disposable_income   REAL NOT NULL,
                    volatility_tolerance INTEGER NOT NULL,
                    updated_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bets (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at    TEXT NOT NULL,
                    sport         TEXT NOT NULL,
                    matchup       TEXT NOT NULL,
                    selection     TEXT NOT NULL,
                    bookmaker     TEXT,
                    price         REAL NOT NULL,      -- American odds
                    stake         REAL NOT NULL,
                    model_prob    REAL,
                    edge          REAL,
                    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|won|lost|void
                    settled_at    TEXT,
                    pnl           REAL                -- net profit/loss once settled
                );
                """
            )

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_profile(self) -> Dict:
        """Return the stored profile, or sensible defaults if none set yet."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()
        if row is None:
            return {
                "liquid_bankroll": 1000.0,
                "disposable_income": 200.0,
                "volatility_tolerance": 5,
                "updated_at": None,
            }
        return dict(row)

    def save_profile(
        self,
        liquid_bankroll: float,
        disposable_income: float,
        volatility_tolerance: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profile (id, liquid_bankroll, disposable_income,
                                     volatility_tolerance, updated_at)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    liquid_bankroll = excluded.liquid_bankroll,
                    disposable_income = excluded.disposable_income,
                    volatility_tolerance = excluded.volatility_tolerance,
                    updated_at = excluded.updated_at
                """,
                (liquid_bankroll, disposable_income, volatility_tolerance, _utcnow()),
            )

    # ------------------------------------------------------------------
    # Bets
    # ------------------------------------------------------------------

    def add_bet(
        self,
        sport: str,
        matchup: str,
        selection: str,
        price: float,
        stake: float,
        bookmaker: str = "",
        model_prob: Optional[float] = None,
        edge: Optional[float] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO bets (created_at, sport, matchup, selection, bookmaker,
                                  price, stake, model_prob, edge, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (_utcnow(), sport, matchup, selection, bookmaker,
                 price, stake, model_prob, edge),
            )
            return int(cur.lastrowid)

    def settle_bet(self, bet_id: int, status: str) -> None:
        """Mark a bet won/lost/void and compute its P&L."""
        if status not in {"won", "lost", "void"}:
            raise ValueError("status must be won, lost, or void")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT price, stake FROM bets WHERE id = ?", (bet_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No bet with id {bet_id}")
            pnl = self._settle_pnl(row["price"], row["stake"], status)
            conn.execute(
                "UPDATE bets SET status = ?, settled_at = ?, pnl = ? WHERE id = ?",
                (status, _utcnow(), pnl, bet_id),
            )

    @staticmethod
    def _settle_pnl(price: float, stake: float, status: str) -> float:
        if status == "void":
            return 0.0
        if status == "lost":
            return -stake
        # won
        if price >= 100:
            return stake * (price / 100.0)
        return stake * (100.0 / abs(price))

    def delete_bet(self, bet_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))

    def list_bets(self, status: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM bets"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def settled_bets_chronological(self) -> List[Dict]:
        """Settled, non-void bets oldest-first — the input for risk metrics."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM bets
                WHERE status IN ('won', 'lost')
                ORDER BY COALESCE(settled_at, created_at) ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]
