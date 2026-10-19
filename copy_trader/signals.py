"""Turn changes in a trader's holdings into orders you can place.

Each poll snapshots every followed trader's positions and compares them with
the last snapshot:

* new position → **OPEN** (buy)      * bigger position → **ADD** (buy)
* smaller      → **TRIM** (sell part) * gone, unresolved → **EXIT** (sell all)

Positions that vanish because the market resolved are payouts, not trades,
and are ignored. Buys are sized to the same share of your bankroll as the
trader's move is of their portfolio, then capped. If the price has already run
past their entry by more than ``max_chase``, the signal is **WAIT** instead.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from copy_trader.config import CopyConfig
from copy_trader.polymarket import Position, profile_url

OPEN, ADD, TRIM, EXIT, WAIT = "OPEN", "ADD", "TRIM", "EXIT", "WAIT"
BUYS = (OPEN, ADD)
SELLS = (TRIM, EXIT)


@dataclass
class Signal:
    action: str
    trader: str
    trader_name: str
    asset: str
    condition_id: str
    title: str
    outcome: str
    their_price: float
    market_price: float
    their_usd: float
    their_weight: float = 0.0
    sell_fraction: float = 0.0
    limit_price: Optional[float] = None
    shares: Optional[float] = None
    stake: Optional[float] = None
    url: str = ""
    notes: List[str] = field(default_factory=list)
    trader_summary: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def side(self) -> str:
        return "SELL" if self.action in SELLS else "BUY"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Signal":
        data = dict(data)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


def _tick_down(price: float) -> float:
    return math.floor(round(price * 100, 6)) / 100


def _tick_up(price: float) -> float:
    return math.ceil(round(price * 100, 6)) / 100


def diff_positions(
    trader: str,
    old: Dict[str, Position],
    new: Dict[str, Position],
    config: CopyConfig,
    trader_name: str = "",
) -> List[Signal]:
    """Signals for what ``trader`` did between two snapshots (keyed by asset)."""
    portfolio = sum(p.value for p in new.values() if not p.resolved) or 1.0
    signals: List[Signal] = []

    def make(action: str, pos: Position, price: float, usd: float, **extra) -> Signal:
        return Signal(
            action=action, trader=trader, trader_name=trader_name, asset=pos.asset,
            condition_id=pos.condition_id, title=pos.title, outcome=pos.outcome,
            their_price=price, market_price=pos.cur_price, their_usd=usd, url=pos.url, **extra,
        )

    for asset, pos in new.items():
        if pos.resolved:
            continue
        before = old.get(asset)
        if before is None:
            usd = pos.cost
            if usd >= config.min_signal_usd:
                signals.append(make(OPEN, pos, pos.avg_price, usd, their_weight=usd / portfolio))
            continue
        delta = pos.size - before.size
        if delta > 0:
            # Their fill price, recovered from the change in average cost.
            fill = (pos.cost - before.cost) / delta
            if not 0 < fill < 1:
                fill = pos.cur_price
            usd = delta * fill
            if usd >= config.min_signal_usd:
                signals.append(make(ADD, pos, fill, usd, their_weight=usd / portfolio))
        elif delta < 0:
            fraction = -delta / before.size
            usd = -delta * pos.cur_price
            if fraction >= config.min_trim_fraction and usd >= config.min_signal_usd:
                signals.append(make(TRIM, pos, pos.cur_price, usd, sell_fraction=fraction))

    for asset, before in old.items():
        if asset in new or before.resolved:
            continue
        usd = before.value
        if usd >= config.min_signal_usd:
            signals.append(make(EXIT, before, before.cur_price, usd, sell_fraction=1.0))
    return signals


def size_buy(signal: Signal, config: CopyConfig) -> Signal:
    """Fill in limit price, shares and stake for an OPEN/ADD, or downgrade it to WAIT."""
    ceiling = min(signal.their_price + config.max_chase, config.max_entry_price)
    if signal.market_price > ceiling:
        reason = (
            f"price is now {signal.market_price * 100:.0f}¢, more than {config.max_chase * 100:.0f}¢ "
            f"above their {signal.their_price * 100:.0f}¢ entry"
            if signal.market_price <= config.max_entry_price
            else f"price is {signal.market_price * 100:.0f}¢, too little upside left"
        )
        signal.action = WAIT
        signal.notes.append(f"don't chase: {reason}. A resting limit at {_tick_down(ceiling) * 100:.0f}¢ is the most to pay.")
        return signal

    limit = _tick_up(min(signal.market_price + config.slippage, ceiling))
    weight = min(signal.their_weight * config.copy_scale, config.max_position_pct)
    stake = config.bankroll_usd * weight
    shares = math.floor(stake / limit * 100) / 100 if limit else 0.0
    if shares * limit < config.min_order_usd:
        signal.action = WAIT
        signal.notes.append(
            f"your copy would be ${stake:,.2f}, under the ${config.min_order_usd:,.2f} minimum. "
            "Raise your bankroll or COPY_SCALE to follow moves this small."
        )
        return signal
    if signal.their_weight * config.copy_scale > config.max_position_pct:
        signal.notes.append(
            f"capped at {config.max_position_pct:.0%} of your bankroll "
            f"(they put {signal.their_weight:.0%} of their portfolio in)"
        )
    signal.limit_price = limit
    signal.shares = shares
    signal.stake = round(shares * limit, 2)
    return signal


def size_sell(signal: Signal, held_shares: float, config: CopyConfig) -> Signal:
    signal.limit_price = max(_tick_down(signal.market_price - config.slippage), 0.01)
    if held_shares > 0:
        signal.shares = math.floor(held_shares * signal.sell_fraction * 100) / 100
        signal.stake = round(signal.shares * signal.limit_price, 2)
    else:
        signal.notes.append("no copied shares on record; if you hold this, sell the same fraction")
    return signal


def consensus_notes(signal: Signal, others: Iterable[Dict[str, Position]]) -> None:
    """Add a note when other followed traders hold the same side or the opposite one."""
    agree = disagree = 0
    for holdings in others:
        for pos in holdings.values():
            if pos.resolved or pos.condition_id != signal.condition_id:
                continue
            if pos.asset == signal.asset:
                agree += 1
            else:
                disagree += 1
    if agree:
        signal.notes.append(f"{agree} other followed trader{'s' if agree > 1 else ''} also hold{'s' if agree == 1 else ''} {signal.outcome}")
    if disagree:
        signal.notes.append(f"⚠ {disagree} other followed trader{'s' if disagree > 1 else ''} hold{'s' if disagree == 1 else ''} the other side")


def _short(wallet: str) -> str:
    return f"{wallet[:6]}…{wallet[-4:]}" if len(wallet) > 12 else wallet


def format_signal(signal: Signal) -> tuple:
    """``(title, body)`` as plain text for every alert channel."""
    who = signal.trader_name or _short(signal.trader)
    their = f"{signal.their_price * 100:.0f}¢"
    now = f"{signal.market_price * 100:.0f}¢"
    if signal.action in BUYS and signal.shares:
        title = f"COPY ▸ BUY {signal.shares:,.2f} {signal.outcome} @ limit {signal.limit_price * 100:.0f}¢ (≈${signal.stake:,.2f})"
        did = f"{'opened' if signal.action == OPEN else 'added'} ${signal.their_usd:,.0f} at ~{their} ({signal.their_weight:.1%} of their portfolio)"
    elif signal.action in SELLS:
        pct = f"{signal.sell_fraction:.0%}"
        amount = f"{signal.shares:,.2f} shares" if signal.shares else f"{pct} of your position"
        title = f"COPY ▸ SELL {amount} of {signal.outcome} @ limit {signal.limit_price * 100:.0f}¢"
        did = "sold out" if signal.action == EXIT else f"sold {pct} of their position"
        did += f" (${signal.their_usd:,.0f}) near {now}"
    else:
        title = f"WAIT ▸ {who} bought {signal.outcome}; no order for you (market {now})"
        did = f"bought ${signal.their_usd:,.0f} at ~{their}"
    lines = [
        signal.title,
        f"Market now {now} · {who} {did}",
        f"Trader: {signal.trader_summary or who} · {profile_url(signal.trader)}",
    ]
    lines += signal.notes
    lines.append(signal.url)
    return title, "\n".join(lines)
