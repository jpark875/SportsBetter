"""Public-API clients, one per platform."""

from whale_tracker.sources.kalshi import KalshiClient
from whale_tracker.sources.polymarket import PolymarketClient

__all__ = ["KalshiClient", "PolymarketClient"]
