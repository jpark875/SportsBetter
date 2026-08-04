"""
data_pipeline/odds_client.py

Live and historical odds ingestion layer.

Supports two providers selectable via ``config.settings.ODDS_PROVIDER``:

  * ``"theodds"``  — The Odds API  (https://the-odds-api.com)
  * ``"oddsjam"``  — OddsJam API   (https://oddsjam.com)

Every public method returns a normalised ``pd.DataFrame`` using a canonical
schema so the rest of the pipeline is provider-agnostic.

Canonical bet columns
---------------------
    game_id          : str   — "<home_team>_<away_team>_<commence_time_date>"
    commence_time    : datetime (UTC)
    home_team        : str
    away_team        : str
    sport_key        : str   — e.g. "basketball_nba"
    bet_type         : str   — "moneyline" | "spread" | "total" | "player_prop"
    market_key       : str   — provider-specific market key
    bookmaker        : str
    outcome_name     : str   — team name, "Over"/"Under", or player name
    price            : float — American odds integer form
    point            : float — spread / total / prop line (NaN where not applicable)
    player_name      : str   — populated for player_prop rows only
    prop_stat        : str   — e.g. "points", "assists", "rebounds"
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config.settings import (
    ODDS_PROVIDER,
    THE_ODDS_API_BASE,
    THE_ODDS_API_KEY,
    ODDSJAM_API_BASE,
    ODDSJAM_API_KEY,
)

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: Dict[str, Any]) -> Any:
    """HTTP GET with error propagation."""
    response = _SESSION.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def _american_to_implied(american: float) -> float:
    """Convert American moneyline to implied probability (no-vig approximation)."""
    if american >= 100:
        return 100 / (american + 100)
    return -american / (-american + 100)


def _build_game_id(home: str, away: str, commence: str) -> str:
    date_part = commence[:10].replace("-", "")
    h = home.replace(" ", "").upper()[:6]
    a = away.replace(" ", "").upper()[:6]
    return f"{h}_{a}_{date_part}"


# ---------------------------------------------------------------------------
# The Odds API adapter
# ---------------------------------------------------------------------------

class _TheOddsAdapter:
    """Thin adapter around The Odds API v4."""

    def __init__(self, api_key: str, base_url: str, sport_key: str = "basketball_nba") -> None:
        self._key = api_key
        self._base = base_url
        self._sport_key = sport_key

    # --- Core markets -------------------------------------------------------

    def fetch_game_odds(
        self,
        markets: List[str] = None,
        bookmakers: Optional[List[str]] = None,
        regions: str = "us",
    ) -> pd.DataFrame:
        """
        Fetch moneyline, spreads, and totals for all live NBA games.

        Parameters
        ----------
        markets : list[str]
            Subset of ``["h2h", "spreads", "totals"]``.
            Defaults to all three.
        bookmakers : list[str], optional
            Filter by specific bookmaker slugs, e.g. ``["draftkings", "fanduel"]``.
        regions : str
            ``"us"`` | ``"uk"`` | ``"eu"`` | ``"au"``.

        Returns
        -------
        pd.DataFrame
            Canonical schema described in module docstring.
        """
        markets = markets or ["h2h", "spreads", "totals"]
        params: Dict[str, Any] = {
            "apiKey": self._key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = f"{self._base}/sports/{self._sport_key}/odds"
        logger.info("TheOddsAPI → GET %s", url)
        raw: List[Dict] = _get(url, params)

        rows: List[Dict] = []
        for event in raw:
            game_id = _build_game_id(
                event["home_team"],
                event["away_team"],
                event["commence_time"],
            )
            for bk in event.get("bookmakers", []):
                for market in bk.get("markets", []):
                    mkey = market["key"]
                    bet_type = self._market_to_bet_type(mkey)
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "game_id": game_id,
                            "commence_time": pd.to_datetime(event["commence_time"]),
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "sport_key": self._sport_key,
                            "bet_type": bet_type,
                            "market_key": mkey,
                            "bookmaker": bk["key"],
                            "outcome_name": outcome["name"],
                            "price": float(outcome.get("price", 0)),
                            "point": float(outcome.get("point", float("nan"))),
                            "player_name": "",
                            "prop_stat": "",
                        })
        return pd.DataFrame(rows)

    # --- Player props -------------------------------------------------------

    def fetch_player_props(
        self,
        event_id: str,
        prop_markets: List[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch player prop lines for a specific game event.

        Parameters
        ----------
        event_id : str
            The Odds API event ID (obtainable from the events endpoint).
        prop_markets : list[str]
            e.g. ``["player_points", "player_rebounds", "player_assists"]``.

        Returns
        -------
        pd.DataFrame
            Canonical schema with ``player_name`` and ``prop_stat`` populated.
        """
        prop_markets = prop_markets or [
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_threes",
            "player_blocks",
            "player_steals",
        ]
        params = {
            "apiKey": self._key,
            "markets": ",".join(prop_markets),
            "oddsFormat": "american",
        }
        url = f"{self._base}/sports/{self._sport_key}/events/{event_id}/odds"
        logger.info("TheOddsAPI props → GET %s", url)
        raw = _get(url, params)

        rows: List[Dict] = []
        game_id = _build_game_id(
            raw.get("home_team", ""),
            raw.get("away_team", ""),
            raw.get("commence_time", ""),
        )
        for bk in raw.get("bookmakers", []):
            for market in bk.get("markets", []):
                mkey = market["key"]
                prop_stat = mkey.replace("player_", "")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "game_id": game_id,
                        "commence_time": pd.to_datetime(raw.get("commence_time")),
                        "home_team": raw.get("home_team", ""),
                        "away_team": raw.get("away_team", ""),
                        "sport_key": self._sport_key,
                        "bet_type": "player_prop",
                        "market_key": mkey,
                        "bookmaker": bk["key"],
                        "outcome_name": outcome.get("name", ""),
                        "price": float(outcome.get("price", 0)),
                        "point": float(outcome.get("point", float("nan"))),
                        "player_name": outcome.get("description", ""),
                        "prop_stat": prop_stat,
                    })
        return pd.DataFrame(rows)

    # --- Event list ---------------------------------------------------------

    def fetch_event_ids(self) -> pd.DataFrame:
        """Return a mapping of ``event_id → (home_team, away_team, commence_time)``."""
        params = {"apiKey": self._key, "dateFormat": "iso"}
        url = f"{self._base}/sports/{self._sport_key}/events"
        raw: List[Dict] = _get(url, params)
        return pd.DataFrame([
            {
                "event_id": e["id"],
                "home_team": e["home_team"],
                "away_team": e["away_team"],
                "commence_time": pd.to_datetime(e["commence_time"]),
            }
            for e in raw
        ])

    @staticmethod
    def _market_to_bet_type(market_key: str) -> str:
        mapping = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
        return mapping.get(market_key, market_key)


# ---------------------------------------------------------------------------
# OddsJam adapter
# ---------------------------------------------------------------------------

class _OddsJamAdapter:
    """Thin adapter around the OddsJam v2 API."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._key = api_key
        self._base = base_url

    def fetch_game_odds(self) -> pd.DataFrame:
        """Fetch live NBA game odds from OddsJam."""
        params = {
            "key": self._key,
            "sport": "basketball",
            "league": "NBA",
            "market_name": "Moneyline,Point Spread,Total Points",
        }
        url = f"{self._base}/game-odds"
        logger.info("OddsJam → GET %s", url)
        raw = _get(url, params)

        rows: List[Dict] = []
        for game in raw.get("data", []):
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            ct = game.get("start_time", "")
            game_id = _build_game_id(home, away, ct)

            for odds_entry in game.get("odds", []):
                market_name = odds_entry.get("market_name", "")
                bet_type = self._market_to_bet_type(market_name)
                rows.append({
                    "game_id": game_id,
                    "commence_time": pd.to_datetime(ct),
                    "home_team": home,
                    "away_team": away,
                    "sport_key": "basketball_nba",
                    "bet_type": bet_type,
                    "market_key": market_name,
                    "bookmaker": odds_entry.get("sportsbook", ""),
                    "outcome_name": odds_entry.get("selection", ""),
                    "price": float(odds_entry.get("price", 0)),
                    "point": float(odds_entry.get("handicap", float("nan"))),
                    "player_name": odds_entry.get("player_name", ""),
                    "prop_stat": odds_entry.get("stat", ""),
                })
        return pd.DataFrame(rows)

    def fetch_player_props(self) -> pd.DataFrame:
        """Fetch live NBA player props from OddsJam."""
        params = {
            "key": self._key,
            "sport": "basketball",
            "league": "NBA",
        }
        url = f"{self._base}/player-props"
        raw = _get(url, params)

        rows: List[Dict] = []
        for entry in raw.get("data", []):
            home = entry.get("home_team", "")
            away = entry.get("away_team", "")
            ct = entry.get("start_time", "")
            rows.append({
                "game_id": _build_game_id(home, away, ct),
                "commence_time": pd.to_datetime(ct),
                "home_team": home,
                "away_team": away,
                "sport_key": "basketball_nba",
                "bet_type": "player_prop",
                "market_key": entry.get("stat", ""),
                "bookmaker": entry.get("sportsbook", ""),
                "outcome_name": entry.get("selection", ""),
                "price": float(entry.get("price", 0)),
                "point": float(entry.get("line", float("nan"))),
                "player_name": entry.get("player_name", ""),
                "prop_stat": entry.get("stat", ""),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _market_to_bet_type(market_name: str) -> str:
        m = market_name.lower()
        if "moneyline" in m:
            return "moneyline"
        if "spread" in m:
            return "spread"
        if "total" in m:
            return "total"
        return "other"


# ---------------------------------------------------------------------------
# Unified public facade
# ---------------------------------------------------------------------------

class OddsClient:
    """
    Provider-agnostic odds client.

    Selects the active adapter from ``config.settings.ODDS_PROVIDER``
    at construction time.  All callers interact only with this class.

    Parameters
    ----------
    provider : str, optional
        Override ``ODDS_PROVIDER`` setting.  ``"theodds"`` or ``"oddsjam"``.
    sport_key : str, optional
        Override the sport key sent to the provider, e.g.
        ``"basketball_nba"`` or ``"soccer_fifa_world_cup"``.
    """

    def __init__(self, provider: Optional[str] = None, sport_key: Optional[str] = None) -> None:
        from config.settings import NBA_SPORT_KEY
        prov = (provider or ODDS_PROVIDER).lower()
        sk = sport_key or NBA_SPORT_KEY
        if prov == "theodds":
            self._adapter = _TheOddsAdapter(THE_ODDS_API_KEY, THE_ODDS_API_BASE, sport_key=sk)
        elif prov == "oddsjam":
            self._adapter = _OddsJamAdapter(ODDSJAM_API_KEY, ODDSJAM_API_BASE)
        else:
            raise ValueError(f"Unknown provider '{prov}'. Use 'theodds' or 'oddsjam'.")
        self._provider = prov
        self._sport_key = sk
        logger.info("OddsClient initialised with provider='%s' sport='%s'", prov, sk)

    def get_game_odds(self, **kwargs) -> pd.DataFrame:
        """
        Fetch moneyline, spread, and total lines for tonight's slate.

        Returns
        -------
        pd.DataFrame
            Canonical odds schema (see module docstring).
        """
        return self._adapter.fetch_game_odds(**kwargs)

    def get_player_props(self, **kwargs) -> pd.DataFrame:
        """
        Fetch player prop lines.

        For The Odds API, pass ``event_id=<str>`` in kwargs.

        Returns
        -------
        pd.DataFrame
            Canonical odds schema with ``player_name`` and ``prop_stat`` populated.
        """
        return self._adapter.fetch_player_props(**kwargs)

    def get_best_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce a multi-book odds DataFrame to the single best-priced line
        per ``(game_id, bet_type, outcome_name, point)``.

        This represents the true market edge available to a sharp bettor.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`get_game_odds` or :meth:`get_player_props`.

        Returns
        -------
        pd.DataFrame
            One row per unique wager opportunity at the best available price.
        """
        if df.empty:
            return df
        key_cols = ["game_id", "bet_type", "outcome_name", "point"]
        best = (
            df.sort_values("price", ascending=False)
            .groupby(key_cols, dropna=False)
            .first()
            .reset_index()
        )
        return best

    def get_implied_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append ``implied_prob`` column (raw, no vig removed) to an odds DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Any canonical odds DataFrame with a ``price`` column.

        Returns
        -------
        pd.DataFrame
            Input with ``implied_prob`` column appended.
        """
        if df.empty or "price" not in df.columns:
            return df
        df = df.copy()
        df["implied_prob"] = df["price"].apply(_american_to_implied)
        return df
