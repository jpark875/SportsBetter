"""
config/settings.py
Central configuration management. All secrets and tunables live here,
loaded from environment variables via python-dotenv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# API Keys & Endpoints
# ---------------------------------------------------------------------------

THE_ODDS_API_KEY: str = os.getenv("THE_ODDS_API_KEY", "")
THE_ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"

ODDSJAM_API_KEY: str = os.getenv("ODDSJAM_API_KEY", "")
ODDSJAM_API_BASE: str = "https://api.oddsjam.com/api/v2"

# Choose "theodds" or "oddsjam"
ODDS_PROVIDER: str = os.getenv("ODDS_PROVIDER", "theodds")

# ---------------------------------------------------------------------------
# Sport keys  (passed to odds providers)
# ---------------------------------------------------------------------------

NBA_SPORT_KEY: str = "basketball_nba"
WC_SPORT_KEY: str = os.getenv("WC_SPORT_KEY", "soccer_fifa_world_cup")

# ---------------------------------------------------------------------------
# World Cup model artefacts
# ---------------------------------------------------------------------------

WC_MODEL_PATH: str = os.path.join(os.getenv("MODEL_DIR", "artefacts"), "wc_lgbm.pkl")
WC_CALIBRATOR_PATH: str = os.path.join(os.getenv("MODEL_DIR", "artefacts"), "wc_calibrator.pkl")

# ---------------------------------------------------------------------------
# International football data source (martj42/international_results, no key needed)
# ---------------------------------------------------------------------------

FOOTBALL_DATA_URL: str = os.getenv(
    "FOOTBALL_DATA_URL",
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
)

GOALSCORERS_DATA_URL: str = os.getenv(
    "GOALSCORERS_DATA_URL",
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
)

WC_FEATURE_COLS: list = [
    # ELO ratings (pre-match)
    "HOME_ELO",
    "AWAY_ELO",
    "DELTA_ELO",
    # Rolling form — last 5 competitive matches
    "HOME_FORM_LAST5",
    "AWAY_FORM_LAST5",
    "HOME_GOALS_FOR_LAST5",
    "AWAY_GOALS_FOR_LAST5",
    "HOME_GOALS_AGAINST_LAST5",
    "AWAY_GOALS_AGAINST_LAST5",
    # EWM attack/defense ratings (span=7)
    "HOME_ATK_RATING",
    "AWAY_ATK_RATING",
    "HOME_DEF_RATING",
    "AWAY_DEF_RATING",
    # Attack vs opponent's defense matchup scores
    "HOME_ATK_VS_AWAY_DEF",
    "AWAY_ATK_VS_HOME_DEF",
    # Player star-concentration (top scorer share over last 20 matches)
    "HOME_STAR_CONC",
    "AWAY_STAR_CONC",
    # Venue
    "IS_NEUTRAL",
]

# ---------------------------------------------------------------------------
# NBA API throttle (requests per minute — nba_api is unofficial, be gentle)
# ---------------------------------------------------------------------------

NBA_API_CALLS_PER_MINUTE: int = int(os.getenv("NBA_API_CALLS_PER_MINUTE", "20"))
NBA_API_TIMEOUT: int = int(os.getenv("NBA_API_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Model artefacts
# ---------------------------------------------------------------------------

MODEL_DIR: str = os.getenv("MODEL_DIR", "artefacts")
LGBM_MODEL_PATH: str = os.path.join(MODEL_DIR, "lgbm_win_prob.pkl")
CALIBRATOR_PATH: str = os.path.join(MODEL_DIR, "isotonic_calibrator.pkl")

# ---------------------------------------------------------------------------
# Portfolio optimiser
# ---------------------------------------------------------------------------

MAX_PORTFOLIO_KELLY_FRACTION: float = float(
    os.getenv("MAX_PORTFOLIO_KELLY_FRACTION", "0.25")
)
MAX_SINGLE_BET_FRACTION: float = float(os.getenv("MAX_SINGLE_BET_FRACTION", "0.05"))
CORRELATION_SHRINKAGE_ALPHA: float = float(
    os.getenv("CORRELATION_SHRINKAGE_ALPHA", "0.1")
)  # Ledoit-Wolf coefficient

# ---------------------------------------------------------------------------
# Risk bridge defaults (overridden at runtime by user input)
# ---------------------------------------------------------------------------

@dataclass
class UserRiskProfile:
    """User-supplied financial parameters passed to the risk bridge."""

    liquid_bankroll: float = 1_000.0
    disposable_income: float = 200.0
    volatility_tolerance: int = 5          # 1 (conservative) – 10 (aggressive)
    max_drawdown_pct: float = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.volatility_tolerance <= 10:
            raise ValueError("volatility_tolerance must be between 1 and 10.")
        # Linear mapping: tolerance 1 → 5 % drawdown, 10 → 30 % drawdown
        self.max_drawdown_pct = 0.05 + (self.volatility_tolerance - 1) * (0.25 / 9)

    @property
    def effective_bankroll(self) -> float:
        """Capital that may be wagered: smaller of bankroll or disposable income."""
        return min(self.liquid_bankroll, self.disposable_income)
