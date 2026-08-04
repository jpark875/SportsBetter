"""
sports/base.py

The sport-plugin contract.  Adding a new sport to the entire app
(training script, web app, arbitrage scanner, parlay builder) means
writing one subclass of :class:`SportPlugin` that can answer a single
question — "give me every historical game as date/home/away/scores" —
plus a :class:`SportConfig` of tuning constants, and registering it in
``sports/__init__.py``.  Everything downstream is shared code.

Model artefacts per sport land in ``MODEL_DIR`` as:
    <key>_model.pkl   — fitted ModelTrainer (binary or 3-class)
    <key>_state.pkl   — dict: current ELO, latest team stats, calibrator,
                        metadata (trained_at, feature_cols, has_draws)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from config.settings import MODEL_DIR


@dataclass(frozen=True)
class SportConfig:
    """Per-sport constants consumed by the generic feature engine and UI."""

    key: str                 # registry key, e.g. "mlb"
    display_name: str        # e.g. "MLB Baseball"
    odds_sport_key: str      # TheOddsAPI sport key, e.g. "baseball_mlb"
    has_draws: bool          # True → 3-way market (soccer), False → 2-way
    elo_k: float             # ELO K-factor (lower for long seasons)
    hfa_elo: float           # home-advantage ELO bonus inside expectation
    default_score: float     # typical per-game score, prior for unseen teams
    score_label: str         # "runs" | "goals" | "points" — UI copy only
    form_window: int = 5
    ewm_span: int = 7


class SportPlugin(ABC):
    """Base class every sport module implements."""

    config: SportConfig

    @abstractmethod
    def fetch_history(self) -> pd.DataFrame:
        """
        Return all available historical games with columns:
            date (datetime), home_team, away_team,
            home_score (float), away_score (float)

        Team names must match what TheOddsAPI uses for this sport —
        override :meth:`normalize_name` when the sources disagree.
        """

    def normalize_name(self, name: str) -> str:
        """Map an odds-provider team name to this plugin's canonical name."""
        return name.strip()

    # ------------------------------------------------------------------
    # Artefact locations (shared convention)
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> str:
        return os.path.join(MODEL_DIR, f"{self.config.key}_model.pkl")

    @property
    def state_path(self) -> str:
        return os.path.join(MODEL_DIR, f"{self.config.key}_state.pkl")

    def is_trained(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.state_path)
