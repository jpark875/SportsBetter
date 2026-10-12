"""sports/ — the sport-plugin registry."""

from __future__ import annotations

import importlib
from typing import Dict, List

from sports.base import SportPlugin

_PLUGIN_MODULES: List[str] = ["mlb", "epl", "nba", "wnba"]

_cache: Dict[str, SportPlugin] = {}


def available_sports() -> List[str]:
    """Registry keys in display order."""
    return list(_PLUGIN_MODULES)


def get_plugin(key: str) -> SportPlugin:
    """Return the (cached) plugin instance for a registry key."""
    if key not in _PLUGIN_MODULES:
        raise KeyError(
            f"Unknown sport '{key}'. Available: {', '.join(_PLUGIN_MODULES)}"
        )
    if key not in _cache:
        module = importlib.import_module(f"sports.{key}")
        _cache[key] = module.PLUGIN
    return _cache[key]


def all_plugins() -> List[SportPlugin]:
    return [get_plugin(k) for k in _PLUGIN_MODULES]
