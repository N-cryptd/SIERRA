"""Configuration helpers and constants for the SIERRA environment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from yaml import safe_load

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return safe_load(handle.read())


CONFIG = _load_config()

TIME_CONSTANTS: Dict[str, int] = CONFIG.get("time_constants", {})
RESOURCE_LIMITS: Dict[str, int] = CONFIG.get("resource_limits", {})
INVENTORY_CONSTANTS: Dict[str, int] = CONFIG.get("inventory_constants", {})
CRAFTING: Dict[str, dict] = CONFIG.get("crafting", {})
CRAFTING_RECIPES: Dict[str, dict] = CRAFTING.get("recipes", {})
CRAFTING_REWARDS: Dict[str, float] = CRAFTING.get("rewards", {})
GAMEPLAY_CONSTANTS: Dict[str, float] = CONFIG.get("gameplay", {})
ENVIRONMENT_CYCLE_CONSTANTS: Dict[str, int] = CONFIG.get("environment_cycles", {})
AGENT_CONFIG: Dict[str, int] = CONFIG.get("agent", {})

# Convenience tuples
WEATHER_TYPES: Tuple[str, ...] = tuple(ENVIRONMENT_CYCLE_CONSTANTS.get("WEATHER_TYPES", []))
SEASON_TYPES: Tuple[str, ...] = tuple(ENVIRONMENT_CYCLE_CONSTANTS.get("SEASON_TYPES", []))


def limit_key_to_resource(limit_key: str) -> str:
    name = limit_key.lower()
    if name.startswith("max_"):
        name = name[4:]
    if name.endswith("_sources"):
        name = name[:-8]
    if name.endswith("s"):
        name = name[:-1]
    return name


RESOURCE_TYPES = [
    limit_key_to_resource(key)
    for key in RESOURCE_LIMITS
    if key != "MAX_THREATS"
]

__all__ = [
    "CONFIG",
    "TIME_CONSTANTS",
    "RESOURCE_LIMITS",
    "INVENTORY_CONSTANTS",
    "CRAFTING_RECIPES",
    "CRAFTING_REWARDS",
    "GAMEPLAY_CONSTANTS",
    "ENVIRONMENT_CYCLE_CONSTANTS",
    "WEATHER_TYPES",
    "SEASON_TYPES",
    "AGENT_CONFIG",
    "limit_key_to_resource",
    "RESOURCE_TYPES",
]
