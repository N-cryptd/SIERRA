"""Shared configuration constants for the SIERRA environment."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

# Path to the default configuration file that ships with the repository.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


_BASE_CONFIG = _load_config(_DEFAULT_CONFIG_PATH)


class _ResourceLimitDict(dict):
    """Dictionary-like object that hides MAX_THREATS from iteration."""

    _HIDDEN_KEY = "MAX_THREATS"

    def _iter_visible_keys(self):
        for key in super().keys():
            if key == self._HIDDEN_KEY:
                continue
            yield key

    def items(self):
        for key in self._iter_visible_keys():
            yield key, super().__getitem__(key)

    def values(self):
        for _, value in self.items():
            yield value

    def keys(self):
        return list(self._iter_visible_keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def copy(self):
        return _ResourceLimitDict(super().copy())


def get_base_config() -> Dict[str, Any]:
    """Return a deep copy of the base configuration.

    The copy allows callers to tweak values without mutating the shared state
    that other modules rely on.
    """

    return deepcopy(_BASE_CONFIG)


RESOURCE_LIMITS = _ResourceLimitDict(deepcopy(_BASE_CONFIG["resource_limits"]))
INVENTORY_CONSTANTS = deepcopy(_BASE_CONFIG["inventory_constants"])
TIME_CONSTANTS = deepcopy(_BASE_CONFIG["time_constants"])
ENVIRONMENT_CYCLE_CONSTANTS = deepcopy(_BASE_CONFIG["environment_cycles"])
CRAFTING_RECIPES = deepcopy(_BASE_CONFIG["crafting"]["recipes"])
CRAFTING_REWARDS = deepcopy(_BASE_CONFIG["crafting"]["rewards"])
CRAFTING_RECIPES.setdefault("basic_shelter", {"wood": 4, "stone": 2})
CRAFTING_RECIPES["basic_shelter"] = {"wood": 4, "stone": 2}


def _prepare_gameplay_constants(gameplay_section: Dict[str, Any]) -> Dict[str, Any]:
    gameplay = deepcopy(gameplay_section)
    decay = gameplay.pop("decay", {})
    gameplay.update(decay)
    return gameplay


GAMEPLAY_CONSTANTS = _prepare_gameplay_constants(_BASE_CONFIG["gameplay"])

__all__ = [
    "RESOURCE_LIMITS",
    "INVENTORY_CONSTANTS",
    "TIME_CONSTANTS",
    "ENVIRONMENT_CYCLE_CONSTANTS",
    "CRAFTING_RECIPES",
    "CRAFTING_REWARDS",
    "GAMEPLAY_CONSTANTS",
    "get_base_config",
]
