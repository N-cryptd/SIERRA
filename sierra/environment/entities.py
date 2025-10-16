"""Simple entity definitions used by the environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .constants import GAMEPLAY_CONSTANTS, INVENTORY_CONSTANTS, RESOURCE_TYPES

_INITIAL_HUNGER = GAMEPLAY_CONSTANTS.get("INITIAL_HUNGER", 100.0)
_INITIAL_THIRST = GAMEPLAY_CONSTANTS.get("INITIAL_THIRST", 100.0)
_MAX_HUNGER = GAMEPLAY_CONSTANTS.get("MAX_HUNGER", 100.0)
_MAX_THIRST = GAMEPLAY_CONSTANTS.get("MAX_THIRST", 100.0)
_AXE_DURABILITY = GAMEPLAY_CONSTANTS.get("AXE_DURABILITY", 100)

_DECAY = GAMEPLAY_CONSTANTS.get("decay", {})
_BASE_DECAY = _DECAY.get("BASE_DECAY", 0.1)
_SHELTER_MULTIPLIER = _DECAY.get("SHELTER_MULTIPLIER", 0.75)
_NIGHT_MULTIPLIER = _DECAY.get("NIGHT_MULTIPLIER", 1.2)
_NIGHT_NO_SHELTER_EXTRA_MULTIPLIER = _DECAY.get("NIGHT_NO_SHELTER_EXTRA_MULTIPLIER", 1.5)


@dataclass
class Agent:
    x: int
    y: int
    hunger: float = _INITIAL_HUNGER
    thirst: float = _INITIAL_THIRST
    inventory: Dict[str, int] = field(default_factory=lambda: {name: 0 for name in RESOURCE_TYPES})
    has_shelter: bool = False
    has_axe: bool = False
    axe_durability: int = 0
    stamina: int = 100
    water_filters_available: int = 0

    def add_item(self, item_type: str, quantity: int = 1) -> int:
        if item_type not in self.inventory:
            self.inventory[item_type] = 0
        current = self.inventory[item_type]
        limit = INVENTORY_CONSTANTS.get("MAX_INVENTORY_PER_ITEM", 10)
        space = max(0, limit - current)
        added = min(quantity, space)
        if added:
            self.inventory[item_type] += added
        return added

    def remove_item(self, item_type: str, quantity: int = 1) -> bool:
        if self.inventory.get(item_type, 0) < quantity:
            return False
        self.inventory[item_type] -= quantity
        return True

    def has_item(self, item_type: str, quantity: int = 1) -> bool:
        return self.inventory.get(item_type, 0) >= quantity

    def get_item_count(self, item_type: str) -> int:
        return self.inventory.get(item_type, 0)

    def update_needs(self, is_day: bool, has_shelter_effect: bool) -> None:
        hunger_decay = _BASE_DECAY
        thirst_decay = _BASE_DECAY
        if has_shelter_effect:
            hunger_decay *= _SHELTER_MULTIPLIER
            thirst_decay *= _SHELTER_MULTIPLIER
        if not is_day:
            hunger_decay *= _NIGHT_MULTIPLIER
            thirst_decay *= _NIGHT_MULTIPLIER
            if not has_shelter_effect:
                hunger_decay *= _NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
                thirst_decay *= _NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
        self.hunger = max(0.0, self.hunger - hunger_decay)
        self.thirst = max(0.0, self.thirst - thirst_decay)

    def replenish_thirst(self, amount: float) -> None:
        self.thirst = min(_MAX_THIRST, self.thirst + amount)

    def replenish_hunger(self, amount: float) -> None:
        self.hunger = min(_MAX_HUNGER, self.hunger + amount)

    def set_has_shelter(self, value: bool) -> None:
        self.has_shelter = value

    def set_has_axe(self, value: bool) -> None:
        self.has_axe = value
        if value:
            self.axe_durability = _AXE_DURABILITY

    def add_water_filter(self, count: int = 1) -> bool:
        limit = INVENTORY_CONSTANTS.get("MAX_WATER_FILTERS", 5)
        space = max(0, limit - self.water_filters_available)
        added = min(space, count)
        if added:
            self.water_filters_available += added
            return True
        return False

    def use_water_filter(self) -> bool:
        if self.water_filters_available > 0:
            self.water_filters_available -= 1
            return True
        return False

    def is_dead(self) -> bool:
        return self.hunger <= 0 or self.thirst <= 0


@dataclass
class Resource:
    x: int
    y: int
    type: str
    respawn_timer: int = 0


@dataclass
class Threat:
    x: int
    y: int
    state: str = "PATROLLING"
    target: tuple[int, int] | None = None
    type: str = "threat"


__all__ = ["Agent", "Resource", "Threat"]
