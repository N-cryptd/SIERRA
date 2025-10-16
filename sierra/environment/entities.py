"""Entity definitions for the SIERRA environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

DEFAULT_RESOURCE_TYPES = (
    "food",
    "water",
    "wood",
    "stone",
    "charcoal",
    "cloth",
    "murky_water",
    "sharpening_stone",
    "plank",
    "shelter_frame",
    "threat",
)

INVENTORY_RESOURCE_TYPES = tuple(
    resource for resource in DEFAULT_RESOURCE_TYPES if resource not in {"threat"}
)


@dataclass
class Agent:
    """Represents the controllable agent inside the grid world."""

    x: int
    y: int
    inventory_limit: int
    max_water_filters: int
    hunger: float = 100.0
    thirst: float = 100.0
    has_shelter: bool = False
    has_axe: bool = False
    axe_durability: int = 0
    stamina: float = 100.0
    water_filters_available: int = 0
    inventory: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.inventory:
            self.inventory = {resource: 0 for resource in INVENTORY_RESOURCE_TYPES}

    # ------------------------------------------------------------------
    # Inventory helpers
    # ------------------------------------------------------------------
    def add_item(self, item_type: str, quantity: int) -> int:
        current = self.inventory.get(item_type, 0)
        max_allowed = self.inventory_limit
        added = max(0, min(quantity, max_allowed - current))
        if added:
            self.inventory[item_type] = current + added
        return added

    def remove_item(self, item_type: str, quantity: int) -> int:
        available = self.inventory.get(item_type, 0)
        removed = min(quantity, available)
        if removed:
            self.inventory[item_type] = available - removed
        return removed

    def has_item(self, item_type: str, quantity: int = 1) -> bool:
        return self.inventory.get(item_type, 0) >= quantity

    def get_item_count(self, item_type: str) -> int:
        return self.inventory.get(item_type, 0)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def set_has_shelter(self, value: bool) -> None:
        self.has_shelter = value

    def set_has_axe(self, value: bool) -> None:
        self.has_axe = value
        if value:
            self.axe_durability = 100

    def add_water_filter(self, quantity: int = 1) -> int:
        added = max(
            0,
            min(quantity, self.max_water_filters - self.water_filters_available),
        )
        if added:
            self.water_filters_available += added
        return added

    def use_water_filter(self) -> bool:
        if self.water_filters_available > 0:
            self.water_filters_available -= 1
            return True
        return False

    # ------------------------------------------------------------------
    # Needs handling
    # ------------------------------------------------------------------
    def update_needs(
        self,
        *,
        is_day: bool,
        base_decay: float,
        shelter_multiplier: float,
        night_multiplier: float,
        night_no_shelter_multiplier: float,
    ) -> None:
        hunger_decay = base_decay
        thirst_decay = base_decay

        if self.has_shelter:
            hunger_decay *= shelter_multiplier
            thirst_decay *= shelter_multiplier

        if not is_day:
            hunger_decay *= night_multiplier
            thirst_decay *= night_multiplier
            if not self.has_shelter:
                hunger_decay *= night_no_shelter_multiplier
                thirst_decay *= night_no_shelter_multiplier

        self.hunger = max(0.0, self.hunger - hunger_decay)
        self.thirst = max(0.0, self.thirst - thirst_decay)

    def replenish_thirst(self, amount: float, *, max_value: float) -> None:
        self.thirst = min(max_value, self.thirst + amount)

    def replenish_hunger(self, amount: float, *, max_value: float) -> None:
        self.hunger = min(max_value, self.hunger + amount)

    def is_dead(self) -> bool:
        return self.hunger <= 0.0 or self.thirst <= 0.0


@dataclass
class Resource:
    x: int
    y: int
    type: str
    respawn_timer: int = 0

    def __post_init__(self) -> None:
        if self.type not in DEFAULT_RESOURCE_TYPES:
            raise ValueError(f"Invalid resource type: {self.type}")


@dataclass
class Threat:
    x: int
    y: int

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
