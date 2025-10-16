"""Entity definitions used by the SIERRA environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .constants import GAMEPLAY_CONSTANTS, INVENTORY_CONSTANTS


BASE_DECAY = GAMEPLAY_CONSTANTS["BASE_DECAY"]
SHELTER_MULTIPLIER = GAMEPLAY_CONSTANTS["SHELTER_MULTIPLIER"]
NIGHT_MULTIPLIER = GAMEPLAY_CONSTANTS["NIGHT_MULTIPLIER"]
NIGHT_NO_SHELTER_EXTRA_MULTIPLIER = GAMEPLAY_CONSTANTS["NIGHT_NO_SHELTER_EXTRA_MULTIPLIER"]
THREAT_DAMAGE = GAMEPLAY_CONSTANTS["THREAT_DAMAGE"]
AXE_DURABILITY = GAMEPLAY_CONSTANTS["AXE_DURABILITY"]


@dataclass
class Agent:
    """Represents the controllable agent within the grid world."""

    x: int
    y: int
    hunger: float = GAMEPLAY_CONSTANTS["INITIAL_HUNGER"]
    thirst: float = GAMEPLAY_CONSTANTS["INITIAL_THIRST"]
    has_shelter: bool = False
    has_axe: bool = False
    axe_durability: int = 0
    stamina: int = 100
    water_filters_available: int = 0
    inventory: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.inventory:
            self.inventory = {material: 0 for material in Resource.MATERIAL_TYPES}

    # ------------------------------------------------------------------
    # Inventory helpers
    # ------------------------------------------------------------------
    def add_item(self, item_type: str, quantity: int = 1) -> int:
        """Add an item to the inventory respecting capacity limits.

        Returns the quantity that was actually added.
        """

        if item_type not in self.inventory:
            self.inventory[item_type] = 0

        remaining_capacity = INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"] - self.inventory[item_type]
        if remaining_capacity <= 0:
            return 0

        quantity_added = min(quantity, remaining_capacity)
        self.inventory[item_type] += quantity_added
        return quantity_added

    def remove_item(self, item_type: str, quantity: int = 1) -> bool:
        if self.inventory.get(item_type, 0) < quantity:
            return False
        self.inventory[item_type] -= quantity
        return True

    def get_item_count(self, item_type: str) -> int:
        return self.inventory.get(item_type, 0)

    def has_item(self, item_type: str, quantity: int = 1) -> bool:
        return self.inventory.get(item_type, 0) >= quantity

    def add_water_filter(self, count: int = 1) -> int:
        remaining_capacity = INVENTORY_CONSTANTS["MAX_WATER_FILTERS"] - self.water_filters_available
        if remaining_capacity <= 0:
            return 0
        amount_added = min(count, remaining_capacity)
        self.water_filters_available += amount_added
        return amount_added

    def use_water_filter(self) -> bool:
        if self.water_filters_available <= 0:
            return False
        self.water_filters_available -= 1
        return True

    def set_has_shelter(self, value: bool) -> None:
        self.has_shelter = value

    def set_has_axe(self, value: bool) -> None:
        self.has_axe = value
        if value:
            self.axe_durability = AXE_DURABILITY

    # ------------------------------------------------------------------
    # Needs management
    # ------------------------------------------------------------------
    def update_needs(self, is_day: bool) -> float:
        """Update hunger and thirst values and return the applied decay."""

        hunger_decay = BASE_DECAY
        thirst_decay = BASE_DECAY

        if self.has_shelter:
            hunger_decay *= SHELTER_MULTIPLIER
            thirst_decay *= SHELTER_MULTIPLIER

        if not is_day:
            hunger_decay *= NIGHT_MULTIPLIER
            thirst_decay *= NIGHT_MULTIPLIER
            if not self.has_shelter:
                hunger_decay *= NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
                thirst_decay *= NIGHT_NO_SHELTER_EXTRA_MULTIPLIER

        self.hunger = max(0.0, self.hunger - hunger_decay)
        self.thirst = max(0.0, self.thirst - thirst_decay)
        return hunger_decay

    def replenish_thirst(self, amount: float) -> None:
        max_thirst = GAMEPLAY_CONSTANTS["MAX_THIRST"]
        self.thirst = min(max_thirst, self.thirst + amount)

    def replenish_hunger(self, amount: float) -> None:
        max_hunger = GAMEPLAY_CONSTANTS["MAX_HUNGER"]
        self.hunger = min(max_hunger, self.hunger + amount)

    def apply_threat_damage(self) -> None:
        self.hunger = max(0.0, self.hunger - THREAT_DAMAGE)
        self.thirst = max(0.0, self.thirst - THREAT_DAMAGE)

    def is_dead(self) -> bool:
        return self.hunger <= 0 or self.thirst <= 0


@dataclass
class Resource:
    x: int
    y: int
    type: str = "food"

    MATERIAL_TYPES = [
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
    ]

    def __post_init__(self) -> None:
        if self.type not in self.MATERIAL_TYPES:
            raise ValueError(f"Invalid resource type: {self.type}")


@dataclass
class Threat:
    x: int
    y: int

    def move_towards(self, target_x: int, target_y: int) -> None:
        if target_x > self.x:
            self.x += 1
        elif target_x < self.x:
            self.x -= 1
        if target_y > self.y:
            self.y += 1
        elif target_y < self.y:
            self.y -= 1
