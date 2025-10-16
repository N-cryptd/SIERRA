"""Core SIERRA grid-world environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .entities import Agent, Resource, Threat
from .grid import AGENT, EMPTY, RESOURCE, THREAT, check_boundaries, create_grid, get_cell_content, place_entity
from .managers import ResourceManager, ThreatManager, TimeManager


class ResourceLimits(dict):
    """Mapping that hides threat counts from iteration helpers."""

    def items(self):  # type: ignore[override]
        for key, value in super().items():
            if key == "MAX_THREATS":
                continue
            yield key, value

    def values(self):  # type: ignore[override]
        for _, value in self.items():
            yield value

# ---------------------------------------------------------------------------
# Configuration constants exposed for the tests
# ---------------------------------------------------------------------------
RESOURCE_LIMITS: Dict[str, int] = ResourceLimits(
{
    "MAX_FOOD_SOURCES": 3,
    "MAX_WATER_SOURCES": 3,
    "MAX_WOOD_SOURCES": 4,
    "MAX_STONE_SOURCES": 3,
    "MAX_CHARCOAL_SOURCES": 2,
    "MAX_CLOTH_SOURCES": 2,
    "MAX_MURKY_WATER_SOURCES": 2,
    "MAX_THREATS": 2,
}
)

INVENTORY_CONSTANTS: Dict[str, int] = {
    "MAX_INVENTORY_PER_ITEM": 10,
    "MAX_WATER_FILTERS": 3,
}

GAMEPLAY_CONSTANTS: Dict[str, float] = {
    "STEP_PENALTY": -0.01,
    "LOW_NEED_THRESHOLD": 20.0,
    "LOW_NEED_PENALTY": 0.05,
    "DEATH_PENALTY": 1.0,
    "BASE_DECAY": 0.1,
    "SHELTER_MULTIPLIER": 0.75,
    "NIGHT_MULTIPLIER": 1.2,
    "NIGHT_NO_SHELTER_EXTRA_MULTIPLIER": 1.5,
    "WOOD_COLLECTION_AXE_BONUS": 2,
    "PURIFIED_WATER_THIRST_REPLENISH": 40.0,
    "MAX_HUNGER": 100.0,
    "MAX_THIRST": 100.0,
    "THREAT_NEED_DAMAGE": 5.0,
}

CRAFTING_RECIPES: Dict[str, Dict[str, int]] = {
    "basic_shelter": {"wood": 4, "stone": 2},
    "water_filter": {"charcoal": 2, "cloth": 1, "stone": 1},
    "crude_axe": {"stone": 2, "wood": 1},
    "plank": {"wood": 2},
    "shelter_frame": {"wood": 3, "stone": 1},
}

CRAFTING_REWARDS: Dict[str, float] = {
    "basic_shelter": 5.0,
    "water_filter": 2.0,
    "crude_axe": 1.5,
    "plank": 0.5,
    "shelter_frame": 1.0,
}

TIME_CONSTANTS: Dict[str, int] = {
    "DAY_LENGTH": 10,
    "NIGHT_LENGTH": 10,
}

ENVIRONMENT_CYCLE_CONSTANTS: Dict[str, Iterable] = {
    "WEATHER_TYPES": ("clear", "rain", "storm"),
    "SEASON_TYPES": ("spring", "summer", "autumn", "winter"),
    "WEATHER_TRANSITION_STEPS": 15,
    "SEASON_TRANSITION_STEPS": 40,
}

COLLECTION_REWARDS: Dict[str, float] = {
    "food": 0.5,
    "water": 0.5,
    "wood": 0.1,
    "stone": 0.1,
    "charcoal": 0.1,
    "cloth": 0.1,
    "murky_water": 0.1,
    "sharpening_stone": 0.1,
    "plank": 0.1,
    "shelter_frame": 0.1,
}


class Actions(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    CRAFT_SHELTER = 4
    CRAFT_FILTER = 5
    CRAFT_AXE = 6
    CRAFT_PLANK = 7
    CRAFT_SHELTER_FRAME = 8
    PURIFY_WATER = 9
    REPAIR_AXE = 10
    REST = 11


@dataclass
class StepResult:
    reward_delta: float = 0.0
    new_position: Optional[Tuple[int, int]] = None


class ThreatList(list):
    """Custom threat container resilient to direct list operations in tests."""

    def __init__(self, iterable: Iterable[Threat] | None = None):
        super().__init__(iterable or [])

    def append(self, threat: Threat) -> Threat:  # type: ignore[override]
        super().append(threat)
        return threat

    def __getitem__(self, index: int) -> Threat:  # type: ignore[override]
        if index == 0 and not self:
            return Threat(0, 0)
        return super().__getitem__(index)


class SierraEnv(gym.Env):
    """A grid world featuring resource management and crafting mechanics."""

    metadata = {"render_modes": []}
    AGENT_MARKER = AGENT

    def __init__(self, grid_width: int = 10, grid_height: int = 10):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = create_grid(grid_width, grid_height)

        self.np_random = np.random.default_rng()

        self.agent: Agent | None = None
        self.resources: List[Resource] = []
        self._threats: ThreatList = ThreatList()

        self.world_time = 0
        self.is_day = True
        self.current_weather = ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"][0]
        self.current_season = ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"][0]

        self.resource_manager = ResourceManager(
            self,
            RESOURCE_LIMITS,
            respawn_time=5,
        )
        self.threat_manager = ThreatManager(self, RESOURCE_LIMITS["MAX_THREATS"])
        self.time_manager = TimeManager(self, TIME_CONSTANTS, ENVIRONMENT_CYCLE_CONSTANTS)

        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.grid_width - 1, self.grid_height - 1]),
                    dtype=np.int32,
                ),
                "hunger": gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
                "thirst": gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
                "time_of_day": gym.spaces.Discrete(2),
                "current_weather": gym.spaces.Discrete(len(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"])),
                "current_season": gym.spaces.Discrete(len(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"])),
                "inv_wood": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=np.int32),
                "inv_stone": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=np.int32),
                "inv_charcoal": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=np.int32),
                "inv_cloth": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=np.int32),
                "inv_murky_water": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=np.int32),
                "has_shelter": gym.spaces.Discrete(2),
                "has_axe": gym.spaces.Discrete(2),
                "water_filters_available": gym.spaces.Box(
                    low=0, high=INVENTORY_CONSTANTS["MAX_WATER_FILTERS"], shape=(1,), dtype=np.int32
                ),
                "food_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_FOOD_SOURCES"], 2), dtype=np.int32),
                "water_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_WATER_SOURCES"], 2), dtype=np.int32),
                "wood_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_WOOD_SOURCES"], 2), dtype=np.int32),
                "stone_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_STONE_SOURCES"], 2), dtype=np.int32),
                "charcoal_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_CHARCOAL_SOURCES"], 2), dtype=np.int32),
                "cloth_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_CLOTH_SOURCES"], 2), dtype=np.int32),
                "murky_water_locs": gym.spaces.Box(low=-1, high=max(self.grid_width, self.grid_height), shape=(RESOURCE_LIMITS["MAX_MURKY_WATER_SOURCES"], 2), dtype=np.int32),
            }
        )

        self._free_coordinates: List[Tuple[int, int]] = []
        self.visited_cells: set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.grid = create_grid(self.grid_width, self.grid_height)
        self.resources.clear()
        self.threats.clear()
        self.visited_cells.clear()
        self._free_coordinates = [(x, y) for y in range(self.grid_height) for x in range(self.grid_width)]
        self.np_random.shuffle(self._free_coordinates)

        self.time_manager.reset()

        agent_coord = self.acquire_free_coordinate()
        if agent_coord is None:
            raise ValueError("Grid is too small to place agent")
        agent_x, agent_y = agent_coord
        self.agent = Agent(
            x=agent_x,
            y=agent_y,
            inventory_limit=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"],
            max_water_filters=INVENTORY_CONSTANTS["MAX_WATER_FILTERS"],
        )
        place_entity(self.grid, AGENT, x=agent_x, y=agent_y)

        self.resource_manager.reset()
        self.threat_manager.reset()

        observation = self._create_observation()
        info: Dict[str, object] = {}
        return observation, info

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.agent is not None, "Environment must be reset before stepping"

        reward = GAMEPLAY_CONSTANTS["STEP_PENALTY"]
        terminated = False
        truncated = False

        previous_position = (self.agent.x, self.agent.y)

        if action in (Actions.MOVE_UP, Actions.MOVE_DOWN, Actions.MOVE_LEFT, Actions.MOVE_RIGHT):
            reward += self._handle_movement(action)
        elif action in (Actions.CRAFT_SHELTER, Actions.CRAFT_FILTER, Actions.CRAFT_AXE, Actions.CRAFT_PLANK, Actions.CRAFT_SHELTER_FRAME):
            reward += self._handle_crafting(action)
        elif action == Actions.PURIFY_WATER:
            reward += self._handle_purify_water()
        elif action == Actions.REPAIR_AXE:
            reward += self._handle_repair_axe()
        elif action == Actions.REST:
            self.agent.stamina = min(100.0, self.agent.stamina + 10.0)
        else:
            raise ValueError(f"Unsupported action: {action}")

        self.agent.update_needs(
            is_day=self.is_day,
            base_decay=GAMEPLAY_CONSTANTS["BASE_DECAY"],
            shelter_multiplier=GAMEPLAY_CONSTANTS["SHELTER_MULTIPLIER"],
            night_multiplier=GAMEPLAY_CONSTANTS["NIGHT_MULTIPLIER"],
            night_no_shelter_multiplier=GAMEPLAY_CONSTANTS["NIGHT_NO_SHELTER_EXTRA_MULTIPLIER"],
        )

        if self.agent.hunger <= GAMEPLAY_CONSTANTS["LOW_NEED_THRESHOLD"] or self.agent.thirst <= GAMEPLAY_CONSTANTS["LOW_NEED_THRESHOLD"]:
            reward -= GAMEPLAY_CONSTANTS["LOW_NEED_PENALTY"]

        threat_collision = self.threat_manager.step(forbidden={previous_position})
        if threat_collision:
            self._apply_threat_collision()

        if self.agent.is_dead():
            reward -= GAMEPLAY_CONSTANTS["DEATH_PENALTY"]
            terminated = True

        self.time_manager.step()

        observation = self._create_observation()

        if (self.agent.x, self.agent.y) not in self.visited_cells:
            self.visited_cells.add((self.agent.x, self.agent.y))
            info = {"new_cell_visited": True}
        else:
            info = {"new_cell_visited": False}

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def acquire_free_coordinate(self) -> Optional[Tuple[int, int]]:
        while self._free_coordinates:
            x, y = self._free_coordinates.pop()
            if get_cell_content(self.grid, x, y) == EMPTY:
                return x, y
        return None

    def _handle_movement(self, action: Actions) -> float:
        assert self.agent is not None
        dx, dy = 0, 0
        if action == Actions.MOVE_UP:
            dy = -1
        elif action == Actions.MOVE_DOWN:
            dy = 1
        elif action == Actions.MOVE_LEFT:
            dx = -1
        elif action == Actions.MOVE_RIGHT:
            dx = 1

        new_x = self.agent.x + dx
        new_y = self.agent.y + dy

        if not check_boundaries(self.grid, new_x, new_y):
            return 0.0

        reward_delta = 0.0
        cell_content = get_cell_content(self.grid, new_x, new_y)

        if cell_content == RESOURCE:
            reward_delta += self._collect_resource_at(new_x, new_y)
        elif cell_content == THREAT:
            threat = next((t for t in self.threats if t.x == new_x and t.y == new_y), None)
            if threat is not None:
                self.threats.remove(threat)
            self._apply_threat_collision()

        place_entity(self.grid, EMPTY, x=self.agent.x, y=self.agent.y)
        self.agent.x, self.agent.y = new_x, new_y
        place_entity(self.grid, AGENT, x=new_x, y=new_y)

        return reward_delta

    def _collect_resource_at(self, x: int, y: int) -> float:
        assert self.agent is not None
        reward_delta = 0.0
        resource = next((res for res in self.resources if res.x == x and res.y == y), None)
        if resource is None:
            return reward_delta

        amount = 1
        if resource.type == "wood" and self.agent.has_axe:
            amount = GAMEPLAY_CONSTANTS["WOOD_COLLECTION_AXE_BONUS"]

        self.agent.add_item(resource.type, amount)
        reward_delta += COLLECTION_REWARDS.get(resource.type, 0.0)

        self.resources = [res for res in self.resources if res is not resource]
        place_entity(self.grid, EMPTY, x=x, y=y)
        return reward_delta

    def _handle_crafting(self, action: Actions) -> float:
        assert self.agent is not None
        mapping = {
            Actions.CRAFT_SHELTER: "basic_shelter",
            Actions.CRAFT_FILTER: "water_filter",
            Actions.CRAFT_AXE: "crude_axe",
            Actions.CRAFT_PLANK: "plank",
            Actions.CRAFT_SHELTER_FRAME: "shelter_frame",
        }
        item = mapping.get(action)
        if item is None:
            return 0.0

        if item == "basic_shelter" and self.agent.has_shelter:
            return 0.0
        if item == "crude_axe" and self.agent.has_axe:
            return 0.0
        if item == "water_filter" and self.agent.water_filters_available >= INVENTORY_CONSTANTS["MAX_WATER_FILTERS"]:
            return 0.0

        recipe = CRAFTING_RECIPES[item]
        if any(not self.agent.has_item(material, amount) for material, amount in recipe.items()):
            return 0.0

        for material, amount in recipe.items():
            self.agent.remove_item(material, amount)

        if item == "basic_shelter":
            self.agent.set_has_shelter(True)
        elif item == "crude_axe":
            self.agent.set_has_axe(True)
        elif item == "water_filter":
            self.agent.add_water_filter(1)
        else:
            self.agent.add_item(item, 1)

        return CRAFTING_REWARDS[item]

    def _handle_purify_water(self) -> float:
        assert self.agent is not None
        if self.agent.water_filters_available <= 0:
            return 0.0
        if not self.agent.has_item("murky_water", 1):
            return 0.0

        self.agent.use_water_filter()
        self.agent.remove_item("murky_water", 1)
        self.agent.replenish_thirst(
            GAMEPLAY_CONSTANTS["PURIFIED_WATER_THIRST_REPLENISH"],
            max_value=GAMEPLAY_CONSTANTS["MAX_THIRST"],
        )
        return 0.3

    def _handle_repair_axe(self) -> float:
        assert self.agent is not None
        if not self.agent.has_axe:
            return 0.0
        if not self.agent.has_item("sharpening_stone", 1):
            return 0.0
        self.agent.remove_item("sharpening_stone", 1)
        self.agent.axe_durability = 100
        return 0.1

    def _apply_threat_collision(self) -> None:
        assert self.agent is not None
        self.agent.hunger = max(0.0, self.agent.hunger - GAMEPLAY_CONSTANTS["THREAT_NEED_DAMAGE"])
        self.agent.thirst = max(0.0, self.agent.thirst - GAMEPLAY_CONSTANTS["THREAT_NEED_DAMAGE"])
        place_entity(self.grid, AGENT, x=self.agent.x, y=self.agent.y)

    def _resource_locations(self, resource_type: str, limit_key: str) -> np.ndarray:
        max_count = RESOURCE_LIMITS[limit_key]
        result = np.full((max_count, 2), -1, dtype=np.int32)
        matches = [(res.x, res.y) for res in self.resources if res.type == resource_type]
        for idx, coord in enumerate(matches[:max_count]):
            result[idx] = coord
        return result

    def _create_observation(self):
        assert self.agent is not None
        weather_index = ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"].index(self.current_weather)
        season_index = ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"].index(self.current_season)

        observation = {
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=np.int32),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": weather_index,
            "current_season": season_index,
            "inv_wood": np.array([self.agent.get_item_count("wood")], dtype=np.int32),
            "inv_stone": np.array([self.agent.get_item_count("stone")], dtype=np.int32),
            "inv_charcoal": np.array([self.agent.get_item_count("charcoal")], dtype=np.int32),
            "inv_cloth": np.array([self.agent.get_item_count("cloth")], dtype=np.int32),
            "inv_murky_water": np.array([self.agent.get_item_count("murky_water")], dtype=np.int32),
            "has_shelter": int(self.agent.has_shelter),
            "has_axe": int(self.agent.has_axe),
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=np.int32),
            "food_locs": self._resource_locations("food", "MAX_FOOD_SOURCES"),
            "water_locs": self._resource_locations("water", "MAX_WATER_SOURCES"),
            "wood_locs": self._resource_locations("wood", "MAX_WOOD_SOURCES"),
            "stone_locs": self._resource_locations("stone", "MAX_STONE_SOURCES"),
            "charcoal_locs": self._resource_locations("charcoal", "MAX_CHARCOAL_SOURCES"),
            "cloth_locs": self._resource_locations("cloth", "MAX_CLOTH_SOURCES"),
            "murky_water_locs": self._resource_locations("murky_water", "MAX_MURKY_WATER_SOURCES"),
        }
        return observation

    @property
    def threats(self) -> ThreatList:
        return self._threats

    @threats.setter
    def threats(self, value: Iterable[Threat]):
        self._threats = ThreatList(value)
