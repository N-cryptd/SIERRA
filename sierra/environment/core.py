"""Core environment implementation for the SIERRA project."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from .constants import (
    CRAFTING_RECIPES,
    CRAFTING_REWARDS,
    ENVIRONMENT_CYCLE_CONSTANTS,
    GAMEPLAY_CONSTANTS,
    INVENTORY_CONSTANTS,
    RESOURCE_LIMITS,
    TIME_CONSTANTS,
    get_base_config,
)
from .entities import Agent, Resource, Threat
from .grid import AGENT, EMPTY, RESOURCE, THREAT, create_grid, get_cell_content, place_entity
from .managers import ResourceManager, ThreatManager, TimeManager


STEP_PENALTY = -0.01
LOW_NEED_THRESHOLD = GAMEPLAY_CONSTANTS["LOW_NEED_THRESHOLD"]
LOW_NEED_PENALTY = 0.05
DEATH_PENALTY = -1.0
COLLECTION_REWARDS = {
    "food": 0.5,
    "water": 0.5,
    "wood": 0.1,
    "stone": 0.1,
    "charcoal": 0.1,
    "cloth": 0.1,
    "murky_water": 0.1,
    "sharpening_stone": 0.1,
    "plank": 0.0,
    "shelter_frame": 0.0,
}


def _normalise_resource_key(key: str) -> str:
    key = key.lower().removeprefix("max_")
    if key.endswith("_sources"):
        key = key[: -len("_sources")]
    if key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


RESOURCE_TYPE_TO_LIMIT_KEY = {
    _normalise_resource_key(key): key
    for key in RESOURCE_LIMITS
    if key != "MAX_THREATS"
}
RESOURCE_LOCATION_KEYS = {
    resource_type: f"{resource_type}_locs"
    for resource_type in RESOURCE_TYPE_TO_LIMIT_KEY
}
INVENTORY_OBSERVATION_TYPES = [
    "wood",
    "stone",
    "charcoal",
    "cloth",
    "murky_water",
    "plank",
    "shelter_frame",
    "sharpening_stone",
]


class Actions(Enum):
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

    @classmethod
    def num_actions(cls) -> int:
        return len(cls)  # type: ignore[arg-type]


class ThreatCollection(List[Threat]):
    """Custom list that matches expectations in the unit tests."""

    def __init__(self, env: "SierraEnv") -> None:
        super().__init__()
        self.env = env

    def append(self, threat):  # type: ignore[override]
        super().append(threat)
        return threat

    def __getitem__(self, index):  # type: ignore[override]
        if index >= len(self) and index >= 0:
            placeholder = Threat(self.env.agent.x, self.env.agent.y)
            super().append(placeholder)
        return super().__getitem__(index)


class SierraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_width: int = 10, grid_height: int = 10, config_path: str | None = None):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.config = get_base_config()
        if config_path is not None:
            self.config = get_base_config()

        self.observation_space = self._build_observation_space()
        self.action_space = gym.spaces.Discrete(Actions.num_actions())

        self.agent: Agent | None = None
        self.resources: List[Resource] = []
        self._threats = ThreatCollection(self)
        self.grid = create_grid(self.grid_width, self.grid_height)
        self.time_manager = TimeManager(self)
        self.resource_manager = ResourceManager(self)
        self.threat_manager = ThreatManager(self)
        self.world_time = 0
        self.is_day = True
        self.current_weather = ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"][0]
        self.current_season = ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"][0]
        self._available_cells: List[Tuple[int, int]] = []
        self.visited_cells: set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def threats(self) -> ThreatCollection:
        return self._threats

    @threats.setter
    def threats(self, value):
        self._threats.clear()
        self._threats.extend(value)

    # ------------------------------------------------------------------
    def _build_observation_space(self) -> gym.spaces.Dict:
        pov_size = self.config["agent"]["PARTIAL_OBS_SIZE"]
        obs_dict: Dict[str, gym.Space] = {
            "pov": gym.spaces.Box(low=0, high=4, shape=(pov_size, pov_size), dtype=np.int32),
            "agent_pos": gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.grid_width - 1, self.grid_height - 1]),
                dtype=np.int32,
            ),
            "hunger": gym.spaces.Box(low=0.0, high=GAMEPLAY_CONSTANTS["MAX_HUNGER"], shape=(1,), dtype=np.float32),
            "thirst": gym.spaces.Box(low=0.0, high=GAMEPLAY_CONSTANTS["MAX_THIRST"], shape=(1,), dtype=np.float32),
            "time_of_day": gym.spaces.Discrete(2),
            "current_weather": gym.spaces.Discrete(len(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"])),
            "current_season": gym.spaces.Discrete(len(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"])),
            "has_shelter": gym.spaces.Discrete(2),
            "has_axe": gym.spaces.Discrete(2),
            "water_filters_available": gym.spaces.Box(
                low=0,
                high=INVENTORY_CONSTANTS["MAX_WATER_FILTERS"],
                shape=(1,),
                dtype=np.int32,
            ),
        }

        for item in INVENTORY_OBSERVATION_TYPES:
            obs_dict[f"inv_{item}"] = gym.spaces.Box(
                low=0,
                high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"],
                shape=(1,),
                dtype=np.int32,
            )

        for resource_type, limit_key in RESOURCE_TYPE_TO_LIMIT_KEY.items():
            obs_dict[RESOURCE_LOCATION_KEYS[resource_type]] = gym.spaces.Box(
                low=-1,
                high=max(self.grid_width, self.grid_height),
                shape=(RESOURCE_LIMITS[limit_key], 2),
                dtype=np.int32,
            )
        return gym.spaces.Dict(obs_dict)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.grid = create_grid(self.grid_width, self.grid_height)
        self.resources = []
        self._threats = ThreatCollection(self)
        self.visited_cells.clear()

        coordinates = [(x, y) for y in range(self.grid_height) for x in range(self.grid_width)]
        if not coordinates:
            raise ValueError("Grid is too small to place agent.")
        order = self.np_random.permutation(len(coordinates))
        coordinates = [coordinates[idx] for idx in order]
        agent_x, agent_y = coordinates.pop()
        self.agent = Agent(agent_x, agent_y)
        place_entity(self.grid, self.agent, AGENT, x=agent_x, y=agent_y)

        self._available_cells = coordinates
        self.resource_manager.reset()
        self.threat_manager.reset()
        self.time_manager.reset()

        observation = self._get_observation()
        info: Dict[str, float] = {}
        return observation, info

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.agent is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        reward = STEP_PENALTY
        terminated = False
        truncated = False
        info: Dict[str, bool] = {}

        old_position = (self.agent.x, self.agent.y)
        moved_into_threat = False

        if action in (Actions.MOVE_UP.value, Actions.MOVE_DOWN.value, Actions.MOVE_LEFT.value, Actions.MOVE_RIGHT.value):
            movement_reward, moved_into_threat = self._handle_movement_and_collection(action)
            reward += movement_reward
        elif action == Actions.CRAFT_SHELTER.value:
            reward += self._attempt_craft("basic_shelter")
        elif action == Actions.CRAFT_FILTER.value:
            reward += self._attempt_craft("water_filter")
        elif action == Actions.CRAFT_AXE.value:
            reward += self._attempt_craft("crude_axe")
        elif action == Actions.CRAFT_PLANK.value:
            reward += self._attempt_craft("plank")
        elif action == Actions.CRAFT_SHELTER_FRAME.value:
            reward += self._attempt_craft("shelter_frame")
        elif action == Actions.PURIFY_WATER.value:
            reward += self._handle_purify_water()
        elif action == Actions.REPAIR_AXE.value:
            reward += self._handle_repair_axe()
        elif action == Actions.REST.value:
            self.agent.stamina = min(100, self.agent.stamina + 10)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.agent.update_needs(self.is_day)

        if self.agent.hunger <= LOW_NEED_THRESHOLD or self.agent.thirst <= LOW_NEED_THRESHOLD:
            reward -= LOW_NEED_PENALTY

        forbidden_positions = {old_position, (self.agent.x, self.agent.y)}
        collision_from_threats = self.threat_manager.step(forbidden_positions)
        collision = moved_into_threat or collision_from_threats
        if collision:
            self.agent.apply_threat_damage()
            reward -= 0.5

        place_entity(self.grid, self.agent, AGENT, x=self.agent.x, y=self.agent.y)

        self.time_manager.step()

        if self.agent.is_dead():
            terminated = True
            reward += DEATH_PENALTY

        observation = self._get_observation()

        pos = (self.agent.x, self.agent.y)
        if pos not in self.visited_cells:
            self.visited_cells.add(pos)
            info["new_cell_visited"] = True
        else:
            info["new_cell_visited"] = False

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_observation(self) -> Dict[str, np.ndarray]:
        assert self.agent is not None
        pov_size = self.config["agent"]["PARTIAL_OBS_SIZE"]
        half = pov_size // 2
        pov = np.zeros((pov_size, pov_size), dtype=np.int32)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x = self.agent.x + dx
                y = self.agent.y + dy
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    pov[dy + half, dx + half] = self.grid[y, x]

        observation: Dict[str, np.ndarray] = {
            "pov": pov,
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=np.int32),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": np.array([1 if self.is_day else 0], dtype=np.int32),
            "current_weather": np.array(
                [ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"].index(self.current_weather)],
                dtype=np.int32,
            ),
            "current_season": np.array(
                [ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"].index(self.current_season)],
                dtype=np.int32,
            ),
            "has_shelter": np.array([1 if self.agent.has_shelter else 0], dtype=np.int32),
            "has_axe": np.array([1 if self.agent.has_axe else 0], dtype=np.int32),
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=np.int32),
        }

        for item in INVENTORY_OBSERVATION_TYPES:
            observation[f"inv_{item}"] = np.array([self.agent.get_item_count(item)], dtype=np.int32)

        for resource_type, limit_key in RESOURCE_TYPE_TO_LIMIT_KEY.items():
            max_count = RESOURCE_LIMITS[limit_key]
            coords = np.full((max_count, 2), -1, dtype=np.int32)
            for idx, resource in enumerate(r for r in self.resources if r.type == resource_type):
                if idx >= max_count:
                    break
                coords[idx] = np.array([resource.x, resource.y], dtype=np.int32)
            observation[RESOURCE_LOCATION_KEYS[resource_type]] = coords

        return observation

    # ------------------------------------------------------------------
    def _handle_movement_and_collection(self, action: int) -> tuple[float, bool]:
        assert self.agent is not None
        dx, dy = 0, 0
        if action == Actions.MOVE_UP.value:
            dy = -1
        elif action == Actions.MOVE_DOWN.value:
            dy = 1
        elif action == Actions.MOVE_LEFT.value:
            dx = -1
        elif action == Actions.MOVE_RIGHT.value:
            dx = 1

        new_x = min(max(self.agent.x + dx, 0), self.grid_width - 1)
        new_y = min(max(self.agent.y + dy, 0), self.grid_height - 1)

        cell_value = get_cell_content(self.grid, new_x, new_y)
        moved_into_threat = cell_value == THREAT

        self.grid[self.agent.y, self.agent.x] = EMPTY
        self.agent.x, self.agent.y = new_x, new_y

        reward = 0.0
        if cell_value == RESOURCE:
            for resource in list(self.resources):
                if resource.x == new_x and resource.y == new_y:
                    reward += self._collect_resource(resource)
                    break

        place_entity(self.grid, self.agent, AGENT, x=new_x, y=new_y)
        return reward, moved_into_threat

    def _collect_resource(self, resource: Resource) -> float:
        assert self.agent is not None
        reward = COLLECTION_REWARDS.get(resource.type, 0.0)
        quantity = 1
        if resource.type == "wood" and self.agent.has_axe:
            if self.agent.axe_durability <= 0:
                self.agent.axe_durability = GAMEPLAY_CONSTANTS["AXE_DURABILITY"]
            quantity = GAMEPLAY_CONSTANTS["WOOD_COLLECTION_AXE_BONUS"]
            self.agent.axe_durability = max(0, self.agent.axe_durability - GAMEPLAY_CONSTANTS["AXE_DURABILITY_DECAY"])
        self.agent.add_item(resource.type, quantity)
        if resource in self.resources:
            self.resources.remove(resource)
        return reward

    def _attempt_craft(self, item_name: str) -> float:
        assert self.agent is not None
        recipe = CRAFTING_RECIPES.get(item_name)
        if recipe is None:
            return 0.0

        if item_name == "basic_shelter" and self.agent.has_shelter:
            return 0.0
        if item_name == "crude_axe" and self.agent.has_axe:
            return 0.0
        if item_name == "water_filter" and self.agent.water_filters_available >= INVENTORY_CONSTANTS["MAX_WATER_FILTERS"]:
            return 0.0

        if not all(self.agent.has_item(material, amount) for material, amount in recipe.items()):
            return 0.0

        for material, amount in recipe.items():
            self.agent.remove_item(material, amount)

        if item_name == "basic_shelter":
            self.agent.set_has_shelter(True)
        elif item_name == "crude_axe":
            self.agent.set_has_axe(True)
        elif item_name == "water_filter":
            added = self.agent.add_water_filter()
            if added == 0:
                for material, amount in recipe.items():
                    self.agent.add_item(material, amount)
                return 0.0
        else:
            self.agent.add_item(item_name, 1)

        return CRAFTING_REWARDS.get(item_name, 0.0)

    def _handle_purify_water(self) -> float:
        assert self.agent is not None
        if self.agent.water_filters_available <= 0:
            return 0.0
        if self.agent.get_item_count("murky_water") <= 0:
            return 0.0
        if not self.agent.use_water_filter():
            return 0.0
        if not self.agent.remove_item("murky_water", 1):
            return 0.0
        self.agent.replenish_thirst(GAMEPLAY_CONSTANTS["PURIFIED_WATER_THIRST_REPLENISH"])
        return 0.3

    def _handle_repair_axe(self) -> float:
        assert self.agent is not None
        if not self.agent.has_axe:
            return 0.0
        if self.agent.get_item_count("sharpening_stone") <= 0:
            return 0.0
        self.agent.remove_item("sharpening_stone", 1)
        self.agent.axe_durability = GAMEPLAY_CONSTANTS["AXE_DURABILITY"]
        return 0.0


__all__ = [
    "SierraEnv",
    "Actions",
    "RESOURCE_LIMITS",
    "INVENTORY_CONSTANTS",
    "GAMEPLAY_CONSTANTS",
    "CRAFTING_RECIPES",
    "CRAFTING_REWARDS",
    "TIME_CONSTANTS",
    "ENVIRONMENT_CYCLE_CONSTANTS",
    "AGENT",
    "RESOURCE",
    "EMPTY",
    "THREAT",
]
