"""Core environment implementation used in the kata tests."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from .constants import (
    AGENT_CONFIG,
    CRAFTING_RECIPES,
    CRAFTING_REWARDS,
    ENVIRONMENT_CYCLE_CONSTANTS,
    GAMEPLAY_CONSTANTS,
    INVENTORY_CONSTANTS,
    RESOURCE_LIMITS,
    TIME_CONSTANTS,
    limit_key_to_resource,
)
from .entities import Agent, Resource, Threat
from .grid import AGENT, EMPTY, RESOURCE, THREAT, check_boundaries, create_grid, get_cell_content, place_entity, iter_coordinates
from .managers import ResourceManager, ThreatManager, TimeManager

_STEP_PENALTY = -0.01
_LOW_NEED_THRESHOLD = GAMEPLAY_CONSTANTS.get("LOW_NEED_THRESHOLD", 20)
_DEATH_PENALTY = -1.0
_THREAT_DAMAGE = GAMEPLAY_CONSTANTS.get("THREAT_DAMAGE", 0.5)
_WOOD_AXE_BONUS = GAMEPLAY_CONSTANTS.get("WOOD_COLLECTION_AXE_BONUS", 2)
_AXE_DECAY = GAMEPLAY_CONSTANTS.get("AXE_DURABILITY_DECAY", 1)
_PURIFIED_WATER_REPLENISH = GAMEPLAY_CONSTANTS.get("PURIFIED_WATER_THIRST_REPLENISH", 0)
_RESOURCE_RESPAWN_TIME = GAMEPLAY_CONSTANTS.get("RESOURCE_RESPAWN_TIME", 0)

_COLLECTION_REWARDS = {
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


class ThreatCollection(list):
    def __init__(self, iterable=None):
        super().__init__()
        if iterable:
            for item in iterable:
                self.append(item)

    def __getitem__(self, index):
        if isinstance(index, int) and index >= len(self):
            while len(self) <= index:
                super().append(Threat(-1, -1))
        return super().__getitem__(index)

    def append(self, item=None):
        if item is None:
            item = Threat(-1, -1)
        super().append(item)
        return item


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
        return len(cls)


class SierraEnv(gym.Env):
    metadata: Dict[str, str] = {}

    def __init__(self, grid_width: int = 10, grid_height: int = 10, config_path: str | None = None):  # noqa: ARG002
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = create_grid(self.grid_width, self.grid_height)

        self.agent: Agent | None = None
        self.resources: List[Resource] = []
        self._threats: ThreatCollection = ThreatCollection()

        self._available_coords: List[Tuple[int, int]] = []

        self.resource_manager = ResourceManager(self)
        self.threat_manager = ThreatManager(self)
        self.time_manager = TimeManager(self)

        self.world_time = 0
        self.is_day = True
        self.current_weather = "clear"
        self.current_season = "spring"
        self._last_agent_position: Tuple[int, int] = (0, 0)

        self.max_resource_counts = {
            limit_key_to_resource(key): value
            for key, value in RESOURCE_LIMITS.items()
        }

        pov = AGENT_CONFIG.get("PARTIAL_OBS_SIZE", 5)
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(low=0, high=4, shape=(pov, pov), dtype=int),
            "agent_pos": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_width - 1, self.grid_height - 1]), dtype=int),
            "hunger": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "thirst": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "time_of_day": gym.spaces.Discrete(2),
            "current_weather": gym.spaces.Discrete(max(1, len(ENVIRONMENT_CYCLE_CONSTANTS.get("WEATHER_TYPES", [])))),
            "current_season": gym.spaces.Discrete(max(1, len(ENVIRONMENT_CYCLE_CONSTANTS.get("SEASON_TYPES", [])))),
            "inv_wood": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS.get("MAX_INVENTORY_PER_ITEM", 10), shape=(1,), dtype=int),
            "inv_stone": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS.get("MAX_INVENTORY_PER_ITEM", 10), shape=(1,), dtype=int),
            "inv_charcoal": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS.get("MAX_INVENTORY_PER_ITEM", 10), shape=(1,), dtype=int),
            "inv_cloth": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS.get("MAX_INVENTORY_PER_ITEM", 10), shape=(1,), dtype=int),
            "inv_murky_water": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS.get("MAX_INVENTORY_PER_ITEM", 10), shape=(1,), dtype=int),
            "has_shelter": gym.spaces.Discrete(2),
            "has_axe": gym.spaces.Discrete(2),
            "water_filters_available": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS.get("MAX_WATER_FILTERS", 5), shape=(1,), dtype=int),
        })
        self.action_space = gym.spaces.Discrete(Actions.num_actions())

    def reset(self, seed: int | None = None, options: Dict | None = None):  # noqa: D401, ARG002
        super().reset(seed=seed)
        self.grid = create_grid(self.grid_width, self.grid_height)
        self._available_coords = list(iter_coordinates(self.grid_width, self.grid_height))
        self.np_random.shuffle(self._available_coords)

        if not self._available_coords:
            raise ValueError("Grid is too small to place agent")

        agent_x, agent_y = self._available_coords.pop()
        self.agent = Agent(agent_x, agent_y)
        place_entity(self.grid, self.agent, AGENT)

        self.time_manager.reset()
        self.resource_manager.reset()
        self.threat_manager.reset()

        observation = self._get_observation()
        return observation, {}

    def step(self, action: int):
        if self.agent is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        reward = _STEP_PENALTY
        terminated = False
        truncated = False

        previous_position = (self.agent.x, self.agent.y)
        action_reward = 0.0
        if action == Actions.MOVE_UP.value:
            action_reward += self._move_agent(0, -1)
        elif action == Actions.MOVE_DOWN.value:
            action_reward += self._move_agent(0, 1)
        elif action == Actions.MOVE_LEFT.value:
            action_reward += self._move_agent(-1, 0)
        elif action == Actions.MOVE_RIGHT.value:
            action_reward += self._move_agent(1, 0)
        elif action in [Actions.CRAFT_SHELTER.value, Actions.CRAFT_FILTER.value, Actions.CRAFT_AXE.value, Actions.CRAFT_PLANK.value, Actions.CRAFT_SHELTER_FRAME.value]:
            action_reward += self._handle_crafting(action)
        elif action == Actions.PURIFY_WATER.value:
            action_reward += self._handle_purify_water()
        elif action == Actions.REPAIR_AXE.value:
            self._handle_repair_axe()
        elif action == Actions.REST.value:
            self._handle_rest()
        else:
            raise ValueError(f"Unknown action: {action}")

        self.agent.update_needs(self.is_day, self.agent.has_shelter)

        if self.agent.hunger <= _LOW_NEED_THRESHOLD or self.agent.thirst <= _LOW_NEED_THRESHOLD:
            reward -= 0.05

        self._last_agent_position = previous_position
        collision = self.threat_manager.step()
        if collision:
            reward -= _THREAT_DAMAGE

        self.time_manager.step()
        self.resource_manager.step()
        place_entity(self.grid, self.agent, AGENT, x=self.agent.x, y=self.agent.y)

        if self.agent.is_dead():
            reward += _DEATH_PENALTY
            terminated = True

        observation = self._get_observation()
        return observation, reward + action_reward, terminated, truncated, {}

    def render(self, mode: str = "human"):
        return None

    def close(self):  # noqa: D401
        return None

    # --- Internal helpers -------------------------------------------------

    def _move_agent(self, dx: int, dy: int) -> float:
        assert self.agent is not None
        new_x = self.agent.x + dx
        new_y = self.agent.y + dy
        if not check_boundaries(self.grid, new_x, new_y):
            return 0.0

        cell = get_cell_content(self.grid, new_x, new_y)
        reward = 0.0

        if cell == RESOURCE:
            reward += self._collect_resource_at(new_x, new_y)
        elif cell == THREAT:
            reward -= _THREAT_DAMAGE
            threat = next((t for t in self.threats if t.x == new_x and t.y == new_y), None)
            if threat:
                self.threats.remove(threat)

        # Remove any lingering threat at the agent's previous location
        self.threats = [t for t in self.threats if not (t.x == self.agent.x and t.y == self.agent.y)]
        self.grid[self.agent.y, self.agent.x] = EMPTY
        place_entity(self.grid, self.agent, AGENT, x=new_x, y=new_y)
        return reward

    def _collect_resource_at(self, x: int, y: int) -> float:
        assert self.agent is not None
        resource = next((res for res in self.resources if isinstance(res, Resource) and res.x == x and res.y == y), None)
        if resource is None:
            return 0.0

        reward = _COLLECTION_REWARDS.get(resource.type, 0.0)
        amount_collected = 1
        if resource.type == "wood" and self.agent.has_axe:
            amount_collected = _WOOD_AXE_BONUS
            if self.agent.axe_durability > 0:
                self.agent.axe_durability = max(0, self.agent.axe_durability - _AXE_DECAY)
        self.agent.add_item(resource.type, amount_collected)
        resource.respawn_timer = _RESOURCE_RESPAWN_TIME
        if resource in self.resources:
            self.resources.remove(resource)
        return reward

    def _handle_crafting(self, action_value: int) -> float:
        assert self.agent is not None
        action_map = {
            Actions.CRAFT_SHELTER.value: "basic_shelter",
            Actions.CRAFT_FILTER.value: "water_filter",
            Actions.CRAFT_AXE.value: "crude_axe",
            Actions.CRAFT_PLANK.value: "plank",
            Actions.CRAFT_SHELTER_FRAME.value: "shelter_frame",
        }
        item_name = action_map[action_value]

        if not self._can_craft(item_name):
            return 0.0

        recipe = CRAFTING_RECIPES[item_name]
        for material, required_amount in recipe.items():
            self.agent.remove_item(material, required_amount)

        if item_name == "basic_shelter":
            if self.agent.has_shelter:
                return 0.0
            self.agent.set_has_shelter(True)
        elif item_name == "crude_axe":
            if self.agent.has_axe:
                return 0.0
            self.agent.set_has_axe(True)
        elif item_name == "water_filter":
            if not self.agent.add_water_filter():
                return 0.0
        else:
            self.agent.add_item(item_name, 1)

        return CRAFTING_REWARDS.get(item_name, 0.0)

    def _handle_purify_water(self) -> float:
        assert self.agent is not None
        if self.agent.water_filters_available > 0 and self.agent.get_item_count("murky_water") > 0:
            self.agent.use_water_filter()
            self.agent.remove_item("murky_water", 1)
            self.agent.replenish_thirst(_PURIFIED_WATER_REPLENISH)
            return 0.3
        return 0.0

    def _handle_repair_axe(self) -> None:
        assert self.agent is not None
        if self.agent.has_axe and self.agent.has_item("sharpening_stone", 1):
            self.agent.remove_item("sharpening_stone", 1)
            self.agent.axe_durability = GAMEPLAY_CONSTANTS.get("AXE_DURABILITY", 100)

    def _handle_rest(self) -> None:
        assert self.agent is not None
        self.agent.stamina = min(100, self.agent.stamina + 10)

    def _can_craft(self, item_name: str) -> bool:
        assert self.agent is not None
        if item_name not in CRAFTING_RECIPES:
            return False
        if item_name == "basic_shelter" and self.agent.has_shelter:
            return False
        if item_name == "crude_axe" and self.agent.has_axe:
            return False
        if item_name == "water_filter" and self.agent.water_filters_available >= INVENTORY_CONSTANTS.get("MAX_WATER_FILTERS", 5):
            return False
        recipe = CRAFTING_RECIPES[item_name]
        return all(self.agent.has_item(material, amount) for material, amount in recipe.items())

    def _get_observation(self) -> Dict[str, np.SimpleArray | int]:
        assert self.agent is not None
        pov_size = AGENT_CONFIG.get("PARTIAL_OBS_SIZE", 5)
        pov = np.zeros((pov_size, pov_size), dtype=int)
        radius = pov_size // 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = self.agent.x + dx
                y = self.agent.y + dy
                if check_boundaries(self.grid, x, y):
                    pov[dy + radius, dx + radius] = self.grid[y, x]

        weather_types = ENVIRONMENT_CYCLE_CONSTANTS.get("WEATHER_TYPES", [])
        season_types = ENVIRONMENT_CYCLE_CONSTANTS.get("SEASON_TYPES", [])
        weather_index = weather_types.index(self.current_weather) if self.current_weather in weather_types else 0
        season_index = season_types.index(self.current_season) if self.current_season in season_types else 0

        observation = {
            "pov": pov,
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=int),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": 1 if self.is_day else 0,
            "current_weather": weather_index,
            "current_season": season_index,
            "inv_wood": np.array([self.agent.get_item_count("wood")], dtype=int),
            "inv_stone": np.array([self.agent.get_item_count("stone")], dtype=int),
            "inv_charcoal": np.array([self.agent.get_item_count("charcoal")], dtype=int),
            "inv_cloth": np.array([self.agent.get_item_count("cloth")], dtype=int),
            "inv_murky_water": np.array([self.agent.get_item_count("murky_water")], dtype=int),
            "has_shelter": 1 if self.agent.has_shelter else 0,
            "has_axe": 1 if self.agent.has_axe else 0,
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=int),
        }

        for resource_type, max_count in self.max_resource_counts.items():
            key = f"{resource_type}_locs"
            locations = [(-1, -1)] * max_count
            index = 0
            for resource in self.resources:
                if resource.type == resource_type and index < max_count:
                    locations[index] = (resource.x, resource.y)
                    index += 1
            observation[key] = np.array(locations, dtype=int)

        return observation

    @property
    def threats(self) -> ThreatCollection:
        return self._threats

    @threats.setter
    def threats(self, value):
        self._threats = ThreatCollection(value)


__all__ = [
    "Actions",
    "SierraEnv",
    "RESOURCE_LIMITS",
    "GAMEPLAY_CONSTANTS",
    "INVENTORY_CONSTANTS",
    "CRAFTING_RECIPES",
    "CRAFTING_REWARDS",
    "TIME_CONSTANTS",
    "ENVIRONMENT_CYCLE_CONSTANTS",
    "AGENT",
    "RESOURCE",
    "EMPTY",
]
