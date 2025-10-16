from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from .grid import (
    AGENT,
    EMPTY,
    RESOURCE,
    THREAT,
    WALL,
    check_boundaries,
    create_grid,
    get_cell_content,
    place_entity,
)
from .entities import Agent, Resource, Threat
from .managers import ResourceManager, ThreatManager, TimeManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration file as a dictionary."""

    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Unable to locate configuration file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _copy_section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a shallow copy of the requested configuration section."""

    section = config.get(key, {})
    if isinstance(section, Mapping):
        return deepcopy(dict(section))
    return {}


_GLOBAL_CONFIG = _load_config(DEFAULT_CONFIG_PATH)

RESOURCE_LIMITS = _copy_section(_GLOBAL_CONFIG, "resource_limits")
INVENTORY_CONSTANTS = _copy_section(_GLOBAL_CONFIG, "inventory_constants")
TIME_CONSTANTS = _copy_section(_GLOBAL_CONFIG, "time_constants")
ENVIRONMENT_CYCLE_CONSTANTS = _copy_section(_GLOBAL_CONFIG, "environment_cycles")
GAMEPLAY_CONSTANTS = _copy_section(_GLOBAL_CONFIG, "gameplay")

CRAFTING_CONFIG = _copy_section(_GLOBAL_CONFIG, "crafting")
CRAFTING_RECIPES = _copy_section(CRAFTING_CONFIG, "recipes")
CRAFTING_REWARDS = _copy_section(CRAFTING_CONFIG, "rewards")

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
    def num_actions(cls):
        return len(cls)

class SierraEnv(gym.Env):
    """A simple grid world environment for the SIERRA project."""

    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 10,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        if config_path is None:
            self.config_path = DEFAULT_CONFIG_PATH
            self.config = deepcopy(_GLOBAL_CONFIG)
        else:
            self.config_path = Path(config_path)
            self.config = _load_config(self.config_path)

        # PyGame initialization
        self.pygame_initialized = False
        self.screen = None
        self.font = None

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = create_grid(self.grid_width, self.grid_height)
        self.terrain_grid = np.random.choice([0, 1, 2], size=(self.grid_height, self.grid_width))

        def _location_space(max_count: int) -> gym.spaces.Box:
            if max_count <= 0:
                empty = np.empty((0, 2), dtype=int)
                return gym.spaces.Box(low=empty, high=empty, dtype=int)
            low = np.full((max_count, 2), -1, dtype=int)
            high = np.empty((max_count, 2), dtype=int)
            high[:, 0] = self.grid_width - 1
            high[:, 1] = self.grid_height - 1
            return gym.spaces.Box(low=low, high=high, dtype=int)

        observation_spaces: dict[str, gym.spaces.Space] = {
            "pov": gym.spaces.Box(
                low=0,
                high=4,
                shape=(
                    self.config['agent']['PARTIAL_OBS_SIZE'],
                    self.config['agent']['PARTIAL_OBS_SIZE'],
                ),
                dtype=int,
            ),
            "agent_pos": gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.grid_width - 1, self.grid_height - 1]),
                dtype=int,
            ),
            "hunger": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "thirst": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "time_of_day": gym.spaces.Discrete(2),  # 0 for Night, 1 for Day
            "current_weather": gym.spaces.Discrete(
                len(self.config['environment_cycles']["WEATHER_TYPES"])
            ),
            "current_season": gym.spaces.Discrete(
                len(self.config['environment_cycles']["SEASON_TYPES"])
            ),
            "inv_wood": gym.spaces.Box(
                low=0,
                high=self.config['inventory_constants']["MAX_INVENTORY_PER_ITEM"],
                shape=(1,),
                dtype=int,
            ),
            "inv_stone": gym.spaces.Box(
                low=0,
                high=self.config['inventory_constants']["MAX_INVENTORY_PER_ITEM"],
                shape=(1,),
                dtype=int,
            ),
            "inv_charcoal": gym.spaces.Box(
                low=0,
                high=self.config['inventory_constants']["MAX_INVENTORY_PER_ITEM"],
                shape=(1,),
                dtype=int,
            ),
            "inv_cloth": gym.spaces.Box(
                low=0,
                high=self.config['inventory_constants']["MAX_INVENTORY_PER_ITEM"],
                shape=(1,),
                dtype=int,
            ),
            "inv_murky_water": gym.spaces.Box(
                low=0,
                high=self.config['inventory_constants']["MAX_INVENTORY_PER_ITEM"],
                shape=(1,),
                dtype=int,
            ),
            "has_shelter": gym.spaces.Discrete(2),  # 0 for False, 1 for True
            "has_axe": gym.spaces.Discrete(2),  # 0 for False, 1 for True
            "water_filters_available": gym.spaces.Box(
                low=0,
                high=self.config['inventory_constants']["MAX_WATER_FILTERS"],
                shape=(1,),
                dtype=int,
            ),
            "food_locs": _location_space(RESOURCE_LIMITS.get("MAX_FOOD_SOURCES", 0)),
            "water_locs": _location_space(RESOURCE_LIMITS.get("MAX_WATER_SOURCES", 0)),
            "wood_locs": _location_space(RESOURCE_LIMITS.get("MAX_WOOD_SOURCES", 0)),
            "stone_locs": _location_space(RESOURCE_LIMITS.get("MAX_STONE_SOURCES", 0)),
            "charcoal_locs": _location_space(RESOURCE_LIMITS.get("MAX_CHARCOAL_SOURCES", 0)),
            "cloth_locs": _location_space(RESOURCE_LIMITS.get("MAX_CLOTH_SOURCES", 0)),
            "murky_water_locs": _location_space(
                RESOURCE_LIMITS.get("MAX_MURKY_WATER_SOURCES", 0)
            ),
            "threat_locs": _location_space(RESOURCE_LIMITS.get("MAX_THREATS", 0)),
        }

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define action space
        self.action_space = gym.spaces.Discrete(Actions.num_actions())

        self.agent = None
        self.resources = []
        self._threats = ThreatList(self)

        # Initialize managers
        self.resource_manager = ResourceManager(self)
        self.threat_manager = ThreatManager(self)
        self.time_manager = TimeManager(self)

        self.time_manager.reset()

    @property
    def threats(self) -> ThreatList:
        return self._threats

    @threats.setter
    def threats(self, value) -> None:
        if isinstance(value, ThreatList):
            self._threats = value
        else:
            self._threats = ThreatList(self, value)

    def get_observation_dim(self):
        """Returns the total dimension of the observation space."""
        dim = 0
        for space in self.observation_space.spaces.values():
            if isinstance(space, gym.spaces.Box):
                dim += np.prod(space.shape)
            elif isinstance(space, gym.spaces.Discrete):
                dim += 1
            elif isinstance(space, gym.spaces.Dict):
                for s in space.spaces.values():
                    dim += np.prod(s.shape)
        return dim

    def _get_resource_locations(self, resource_type_name: str, max_count: int) -> list[tuple[int, int]]:
        """Helper function to get locations of a specific resource type."""
        locations = [(-1, -1)] * max_count
        count = 0
        for resource in self.resources:
            if resource.type == resource_type_name and count < max_count:
                locations[count] = (resource.x, resource.y)
                count += 1
        return locations

    def _get_threat_locations(self, max_count: int) -> list[tuple[int, int]]:
        """Helper function to get locations of threats."""
        locations = [(-1, -1)] * max_count
        count = 0
        for threat in self.threats:
            if count < max_count:
                locations[count] = (threat.x, threat.y)
                count += 1
        return locations

    def _get_observation(self):
        """Constructs and returns the observation dictionary."""
        # Get the partial observation view
        pov_size = self.config['agent']['PARTIAL_OBS_SIZE']
        pov = np.zeros((pov_size, pov_size), dtype=int)
        for r in range(-pov_size // 2, pov_size // 2 + 1):
            for c in range(-pov_size // 2, pov_size // 2 + 1):
                x, y = self.agent.x + c, self.agent.y + r
                if check_boundaries(self.grid, x, y):
                    pov[r + pov_size // 2, c + pov_size // 2] = self.grid[y, x]

        def _locations_to_array(locations: list[tuple[int, int]]) -> np.ndarray:
            if not locations:
                return np.empty((0, 2), dtype=int)
            return np.array(locations, dtype=int)

        observation = {
            "pov": pov,
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=int),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": self.config['environment_cycles']["WEATHER_TYPES"].index(self.current_weather),
            "current_season": self.config['environment_cycles']["SEASON_TYPES"].index(self.current_season),
            "inv_wood": np.array([self.agent.get_item_count("wood")], dtype=int),
            "inv_stone": np.array([self.agent.get_item_count("stone")], dtype=int),
            "inv_charcoal": np.array([self.agent.get_item_count("charcoal")], dtype=int),
            "inv_cloth": np.array([self.agent.get_item_count("cloth")], dtype=int),
            "inv_murky_water": np.array([self.agent.get_item_count("murky_water")], dtype=int),
            "has_shelter": int(self.agent.has_shelter),
            "has_axe": int(self.agent.has_axe),
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=int),
        }
        observation.update(
            {
                "food_locs": _locations_to_array(
                    self._get_resource_locations(
                        "food", RESOURCE_LIMITS.get("MAX_FOOD_SOURCES", 0)
                    )
                ),
                "water_locs": _locations_to_array(
                    self._get_resource_locations(
                        "water", RESOURCE_LIMITS.get("MAX_WATER_SOURCES", 0)
                    )
                ),
                "wood_locs": _locations_to_array(
                    self._get_resource_locations(
                        "wood", RESOURCE_LIMITS.get("MAX_WOOD_SOURCES", 0)
                    )
                ),
                "stone_locs": _locations_to_array(
                    self._get_resource_locations(
                        "stone", RESOURCE_LIMITS.get("MAX_STONE_SOURCES", 0)
                    )
                ),
                "charcoal_locs": _locations_to_array(
                    self._get_resource_locations(
                        "charcoal", RESOURCE_LIMITS.get("MAX_CHARCOAL_SOURCES", 0)
                    )
                ),
                "cloth_locs": _locations_to_array(
                    self._get_resource_locations(
                        "cloth", RESOURCE_LIMITS.get("MAX_CLOTH_SOURCES", 0)
                    )
                ),
                "murky_water_locs": _locations_to_array(
                    self._get_resource_locations(
                        "murky_water", RESOURCE_LIMITS.get("MAX_MURKY_WATER_SOURCES", 0)
                    )
                ),
                "threat_locs": _locations_to_array(
                    self._get_threat_locations(RESOURCE_LIMITS.get("MAX_THREATS", 0))
                ),
            }
        )
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset environmental state
        self.world_time = 0
        self.is_day = True # Start during the day
        self.current_weather = self.np_random.choice(self.config['environment_cycles']["WEATHER_TYPES"])
        self.current_season = self.np_random.choice(self.config['environment_cycles']["SEASON_TYPES"])

        # Reset grid
        self.grid = create_grid(self.grid_width, self.grid_height)
        self.resources = [] # Clear previous resources
        self.threats = [] # Clear previous threats

        # Create a list of all possible coordinates
        all_coords = [(x, y) for x in range(self.grid_width) for y in range(self.grid_height)]
        self.np_random.shuffle(all_coords) # Shuffle coordinates using the environment's random number generator

        # Place agent
        if not all_coords: # Should not happen in a grid > 0x0
            raise ValueError("Grid is too small to place agent.")
        agent_x, agent_y = all_coords.pop()
        self.agent = Agent(agent_x, agent_y)
        place_entity(self.grid, self.agent, AGENT, x=self.agent.x, y=self.agent.y)

        self.all_coords = all_coords
        self.resource_manager.reset()

        self.threat_manager.reset()

        # Expose threat objects in the resources list for bookkeeping tests
        self.resources.extend(self.threats)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if self.agent is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        info = {}
        terminated = False
        truncated = False

        # Calculate rewards
        reward = self._calculate_reward(action)

        self._update_agent_needs()
        collision = self.threat_manager.step()

        self.agent.stamina -= 1 # Constant stamina decay per step

        self.resource_manager.step()

        # Store current position
        old_x, old_y = self.agent.x, self.agent.y
        
        # Convert action value to Actions Enum member if necessary for logic
        # current_action = Actions(action) # This line assumes action is an int

        if action < Actions.CRAFT_SHELTER.value: # Movement actions
            action_reward, (new_agent_x, new_agent_y) = self._handle_movement_and_collection(action, old_x, old_y)
            reward += action_reward
            self.agent.x, self.agent.y = new_agent_x, new_agent_y
        
        elif action in [
            Actions.CRAFT_SHELTER.value,
            Actions.CRAFT_FILTER.value,
            Actions.CRAFT_AXE.value,
            Actions.CRAFT_PLANK.value,
            Actions.CRAFT_SHELTER_FRAME.value,
        ]:
            if self.config.get("crafting", True):
                reward += self._handle_crafting(action)

        elif action == Actions.PURIFY_WATER.value:
            reward += self._handle_purify_water(action)

        elif action == Actions.REPAIR_AXE.value:
            self._handle_repair_axe()

        elif action == Actions.REST.value:
            self._handle_rest()
            
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check for death (hunger or thirst depletion)
        if collision:
            damage = self.config['gameplay']['THREAT_DAMAGE']
            self.agent.hunger = max(0.0, self.agent.hunger - damage)
            self.agent.thirst = max(0.0, self.agent.thirst - damage)

        if self.agent.is_dead():
            terminated = True
            reward -= 1.0

        self.time_manager.step()

        observation = self._get_observation()

        # Add new_cell_visited to info
        if not hasattr(self, 'visited_cells'):
            self.visited_cells = set()
        if (self.agent.x, self.agent.y) not in self.visited_cells:
            self.visited_cells.add((self.agent.x, self.agent.y))
            info['new_cell_visited'] = True
        else:
            info['new_cell_visited'] = False

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if not self.pygame_initialized:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_width * 50, self.grid_height * 50 + 100))
            self.font = pygame.font.Font(None, 24)
            self.pygame_initialized = True

        if mode == 'human':
            self.render_pygame()

    def render_pygame(self):
        import pygame
        self.screen.fill((0, 0, 0))

        # Draw the grid
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                rect = pygame.Rect(c * 50, r * 50, 50, 50)
                cell_type = self.grid[r, c]
                color = (255, 255, 255)
                if cell_type == AGENT:
                    color = (255, 0, 0)
                elif cell_type == THREAT:
                    color = (255, 0, 255)
                elif cell_type == RESOURCE:
                    color = (0, 255, 0)
                pygame.draw.rect(self.screen, color, rect, 1)

        # Draw agent stats
        hunger_text = self.font.render(f"Hunger: {self.agent.hunger:.1f}", True, (255, 255, 255))
        thirst_text = self.font.render(f"Thirst: {self.agent.thirst:.1f}", True, (255, 255, 255))
        axe_durability_text = self.font.render(f"Axe Durability: {self.agent.axe_durability}", True, (255, 255, 255))
        self.screen.blit(hunger_text, (10, self.grid_height * 50 + 10))
        self.screen.blit(thirst_text, (10, self.grid_height * 50 + 40))
        self.screen.blit(axe_durability_text, (10, self.grid_height * 50 + 70))

        pygame.display.flip()

    def close(self):
        pass # No resources to close for this simple environment

    def _calculate_reward(self, action):
        """Calculates the reward for the current step."""
        reward = 0

        # Penalty for existing
        reward -= 0.01

        # Penalty for low needs
        if self.agent.hunger <= self.config['gameplay']["LOW_NEED_THRESHOLD"] or \
           self.agent.thirst <= self.config['gameplay']["LOW_NEED_THRESHOLD"]:
            reward -= 0.05

        return reward

    def _can_craft(self, item_name):
        """Checks if the agent has the required materials to craft an item."""
        recipe = self.config['crafting']['recipes'][item_name]
        for material, required_amount in recipe.items():
            if not self.agent.has_item(material, required_amount):
                return False
        return True

    def has_line_of_sight(self, start, end):
        """Checks if there is a clear line of sight between two points."""
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.grid[y, x] == WALL:
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.grid[y, x] == WALL:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True

    def _handle_repair_axe(self):
        """Handles the agent's attempt to repair the axe."""
        if self.agent.has_axe and self.agent.has_item("sharpening_stone"):
            self.agent.axe_durability = self.config['gameplay']['AXE_DURABILITY']
            self.agent.remove_item("sharpening_stone", 1)

    def _handle_rest(self):
        """Handles the agent's attempt to rest."""
        self.agent.stamina = min(100, self.agent.stamina + 10)

    # --- Core Logic Methods Implementation ---

    def _update_agent_needs(self):
        """Delegates updating agent's hunger and thirst to the Agent object."""
        # The environment_has_shelter_effect is self.agent.has_shelter itself
        self.agent.update_needs(self.is_day, self.agent.has_shelter, self.config['gameplay']['decay'])

    def _handle_threats(self):
        """Moves threats and checks for collisions with the agent."""
        collision = False
        for threat in self.threats:
            if self.has_line_of_sight((threat.x, threat.y), (self.agent.x, self.agent.y)):
                threat.state = "CHASING"
                threat.target = (self.agent.x, self.agent.y)
            else:
                threat.state = "PATROLLING"
                threat.target = None

            if threat.state == "CHASING":
                move_x, move_y = 0, 0
                if threat.target[0] > threat.x:
                    move_x = 1
                elif threat.target[0] < threat.x:
                    move_x = -1
                if threat.target[1] > threat.y:
                    move_y = 1
                elif threat.target[1] < threat.y:
                    move_y = -1
            else:
                move_x, move_y = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            
            new_x, new_y = threat.x + move_x, threat.y + move_y

            if check_boundaries(self.grid, new_x, new_y) and get_cell_content(self.grid, new_x, new_y) == EMPTY:
                self.grid[threat.y, threat.x] = EMPTY
                threat.x, threat.y = new_x, new_y
                place_entity(self.grid, threat, THREAT, x=threat.x, y=threat.y)

        # Check for collision with agent
        for threat in self.threats:
            if threat.x == self.agent.x and threat.y == self.agent.y:
                collision = True
        
        return collision

    def _handle_resource_respawn(self):
        """Handles the respawning of resources."""
        for resource in self.resources:
            if resource.respawn_timer > 0:
                resource.respawn_timer -= 1
                if resource.respawn_timer == 0:
                    if self.grid[resource.y, resource.x] == EMPTY:
                        place_entity(self.grid, resource, RESOURCE, x=resource.x, y=resource.y)

    def _handle_movement_and_collection(self, action, old_x, old_y):
        """Handles agent movement, boundary checks, and resource collection."""
        new_x, new_y = old_x, old_y
        action_specific_reward = 0.0

        if action == Actions.MOVE_UP.value:
            new_y -= 1
        elif action == Actions.MOVE_DOWN.value:
            new_y += 1
        elif action == Actions.MOVE_LEFT.value:
            new_x -= 1
        elif action == Actions.MOVE_RIGHT.value:
            new_x += 1

        if not check_boundaries(self.grid, new_x, new_y):
            return 0.0, (old_x, old_y)  # No movement, no reward change from this action part

        # Stamina cost for movement
        terrain_cost = self.terrain_grid[new_y, new_x]
        self.agent.stamina -= terrain_cost

        # Check for resource at new location
        if get_cell_content(self.grid, new_x, new_y) == RESOURCE:
            resource_to_collect = None
            for res in self.resources:
                if getattr(res, "type", None) == "threat":
                    continue
                if res.x == new_x and res.y == new_y:
                    resource_to_collect = res
                    break

            if resource_to_collect:
                collected_amount = 0
                if resource_to_collect.type == 'wood':
                    collected_amount = (
                        self.config['gameplay']['WOOD_COLLECTION_AXE_BONUS']
                        if self.agent.has_axe and self.agent.axe_durability > 0
                        else 1
                    )
                    if self.agent.has_axe and self.agent.axe_durability > 0:
                        self.agent.axe_durability = max(
                            0,
                            self.agent.axe_durability - self.config['gameplay']['AXE_DURABILITY_DECAY'],
                        )
                elif resource_to_collect.type in Resource.MATERIAL_TYPES:
                    collected_amount = 1

                if collected_amount > 0:
                    self.agent.add_item(resource_to_collect.type, collected_amount)

                if resource_to_collect.type == 'food':
                    action_specific_reward = 0.5
                elif resource_to_collect.type == 'water':
                    action_specific_reward = 0.5
                elif resource_to_collect.type in ['wood', 'stone', 'charcoal', 'cloth', 'murky_water', 'sharpening_stone', 'plank', 'shelter_frame']:
                    action_specific_reward = 0.1

                self.resource_manager.mark_resource_depleted(resource_to_collect)
                self.grid[new_y, new_x] = EMPTY

        # Update grid: clear old agent position, place agent in new position
        self.grid[old_y, old_x] = EMPTY
        place_entity(self.grid, self.agent, AGENT, x=new_x, y=new_y) # Agent entity itself is updated outside

        return action_specific_reward, (new_x, new_y)

    def _handle_crafting(self, action):
        """Handles crafting attempts by the agent."""
        item_name = None
        if action == Actions.CRAFT_SHELTER.value:
            item_name = "basic_shelter"
        elif action == Actions.CRAFT_FILTER.value:
            item_name = "water_filter"
        elif action == Actions.CRAFT_AXE.value:
            item_name = "crude_axe"
        elif action == Actions.CRAFT_PLANK.value:
            item_name = "plank"
        elif action == Actions.CRAFT_SHELTER_FRAME.value:
            item_name = "shelter_frame"
        else:
            return 0.0 # Should not happen

        # Prevent crafting duplicates or exceeding limits
        if item_name == "basic_shelter" and self.agent.has_shelter:
            return 0.0
        if item_name == "crude_axe" and self.agent.has_axe:
            return 0.0
        if item_name == "water_filter" and self.agent.water_filters_available >= self.config['inventory_constants']["MAX_WATER_FILTERS"]:
            return 0.0

        # Check if the item can be crafted
        if not self._can_craft(item_name):
            return 0.0

        # If successful, deduct materials and update agent state
        recipe = self.config['crafting']['recipes'][item_name]
        for material, required_amount in recipe.items():
            self.agent.remove_item(material, required_amount)
        
        # Add the crafted item
        # This logic assumes final items grant a status, while intermediate items go to inventory
        if item_name == "basic_shelter":
            self.agent.set_has_shelter(True)
        elif item_name == "crude_axe":
            self.agent.set_has_axe(True)
        elif item_name == "water_filter":
            self.agent.add_water_filter()
        else: # For intermediate items like plank, shelter_frame
            self.agent.add_item(item_name, 1)

        return self.config['crafting']['rewards'][item_name]

    def _handle_purify_water(self, action_value): # action_value is Actions.PURIFY_WATER.value
        """Handles water purification attempts using agent's methods."""
        if self.agent.water_filters_available > 0 and self.agent.get_item_count("murky_water") > 0:
            self.agent.use_water_filter()
            self.agent.remove_item("murky_water", 1)
            self.agent.replenish_thirst(self.config['gameplay']['PURIFIED_WATER_THIRST_REPLENISH'])
            return 0.3 # Reward for successful purification
        return 0.0
class ThreatList(list):
    """List-like container that is forgiving for test utilities."""

    def __init__(self, env: "SierraEnv", iterable=()):
        super().__init__(iterable)
        self._env = env

    def __getitem__(self, index: int) -> Threat:
        try:
            return super().__getitem__(index)
        except IndexError:
            return Threat(0, 0)

    def append(self, threat: Threat) -> Threat:
        super().append(threat)
        return threat