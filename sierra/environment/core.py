import gymnasium as gym
import numpy as np
import random

from .grid import create_grid, place_entity, check_boundaries, get_cell_content, EMPTY, AGENT, RESOURCE, THREAT
from .entities import Agent, Resource, Threat
from enum import Enum

# --- Constants Grouping ---

# Time Cycle
TIME_CONSTANTS = {
    "DAY_LENGTH": 100,
    "NIGHT_LENGTH": 50,
}

# Resource Generation Limits
RESOURCE_LIMITS = {
    "MAX_FOOD_SOURCES": 2,
    "MAX_WATER_SOURCES": 1,
    "MAX_WOOD_SOURCES": 2,
    "MAX_STONE_SOURCES": 2,
    "MAX_CHARCOAL_SOURCES": 1,
    "MAX_CLOTH_SOURCES": 1,
    "MAX_MURKY_WATER_SOURCES": 2,
    "MAX_THREATS": 2,
    "MAX_SHARPENING_STONES": 1,
}

# Inventory and Item Related
INVENTORY_CONSTANTS = {
    "MAX_INVENTORY_PER_ITEM": 10,
    "MAX_WATER_FILTERS": 5, # Max craftable/usable filters by agent
}

# Crafting
CRAFTING_RECIPES = {
    "basic_shelter": {"wood": 4, "stone": 2},
    "water_filter": {"charcoal": 2, "cloth": 1, "stone": 1},
    "crude_axe": {"stone": 2, "wood": 1}
}

CRAFTING_REWARDS = {
    "basic_shelter": 5.0,
    "water_filter": 0.5,
    "crude_axe": 1.0
}

# Action Definitions
class Actions(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    CRAFT_SHELTER = 4
    CRAFT_FILTER = 5
    CRAFT_AXE = 6
    PURIFY_WATER = 7
    REPAIR_AXE = 8

    @classmethod
    def num_actions(cls):
        return len(cls)

# Gameplay Modifiers and Values
GAMEPLAY_CONSTANTS = {
    "INITIAL_HUNGER": 100.0,
    "INITIAL_THIRST": 100.0,
    "MAX_HUNGER": 100.0,
    "MAX_THIRST": 100.0,
    "PURIFIED_WATER_THIRST_REPLENISH": 40,
    "WOOD_COLLECTION_AXE_BONUS": 2,
    "LOW_NEED_THRESHOLD": 20,
    # Decay constants for Agent needs
    "BASE_DECAY": 0.1,
    "SHELTER_MULTIPLIER": 0.75,
    "NIGHT_MULTIPLIER": 1.2,
    "NIGHT_NO_SHELTER_EXTRA_MULTIPLIER": 1.5,
}

# Weather and Seasons
ENVIRONMENT_CYCLE_CONSTANTS = {
    "WEATHER_TYPES": ['clear', 'rainy', 'cloudy'],
    "SEASON_TYPES": ['spring', 'summer', 'autumn', 'winter'],
    "WEATHER_TRANSITION_STEPS": 200,
    "SEASON_TRANSITION_STEPS": 1000,
}


class SierraEnv(gym.Env):
    """A simple grid world environment for the SIERRA project."""

    def __init__(self, grid_width=10, grid_height=10, config=None):
        super().__init__()

        # PyGame initialization
        self.pygame_initialized = False
        self.screen = None
        self.font = None

        self.config = config if config is not None else {}

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = create_grid(self.grid_width, self.grid_height)

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(low=0, high=4, shape=(5, 5), dtype=int),
            "agent_pos": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_width - 1, self.grid_height - 1]), dtype=int),
            "food_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(2)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(2)]), dtype=int),
            "water_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(1)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(1)]), dtype=int),
            "wood_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_WOOD_SOURCES"])]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_WOOD_SOURCES"])]), dtype=int),
            "stone_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_STONE_SOURCES"])]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_STONE_SOURCES"])]), dtype=int),
            "charcoal_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_CHARCOAL_SOURCES"])]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_CHARCOAL_SOURCES"])]), dtype=int),
            "cloth_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_CLOTH_SOURCES"])]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_CLOTH_SOURCES"])]), dtype=int),
            "murky_water_locs": gym.spaces.Box(
                low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_MURKY_WATER_SOURCES"])]),
                high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_MURKY_WATER_SOURCES"])]),
                dtype=int
            ),
            "sharpening_stone_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_SHARPENING_STONES"])]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_SHARPENING_STONES"])]), dtype=int),
            "threat_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(RESOURCE_LIMITS["MAX_THREATS"])]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(RESOURCE_LIMITS["MAX_THREATS"])]), dtype=int),
            "hunger": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "thirst": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "time_of_day": gym.spaces.Discrete(2), # 0 for Night, 1 for Day
            "current_weather": gym.spaces.Discrete(len(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"])),
            "current_season": gym.spaces.Discrete(len(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"])),
            "inv_wood": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=int),
            "inv_stone": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=int),
            "inv_charcoal": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=int),
            "inv_cloth": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=int),
            "inv_murky_water": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"], shape=(1,), dtype=int),
            "has_shelter": gym.spaces.Discrete(2), # 0 for False, 1 for True
            "has_axe": gym.spaces.Discrete(2),     # 0 for False, 1 for True
            "water_filters_available": gym.spaces.Box(low=0, high=INVENTORY_CONSTANTS["MAX_WATER_FILTERS"], shape=(1,), dtype=int)
        })

        # Define action space
        self.action_space = gym.spaces.Discrete(Actions.num_actions())

        self.agent = None
        self.resources = []
        self.threats = []

        # Initialize environmental state
        self.world_time = 0
        self.is_day = True # Start during the day
        self.current_weather = self.np_random.choice(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"])
        self.current_season = self.np_random.choice(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"])

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
        # Get the 5x5 area around the agent
        pov = np.zeros((5, 5), dtype=int)
        for r in range(-2, 3):
            for c in range(-2, 3):
                x, y = self.agent.x + c, self.agent.y + r
                if check_boundaries(self.grid, x, y):
                    pov[r + 2, c + 2] = self.grid[y, x]

        observation = {
            "pov": pov,
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=int),
            "food_locs": np.array(self._get_resource_locations("food", RESOURCE_LIMITS["MAX_FOOD_SOURCES"]), dtype=int),
            "water_locs": np.array(self._get_resource_locations("water", RESOURCE_LIMITS["MAX_WATER_SOURCES"]), dtype=int),
            "wood_locs": np.array(self._get_resource_locations("wood", RESOURCE_LIMITS["MAX_WOOD_SOURCES"]), dtype=int),
            "stone_locs": np.array(self._get_resource_locations("stone", RESOURCE_LIMITS["MAX_STONE_SOURCES"]), dtype=int),
            "charcoal_locs": np.array(self._get_resource_locations("charcoal", RESOURCE_LIMITS["MAX_CHARCOAL_SOURCES"]), dtype=int),
            "cloth_locs": np.array(self._get_resource_locations("cloth", RESOURCE_LIMITS["MAX_CLOTH_SOURCES"]), dtype=int),
            "murky_water_locs": np.array(self._get_resource_locations("murky_water", RESOURCE_LIMITS["MAX_MURKY_WATER_SOURCES"]), dtype=int),
            "sharpening_stone_locs": np.array(self._get_resource_locations("sharpening_stone", RESOURCE_LIMITS["MAX_SHARPENING_STONES"]), dtype=int),
            "threat_locs": np.array(self._get_threat_locations(RESOURCE_LIMITS["MAX_THREATS"]), dtype=int),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"].index(self.current_weather),
            "current_season": ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"].index(self.current_season),
            "inv_wood": np.array([self.agent.get_item_count("wood")], dtype=int),
            "inv_stone": np.array([self.agent.get_item_count("stone")], dtype=int),
            "inv_charcoal": np.array([self.agent.get_item_count("charcoal")], dtype=int),
            "inv_cloth": np.array([self.agent.get_item_count("cloth")], dtype=int),
            "inv_murky_water": np.array([self.agent.get_item_count("murky_water")], dtype=int),
            "has_shelter": int(self.agent.has_shelter),
            "has_axe": int(self.agent.has_axe),
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=int)
        }
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset environmental state
        self.world_time = 0
        self.is_day = True # Start during the day
        self.current_weather = self.np_random.choice(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"])
        self.current_season = self.np_random.choice(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"])

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

        # Define resource counts based on RESOURCE_LIMITS and season
        resource_counts_dict = {
            'food': RESOURCE_LIMITS["MAX_FOOD_SOURCES"],
            'water': RESOURCE_LIMITS["MAX_WATER_SOURCES"],
            'wood': RESOURCE_LIMITS["MAX_WOOD_SOURCES"],
            'stone': RESOURCE_LIMITS["MAX_STONE_SOURCES"],
            'charcoal': RESOURCE_LIMITS["MAX_CHARCOAL_SOURCES"],
            'cloth': RESOURCE_LIMITS["MAX_CLOTH_SOURCES"],
            'murky_water': RESOURCE_LIMITS["MAX_MURKY_WATER_SOURCES"],
            'sharpening_stone': RESOURCE_LIMITS["MAX_SHARPENING_STONES"]
        }

        if self.current_season == 'winter':
            resource_counts_dict['food'] = 1
            resource_counts_dict['water'] = 0
        
        resource_types_to_spawn = []
        for res_type, count in resource_counts_dict.items():
            resource_types_to_spawn.extend([res_type] * count)
        
        # Shuffle the list of resource types to ensure random placement order if desired,
        # though popping from shuffled coords already randomizes location.
        # self.np_random.shuffle(resource_types_to_spawn) # This shuffle is optional

        for res_type in resource_types_to_spawn:
            if not all_coords:
                # This might happen if grid is too small for agent + all resources
                print(f"Warning: Not enough space to place all resources. Grid size: {self.grid_width}x{self.grid_height}, trying to place {res_type}")
                break 
            
            res_x, res_y = all_coords.pop()
            new_resource = Resource(res_x, res_y, type=res_type)
            self.resources.append(new_resource)
            place_entity(self.grid, new_resource, RESOURCE, x=new_resource.x, y=new_resource.y)
        
        # Spawn threats
        max_threats = self.config.get("max_threats", RESOURCE_LIMITS["MAX_THREATS"])
        for _ in range(max_threats):
            if not all_coords:
                print(f"Warning: Not enough space to place all threats. Grid size: {self.grid_width}x{self.grid_height}")
                break
            
            threat_x, threat_y = all_coords.pop()
            new_threat = Threat(threat_x, threat_y)
            self.threats.append(new_threat)
            place_entity(self.grid, new_threat, THREAT, x=new_threat.x, y=new_threat.y)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if self.agent is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        info = {}

        # Calculate rewards
        reward = self._calculate_reward(action)

        self._update_agent_needs()
        collision = self._handle_threats()
        if collision:
            reward -= 0.5

        self._handle_resource_respawn()

        # Store current position
        old_x, old_y = self.agent.x, self.agent.y
        
        # Convert action value to Actions Enum member if necessary for logic
        # current_action = Actions(action) # This line assumes action is an int

        if action < Actions.CRAFT_SHELTER.value: # Movement actions
            _, (new_agent_x, new_agent_y) = self._handle_movement_and_collection(action, old_x, old_y)
            self.agent.x, self.agent.y = new_agent_x, new_agent_y 
        
        elif action in [Actions.CRAFT_SHELTER.value, Actions.CRAFT_FILTER.value, Actions.CRAFT_AXE.value]:
            if self.config.get("crafting", True):
                self._handle_crafting(action)

        elif action == Actions.PURIFY_WATER.value:
            self._handle_purify_water(action)

        elif action == Actions.REPAIR_AXE.value:
            self._handle_repair_axe()
            
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check for death (hunger or thirst depletion)
        if self.agent.is_dead():
            terminated = True

        # Update environmental state
        self.world_time += 1
        cycle_length = TIME_CONSTANTS["DAY_LENGTH"] + TIME_CONSTANTS["NIGHT_LENGTH"]
        self.is_day = (self.world_time % cycle_length) < TIME_CONSTANTS["DAY_LENGTH"]

        # Basic weather transition
        if self.world_time % ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TRANSITION_STEPS"] == 0:
             self.current_weather = self.np_random.choice(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"])

        # Basic season transition
        if self.world_time % ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TRANSITION_STEPS"] == 0:
             self.current_season = self.np_random.choice(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"])
        
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
                elif cell_type == RESOURCE:
                    color = (0, 255, 0)
                elif cell_type == THREAT:
                    color = (255, 0, 255)
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
        if self.agent.hunger <= GAMEPLAY_CONSTANTS["LOW_NEED_THRESHOLD"] or \
           self.agent.thirst <= GAMEPLAY_CONSTANTS["LOW_NEED_THRESHOLD"]:
            reward -= 0.05

        # Penalty for death
        if self.agent.is_dead():
            reward -= 1.0

        # Penalty for hoarding resources
        for item, count in self.agent.inventory.items():
            if count > INVENTORY_CONSTANTS["MAX_INVENTORY_PER_ITEM"] * 0.8:
                reward -= 0.01

        # Reward for crafting
        if action in [Actions.CRAFT_SHELTER.value, Actions.CRAFT_FILTER.value, Actions.CRAFT_AXE.value]:
            item_name = None
            if action == Actions.CRAFT_SHELTER.value:
                item_name = "basic_shelter"
            elif action == Actions.CRAFT_FILTER.value:
                item_name = "water_filter"
            elif action == Actions.CRAFT_AXE.value:
                item_name = "crude_axe"
            
            if self._can_craft(item_name):
                reward += CRAFTING_REWARDS[item_name]

        # Reward for purifying water
        if action == Actions.PURIFY_WATER.value:
            if self.agent.water_filters_available > 0 and self.agent.get_item_count("murky_water") > 0:
                reward += 0.3

        # Penalty for threat contact
        for threat in self.threats:
            if threat.x == self.agent.x and threat.y == self.agent.y:
                reward -= 0.5

        return reward

    def _can_craft(self, item_name):
        """Checks if the agent has the required materials to craft an item."""
        recipe = CRAFTING_RECIPES[item_name]
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
            self.agent.axe_durability = 100
            self.agent.remove_item("sharpening_stone", 1)

    # --- Core Logic Methods Implementation ---

    def _update_agent_needs(self):
        """Delegates updating agent's hunger and thirst to the Agent object."""
        # The environment_has_shelter_effect is self.agent.has_shelter itself
        self.agent.update_needs(self.is_day, self.agent.has_shelter)

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

        # Check for resource at new location
        if get_cell_content(self.grid, new_x, new_y) == RESOURCE:
            resource_to_collect = None
            for res in self.resources:
                if res.x == new_x and res.y == new_y:
                    resource_to_collect = res
                    break
            
            if resource_to_collect:
                collected_amount = 0
                if resource_to_collect.type == 'wood':
                    collected_amount = GAMEPLAY_CONSTANTS['WOOD_COLLECTION_AXE_BONUS'] if self.agent.has_axe and self.agent.axe_durability > 0 else 1
                    if self.agent.has_axe and self.agent.axe_durability > 0:
                        self.agent.axe_durability -= 1
                elif resource_to_collect.type in Resource.MATERIAL_TYPES: # Includes food, water now
                    collected_amount = 1 # Assuming 1 for other resources
                
                # Use agent's method to add item
                self.agent.add_item(resource_to_collect.type, collected_amount)
                
                # Assign reward based on resource type
                # Assuming food/water collection might have direct effects on hunger/thirst later,
                # or their value is just for scoring here.
                if resource_to_collect.type == 'food':
                    action_specific_reward = 0.5 
                    # Potentially: self.agent.replenish_hunger(SOME_AMOUNT) if not just inventory
                elif resource_to_collect.type == 'water':
                    action_specific_reward = 0.5
                    # Potentially: self.agent.replenish_thirst(SOME_AMOUNT) if not just inventory
                elif resource_to_collect.type in ['wood', 'stone', 'charcoal', 'cloth', 'murky_water']:
                     action_specific_reward = 0.1 
                
                resource_to_collect.respawn_timer = 100 # Set respawn timer
                self.grid[new_y, new_x] = EMPTY # Grid cell becomes empty first

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
        else:
            return 0.0 # Should not happen if action validation is correct

        recipe = CRAFTING_RECIPES[item_name]

        # Check preconditions using agent's state
        if item_name == "basic_shelter" and self.agent.has_shelter:
            return 0.0 
        if item_name == "crude_axe" and self.agent.has_axe:
            return 0.0
        if item_name == "water_filter" and self.agent.water_filters_available >= INVENTORY_CONSTANTS['MAX_WATER_FILTERS']:
            return 0.0

        # Check materials using agent's method
        for material, required_amount in recipe.items():
            if not self.agent.has_item(material, required_amount):
                return 0.0 # Not enough materials

        # If successful, deduct materials and update agent state using agent's methods
        for material, required_amount in recipe.items():
            self.agent.remove_item(material, required_amount)
        
        if item_name == "basic_shelter":
            self.agent.set_has_shelter(True)
        elif item_name == "crude_axe":
            self.agent.set_has_axe(True)
        elif item_name == "water_filter":
            self.agent.add_water_filter() # Defaults to adding 1
            
        return CRAFTING_REWARDS[item_name]

    def _handle_purify_water(self, action_value): # action_value is Actions.PURIFY_WATER.value
        """Handles water purification attempts using agent's methods."""
        if self.agent.water_filters_available > 0 and self.agent.get_item_count("murky_water") > 0:
            self.agent.use_water_filter()
            self.agent.remove_item("murky_water", 1)
            self.agent.replenish_thirst(GAMEPLAY_CONSTANTS['PURIFIED_WATER_THIRST_REPLENISH'])
            return 0.3 # Reward for successful purification
        return 0.0