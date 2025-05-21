import gymnasium as gym
import numpy as np
import random

from .grid import create_grid, place_entity, check_boundaries, get_cell_content, EMPTY, AGENT, RESOURCE
from .entities import Agent, Resource

# Environmental Constants
DAY_LENGTH = 100
NIGHT_LENGTH = 50
MAX_WOOD_SOURCES = 2
MAX_STONE_SOURCES = 2
MAX_CHARCOAL_SOURCES = 1
MAX_CLOTH_SOURCES = 1
MAX_INVENTORY_PER_ITEM = 10
MAX_WATER_FILTERS = 5

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
# Action mapping
ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_CRAFT_SHELTER = 4
ACTION_CRAFT_FILTER = 5
ACTION_CRAFT_AXE = 6
ACTION_PURIFY_WATER = 7
NUM_ACTIONS = 8 # 4 move + 3 craft + 1 purify

MAX_MURKY_WATER_SOURCES = 2
PURIFIED_WATER_THIRST_REPLENISH = 40
WOOD_COLLECTION_AXE_BONUS = 2

WEATHER_TYPES = ['clear', 'rainy', 'cloudy']
SEASON_TYPES = ['spring', 'summer', 'autumn', 'winter']
WEATHER_TRANSITION_STEPS = 200
SEASON_TRANSITION_STEPS = 1000

class SierraEnv(gym.Env):
    """A simple grid world environment for the SIERRA project."""

    def __init__(self, grid_width=10, grid_height=10):
        super().__init__()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = create_grid(self.grid_width, self.grid_height)

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "agent_pos": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_width - 1, self.grid_height - 1]), dtype=int),
            "food_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(2)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(2)]), dtype=int),
            "water_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(1)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(1)]), dtype=int),
            "wood_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(MAX_WOOD_SOURCES)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(MAX_WOOD_SOURCES)]), dtype=int),
            "stone_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(MAX_STONE_SOURCES)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(MAX_STONE_SOURCES)]), dtype=int),
            "charcoal_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(MAX_CHARCOAL_SOURCES)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(MAX_CHARCOAL_SOURCES)]), dtype=int),
            "cloth_locs": gym.spaces.Box(low=np.array([[-1, -1] for _ in range(MAX_CLOTH_SOURCES)]), high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(MAX_CLOTH_SOURCES)]), dtype=int),
            "murky_water_locs": gym.spaces.Box(
                low=np.array([[-1, -1] for _ in range(MAX_MURKY_WATER_SOURCES)]),
                high=np.array([[self.grid_width - 1, self.grid_height - 1] for _ in range(MAX_MURKY_WATER_SOURCES)]),
                dtype=int
            ),
            "hunger": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "thirst": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "time_of_day": gym.spaces.Discrete(2), # 0 for Night, 1 for Day
            "current_weather": gym.spaces.Discrete(len(WEATHER_TYPES)),
            "current_season": gym.spaces.Discrete(len(SEASON_TYPES)),
            "inv_wood": gym.spaces.Box(low=0, high=MAX_INVENTORY_PER_ITEM, shape=(1,), dtype=int),
            "inv_stone": gym.spaces.Box(low=0, high=MAX_INVENTORY_PER_ITEM, shape=(1,), dtype=int),
            "inv_charcoal": gym.spaces.Box(low=0, high=MAX_INVENTORY_PER_ITEM, shape=(1,), dtype=int),
            "inv_cloth": gym.spaces.Box(low=0, high=MAX_INVENTORY_PER_ITEM, shape=(1,), dtype=int),
            "inv_murky_water": gym.spaces.Box(low=0, high=MAX_INVENTORY_PER_ITEM, shape=(1,), dtype=int),
            "has_shelter": gym.spaces.Discrete(2), # 0 for False, 1 for True
            "has_axe": gym.spaces.Discrete(2),     # 0 for False, 1 for True
            "water_filters_available": gym.spaces.Box(low=0, high=MAX_WATER_FILTERS, shape=(1,), dtype=int)
        })

        # Define action space
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self.agent = None
        self.resources = []

        # Initialize environmental state
        self.world_time = 0
        self.is_day = True # Start during the day
        self.current_weather = self.np_random.choice(WEATHER_TYPES)
        self.current_season = self.np_random.choice(SEASON_TYPES)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset environmental state
        self.world_time = 0
        self.is_day = True # Start during the day
        self.current_weather = self.np_random.choice(WEATHER_TYPES)
        self.current_season = self.np_random.choice(SEASON_TYPES)

        # Reset grid
        self.grid = create_grid(self.grid_width, self.grid_height)

        # Place agent randomly
        agent_x = self.np_random.integers(0, self.grid_width)
        agent_y = self.np_random.integers(0, self.grid_height)
        self.agent = Agent(agent_x, agent_y, energy=100) # Initialize agent with energy
        place_entity(self.grid, self.agent, AGENT)

        # Place resources randomly (ensure no overlap with agent or other resources)
        self.resources = []
        resource_types = (['food'] * 2 + ['water'] * 1 +
                          ['wood'] * MAX_WOOD_SOURCES +
                          ['stone'] * MAX_STONE_SOURCES +
                          ['charcoal'] * MAX_CHARCOAL_SOURCES +
                          ['cloth'] * MAX_CLOTH_SOURCES + 
                          ['murky_water'] * MAX_MURKY_WATER_SOURCES)

        for res_type in resource_types:
            placed = False
            while not placed:
                res_x = self.np_random.integers(0, self.grid_width)
                res_y = self.np_random.integers(0, self.grid_height)
                if get_cell_content(self.grid, res_x, res_y) == EMPTY:
                    new_resource = Resource(res_x, res_y, type=res_type)
                    self.resources.append(new_resource)
                    place_entity(self.grid, new_resource, RESOURCE)
                    placed = True

        observation = {
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=int),
            "food_locs": np.array([[-1, -1]] * 2, dtype=int),
            "water_locs": np.array([[-1, -1]] * 1, dtype=int),
            "wood_locs": np.array([[-1, -1]] * MAX_WOOD_SOURCES, dtype=int),
            "stone_locs": np.array([[-1, -1]] * MAX_STONE_SOURCES, dtype=int),
            "charcoal_locs": np.array([[-1, -1]] * MAX_CHARCOAL_SOURCES, dtype=int),
            "cloth_locs": np.array([[-1, -1]] * MAX_CLOTH_SOURCES, dtype=int),
            "murky_water_locs": np.array([[-1, -1]] * MAX_MURKY_WATER_SOURCES, dtype=int),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": WEATHER_TYPES.index(self.current_weather),
            "current_season": SEASON_TYPES.index(self.current_season),
            "inv_wood": np.array([self.agent.inventory.get("wood", 0)], dtype=int),
            "inv_stone": np.array([self.agent.inventory.get("stone", 0)], dtype=int),
            "inv_charcoal": np.array([self.agent.inventory.get("charcoal", 0)], dtype=int),
            "inv_cloth": np.array([self.agent.inventory.get("cloth", 0)], dtype=int),
            "inv_murky_water": np.array([self.agent.inventory.get("murky_water", 0)], dtype=int),
            "has_shelter": int(self.agent.has_shelter),
            "has_axe": int(self.agent.has_axe),
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=int)
        }

        food_count = 0
        water_count = 0
        wood_count = 0
        stone_count = 0
        charcoal_count = 0
        cloth_count = 0
        murky_water_count = 0
        for res in self.resources:
            if res.type == 'food' and food_count < 2:
                observation["food_locs"][food_count] = [res.x, res.y]
                food_count += 1
            elif res.type == 'water' and water_count < 1:
                observation["water_locs"][water_count] = [res.x, res.y]
                water_count += 1
            elif res.type == 'wood' and wood_count < MAX_WOOD_SOURCES:
                observation["wood_locs"][wood_count] = [res.x, res.y]
                wood_count += 1
            elif res.type == 'stone' and stone_count < MAX_STONE_SOURCES:
                observation["stone_locs"][stone_count] = [res.x, res.y]
                stone_count += 1
            elif res.type == 'charcoal' and charcoal_count < MAX_CHARCOAL_SOURCES:
                observation["charcoal_locs"][charcoal_count] = [res.x, res.y]
                charcoal_count += 1
            elif res.type == 'cloth' and cloth_count < MAX_CLOTH_SOURCES:
                observation["cloth_locs"][cloth_count] = [res.x, res.y]
                cloth_count += 1
            elif res.type == 'murky_water' and murky_water_count < MAX_MURKY_WATER_SOURCES:
                observation["murky_water_locs"][murky_water_count] = [res.x, res.y]
                murky_water_count += 1
        info = {}

        return observation, info

    def step(self, action):
        if self.agent is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        reward = -0.01 # Small negative step penalty
        terminated = False
        truncated = False
        info = {}

        # Decrement agent needs
        hunger_decay = 0.1
        thirst_decay = 0.1
        if self.agent.has_shelter:
            hunger_decay *= 0.75 # 25% reduction
            thirst_decay *= 0.75 # 25% reduction

        # Slightly increase decay during night
        if not self.is_day:
            hunger_decay *= 1.2
            thirst_decay *= 1.2

        self.agent.hunger -= hunger_decay
        self.agent.thirst -= thirst_decay

        # Store current position
        old_x, old_y = self.agent.x, self.agent.y

        if action < 4: # Movement actions
            # Calculate new position based on action
            if action == ACTION_MOVE_UP:
                new_x, new_y = old_x, old_y - 1
            elif action == ACTION_MOVE_DOWN:
                new_x, new_y = old_x, old_y + 1
            elif action == ACTION_MOVE_LEFT:
                new_x, new_y = old_x - 1, old_y
            elif action == ACTION_MOVE_RIGHT:
                new_x, new_y = old_x + 1, old_y
            else: # Should not happen given the if condition
                new_x, new_y = old_x, old_y

            # Check for boundary collision
            if check_boundaries(self.grid, new_x, new_y):
                # Check for resource collection
                cell_content = get_cell_content(self.grid, new_x, new_y)
                if cell_content == RESOURCE:
                    collected_resource = None
                    for res in self.resources:
                        if res.x == new_x and res.y == new_y:
                            collected_resource = res
                            break

                    if collected_resource:
                        if collected_resource.type == 'food':
                            reward += 0.5 # Reward for collecting food
                            self.agent.hunger = min(100, self.agent.hunger + 30) # Replenish hunger
                        elif collected_resource.type == 'water':
                            reward += 0.5 # Reward for collecting water
                            self.agent.thirst = min(100, self.agent.thirst + 30) # Replenish thirst
                        elif collected_resource.type == 'murky_water':
                            current_amount = self.agent.inventory.get('murky_water', 0)
                            self.agent.inventory['murky_water'] = min(current_amount + 1, MAX_INVENTORY_PER_ITEM)
                            reward += 0.1 # Small reward for collecting murky water
                        elif collected_resource.type == 'wood':
                            amount_to_add = WOOD_COLLECTION_AXE_BONUS if self.agent.has_axe else 1
                            current_amount = self.agent.inventory.get('wood', 0)
                            self.agent.inventory['wood'] = min(current_amount + amount_to_add, MAX_INVENTORY_PER_ITEM)
                            reward += 0.2 # Or adjust reward based on amount collected
                        elif collected_resource.type in Resource.MATERIAL_TYPES: # Handles other materials like stone, charcoal, cloth
                            current_amount = self.agent.inventory.get(collected_resource.type, 0)
                            self.agent.inventory[collected_resource.type] = min(current_amount + 1, MAX_INVENTORY_PER_ITEM)
                            reward += 0.2 # Specific reward for collecting other materials
                        
                        # Remove collected resource from grid and list
                        self.resources.remove(collected_resource)
                        self.grid[new_y, new_x] = EMPTY

                # Move agent if the new position is empty
                if get_cell_content(self.grid, new_x, new_y) == EMPTY:
                    self.grid[old_y, old_x] = EMPTY # Clear old position
                    self.agent.x, self.agent.y = new_x, new_y
                    place_entity(self.grid, self.agent, AGENT)
            else:
                # Collision with wall/boundary
                reward -= 0.1 # Small penalty for hitting wall
                # Agent stays in the same position
        
        elif action == ACTION_CRAFT_SHELTER:
            recipe = CRAFTING_RECIPES["basic_shelter"]
            can_craft = True
            if not self.agent.has_shelter: # Check if shelter already exists
                for material, required_amount in recipe.items():
                    if self.agent.inventory.get(material, 0) < required_amount:
                        can_craft = False
                        break
                if can_craft:
                    for material, required_amount in recipe.items():
                        self.agent.inventory[material] -= required_amount
                    self.agent.has_shelter = True
                    reward += CRAFTING_REWARDS["basic_shelter"]
            # If already has shelter or cannot craft, default step penalty applies
            
        elif action == ACTION_CRAFT_FILTER:
            recipe = CRAFTING_RECIPES["water_filter"]
            can_craft = True
            if self.agent.water_filters_available < MAX_WATER_FILTERS: # Check if max filters reached
                for material, required_amount in recipe.items():
                    if self.agent.inventory.get(material, 0) < required_amount:
                        can_craft = False
                        break
                if can_craft:
                    for material, required_amount in recipe.items():
                        self.agent.inventory[material] -= required_amount
                    self.agent.water_filters_available += 1
                    reward += CRAFTING_REWARDS["water_filter"]
            # If max filters reached or cannot craft, default step penalty applies

        elif action == ACTION_CRAFT_AXE:
            recipe = CRAFTING_RECIPES["crude_axe"]
            can_craft = True
            if not self.agent.has_axe: # Check if axe already exists
                for material, required_amount in recipe.items():
                    if self.agent.inventory.get(material, 0) < required_amount:
                        can_craft = False
                        break
                if can_craft:
                    for material, required_amount in recipe.items():
                        self.agent.inventory[material] -= required_amount
                    self.agent.has_axe = True
                    reward += CRAFTING_REWARDS["crude_axe"]
            # If already has axe or cannot craft, default step penalty applies

        elif action == ACTION_PURIFY_WATER:
            if self.agent.water_filters_available > 0 and self.agent.inventory.get('murky_water', 0) > 0:
                self.agent.water_filters_available -= 1
                self.agent.inventory['murky_water'] -= 1
                self.agent.thirst = min(100, self.agent.thirst + PURIFIED_WATER_THIRST_REPLENISH)
                reward += 0.3 # Reward for successful purification
            # If no filter or no murky water, default step penalty applies
            
        else:
            raise ValueError("Invalid action")

        # Check for low needs penalty (optional)
        if self.agent.hunger <= 20 or self.agent.thirst <= 20:
             reward -= 0.05 # Small penalty for low needs

        # Check for death (hunger or thirst depletion)
        if self.agent.hunger <= 0 or self.agent.thirst <= 0:
            terminated = True
            reward -= 1.0 # Large penalty for death

        # Update environmental state
        self.world_time += 1
        cycle_time = DAY_LENGTH + NIGHT_LENGTH
        self.is_day = (self.world_time % cycle_time) < DAY_LENGTH

        # Basic weather transition
        if self.world_time % WEATHER_TRANSITION_STEPS == 0:
             self.current_weather = self.np_random.choice(WEATHER_TYPES)

        # Basic season transition
        if self.world_time % SEASON_TRANSITION_STEPS == 0:
             self.current_season = self.np_random.choice(SEASON_TYPES)


        # Prepare observation
        food_locs = [(-1, -1)] * 2
        water_locs = [(-1, -1)] * 1
        wood_locs = [(-1, -1)] * MAX_WOOD_SOURCES
        stone_locs = [(-1, -1)] * MAX_STONE_SOURCES
        charcoal_locs = [(-1, -1)] * MAX_CHARCOAL_SOURCES
        cloth_locs = [(-1, -1)] * MAX_CLOTH_SOURCES
        murky_water_locs = [(-1,-1)] * MAX_MURKY_WATER_SOURCES
        food_count = 0
        water_count = 0
        wood_count = 0
        stone_count = 0
        charcoal_count = 0
        cloth_count = 0
        murky_water_count = 0

        for res in self.resources:
            if res.type == 'food' and food_count < 2:
                food_locs[food_count] = (res.x, res.y)
                food_count += 1
            elif res.type == 'water' and water_count < 1:
                water_locs[water_count] = (res.x, res.y)
                water_count += 1
            elif res.type == 'wood' and wood_count < MAX_WOOD_SOURCES:
                wood_locs[wood_count] = (res.x, res.y)
                wood_count += 1
            elif res.type == 'stone' and stone_count < MAX_STONE_SOURCES:
                stone_locs[stone_count] = (res.x, res.y)
                stone_count += 1
            elif res.type == 'charcoal' and charcoal_count < MAX_CHARCOAL_SOURCES:
                charcoal_locs[charcoal_count] = (res.x, res.y)
                charcoal_count += 1
            elif res.type == 'cloth' and cloth_count < MAX_CLOTH_SOURCES:
                cloth_locs[cloth_count] = (res.x, res.y)
                cloth_count += 1
            elif res.type == 'murky_water' and murky_water_count < MAX_MURKY_WATER_SOURCES:
                murky_water_locs[murky_water_count] = (res.x, res.y)
                murky_water_count += 1

        observation = {
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=int),
            "food_locs": np.array(food_locs, dtype=int),
            "water_locs": np.array(water_locs, dtype=int),
            "wood_locs": np.array(wood_locs, dtype=int),
            "stone_locs": np.array(stone_locs, dtype=int),
            "charcoal_locs": np.array(charcoal_locs, dtype=int),
            "cloth_locs": np.array(cloth_locs, dtype=int),
            "murky_water_locs": np.array(murky_water_locs, dtype=int),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": WEATHER_TYPES.index(self.current_weather),
            "current_season": SEASON_TYPES.index(self.current_season),
            "inv_wood": np.array([self.agent.inventory.get("wood", 0)], dtype=int),
            "inv_stone": np.array([self.agent.inventory.get("stone", 0)], dtype=int),
            "inv_charcoal": np.array([self.agent.inventory.get("charcoal", 0)], dtype=int),
            "inv_cloth": np.array([self.agent.inventory.get("cloth", 0)], dtype=int),
            "inv_murky_water": np.array([self.agent.inventory.get("murky_water", 0)], dtype=int),
            "has_shelter": int(self.agent.has_shelter),
            "has_axe": int(self.agent.has_axe),
            "water_filters_available": np.array([self.agent.water_filters_available], dtype=int)
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.grid is None:
            return "Environment not initialized."

        output = ""
        for row in self.grid:
            output += " ".join(["." if cell == EMPTY else "W" if cell == WALL else "A" if cell == AGENT else "R" for cell in row]) + "\n"
        print(output)
        return output

    def close(self):
        pass # No resources to close for this simple environment