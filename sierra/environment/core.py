import gymnasium as gym
import numpy as np
import random

from .grid import create_grid, place_entity, check_boundaries, get_cell_content, EMPTY, AGENT, RESOURCE
from .entities import Agent, Resource

# Environmental Constants
DAY_LENGTH = 100
NIGHT_LENGTH = 50
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
            "hunger": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "thirst": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "time_of_day": gym.spaces.Discrete(2), # 0 for Night, 1 for Day
            "current_weather": gym.spaces.Discrete(len(WEATHER_TYPES)),
            "current_season": gym.spaces.Discrete(len(SEASON_TYPES)),
        })

        # Define action space (0: up, 1: down, 2: left, 3: right)
        self.action_space = gym.spaces.Discrete(4)

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
        resource_types = ['food'] * 2 + ['water'] * 1

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
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": WEATHER_TYPES.index(self.current_weather),
            "current_season": SEASON_TYPES.index(self.current_season),
        }

        food_count = 0
        water_count = 0
        for res in self.resources:
            if res.type == 'food' and food_count < 2:
                observation["food_locs"][food_count] = [res.x, res.y]
                food_count += 1
            elif res.type == 'water' and water_count < 1:
                observation["water_locs"][water_count] = [res.x, res.y]
                water_count += 1

        info = {}

        return observation, info

    def step(self, action):
        if self.agent is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")

        # Store current position
        old_x, old_y = self.agent.x, self.agent.y

        # Calculate new position based on action
        if action == 0: # Up
            new_x, new_y = old_x, old_y - 1
        elif action == 1: # Down
            new_x, new_y = old_x, old_y + 1
        elif action == 2: # Left
            new_x, new_y = old_x - 1, old_y
        elif action == 3: # Right
            new_x, new_y = old_x + 1, old_y
        else:
            raise ValueError("Invalid action")

        reward = -0.01 # Small negative step penalty
        terminated = False
        truncated = False
        info = {}

        # Decrement agent needs
        hunger_decay = 0.1
        thirst_decay = 0.1

        # Slightly increase decay during night
        if not self.is_day:
            hunger_decay *= 1.2
            thirst_decay *= 1.2

        self.agent.hunger -= hunger_decay
        self.agent.thirst -= thirst_decay

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
        food_count = 0
        water_count = 0

        for res in self.resources:
            if res.type == 'food' and food_count < 2:
                food_locs[food_count] = (res.x, res.y)
                food_count += 1
            elif res.type == 'water' and water_count < 1:
                water_locs[water_count] = (res.x, res.y)
                water_count += 1

        observation = {
            "agent_pos": np.array([self.agent.x, self.agent.y], dtype=int),
            "food_locs": np.array(food_locs, dtype=int),
            "water_locs": np.array(water_locs, dtype=int),
            "hunger": np.array([self.agent.hunger], dtype=np.float32),
            "thirst": np.array([self.agent.thirst], dtype=np.float32),
            "time_of_day": int(self.is_day),
            "current_weather": WEATHER_TYPES.index(self.current_weather),
            "current_season": SEASON_TYPES.index(self.current_season),
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