import gymnasium as gym
import numpy as np
import random

from .grid import create_grid, place_entity, check_boundaries, get_cell_content, EMPTY, AGENT, RESOURCE
from .entities import Agent, Resource

class SierraEnv(gym.Env):
    """A simple grid world environment for the SIERRA project."""

    def __init__(self, grid_width=10, grid_height=10):
        super().__init__()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = create_grid(self.grid_width, self.grid_height)

        # Define observation space (grid state)
        # Define observation space (agent and resource positions)
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_width - 1, self.grid_height - 1]), dtype=int),
            "resource": gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_width - 1, self.grid_height - 1]), dtype=int),
        })

        # Define action space (0: up, 1: down, 2: left, 3: right)
        self.action_space = gym.spaces.Discrete(4)

        self.agent = None
        self.resources = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset grid
        self.grid = create_grid(self.grid_width, self.grid_height)

        # Place agent randomly
        agent_x = self.np_random.integers(0, self.grid_width)
        agent_y = self.np_random.integers(0, self.grid_height)
        self.agent = Agent(agent_x, agent_y, energy=100) # Initialize agent with energy
        place_entity(self.grid, self.agent, AGENT)

        # Place one resource randomly (ensure it's not on the agent's initial position)
        resource_x, resource_y = agent_x, agent_y
        while resource_x == agent_x and resource_y == agent_y:
            resource_x = self.np_random.integers(0, self.grid_width)
            resource_y = self.np_random.integers(0, self.grid_height)
        self.resources = [Resource(resource_x, resource_y)]
        place_entity(self.grid, self.resources[0], RESOURCE)

        observation = {
            "agent": np.array([self.agent.x, self.agent.y], dtype=int),
            "resource": np.array([self.resources[0].x, self.resources[0].y], dtype=int) if self.resources else np.array([-1, -1], dtype=int), # Use -1, -1 if no resource
        }
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

        # Decrement agent energy
        self.agent.energy -= 1

        # Check for boundary collision
        if check_boundaries(self.grid, new_x, new_y):
            # Check for resource collection
            if get_cell_content(self.grid, new_x, new_y) == RESOURCE:
                reward += 1.0 # Reward for collecting resource
                self.agent.energy = min(100, self.agent.energy + 50) # Replenish energy, cap at 100
                # Remove resource from grid and list
                self.resources = [] # Assuming only one resource for now
                self.grid[new_y, new_x] = EMPTY
                # Do not terminate on resource collection anymore
            elif get_cell_content(self.grid, new_x, new_y) == EMPTY:
                # Move agent
                self.grid[old_y, old_x] = EMPTY # Clear old position
                self.agent.x, self.agent.y = new_x, new_y
                place_entity(self.grid, self.agent, AGENT)
        else:
            # Collision with wall/boundary
            reward -= 0.1 # Small penalty for hitting wall
            # Agent stays in the same position

        # Check for energy depletion
        if self.agent.energy <= 0:
            terminated = True
            reward -= 1.0 # Penalty for running out of energy

        observation = {
            "agent": np.array([self.agent.x, self.agent.y], dtype=int),
            "resource": np.array([self.resources[0].x, self.resources[0].y], dtype=int) if self.resources else np.array([-1, -1], dtype=int), # Use -1, -1 if no resource
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