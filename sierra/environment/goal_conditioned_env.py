
import gymnasium as gym
from sierra.environment.core import SierraEnv, Actions

class GoalConditionedEnv(gym.Wrapper):
    def __init__(self, env: SierraEnv, goal: int):
        super().__init__(env)
        self.goal = goal
        self.goal_space = ["GATHER_WOOD", "CRAFT_AXE", "EXPLORE", "SURVIVE"]

    def step(self, action):
        # Store the state before the step
        prev_obs = self.env._get_observation()

        # Take the step in the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate intrinsic reward based on the goal and the state change
        intrinsic_reward = self._calculate_intrinsic_reward(prev_obs, obs, action, info)
        
        # The worker is driven by the intrinsic reward
        return obs, intrinsic_reward, terminated, truncated, info

    def _calculate_intrinsic_reward(self, prev_obs, current_obs, action, info):
        reward = -0.01 # Small penalty for taking a step
        goal_name = self.goal_space[self.goal]

        if goal_name == "GATHER_WOOD":
            # Reward for each piece of wood gathered this step
            wood_before = prev_obs["inv_wood"][0]
            wood_after = current_obs["inv_wood"][0]
            if wood_after > wood_before:
                reward += (wood_after - wood_before) # Reward scaled by amount gathered
        
        elif goal_name == "CRAFT_AXE":
            # High reward only on the step the axe is successfully crafted
            if not prev_obs["has_axe"] and current_obs["has_axe"]:
                reward = 10.0

        elif goal_name == "EXPLORE":
            # Reward for visiting a new cell
            if info.get("new_cell_visited", False):
                reward = 0.1

        elif goal_name == "SURVIVE":
            # For the survival goal, the intrinsic reward is the extrinsic (environmental) reward
            reward = self.env._calculate_reward(action)

        return reward
