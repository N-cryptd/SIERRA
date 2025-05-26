
import gymnasium as gym
from sierra.environment.core import SierraEnv, Actions

class GoalConditionedEnv(gym.Wrapper):
    def __init__(self, env: SierraEnv, goal: int):
        super().__init__(env)
        self.goal = goal
        self.goal_space = ["GATHER_WOOD", "CRAFT_AXE", "EXPLORE", "SURVIVE"]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate intrinsic reward based on the goal
        intrinsic_reward = self._calculate_intrinsic_reward(obs, action, info)
        
        # The worker is driven by the intrinsic reward
        return obs, intrinsic_reward, terminated, truncated, info

    def _calculate_intrinsic_reward(self, obs, action, info):
        reward = 0.0
        goal_name = self.goal_space[self.goal]

        if goal_name == "GATHER_WOOD":
            # Reward for having more wood than before
            if obs["inv_wood"][0] > self.env.agent.inventory.get("wood", 0) - 1:
                reward = 1.0
        
        elif goal_name == "CRAFT_AXE":
            # High reward for crafting an axe
            if action == Actions.CRAFT_AXE.value and self.env.agent.has_axe:
                reward = 10.0

        elif goal_name == "EXPLORE":
            # Reward for visiting new cells (a simple exploration metric)
            # This requires tracking visited cells, which we can add to the base env info dict
            if info.get("new_cell_visited", False):
                reward = 0.1

        elif goal_name == "SURVIVE":
            # For the survival goal, the intrinsic reward is the extrinsic reward
            reward = self.env._calculate_reward(action) # Accessing private method for directness

        return reward
