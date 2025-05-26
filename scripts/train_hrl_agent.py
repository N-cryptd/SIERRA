import gymnasium as gym
from sierra.environment.core import SierraEnv
from sierra.environment.goal_conditioned_env import GoalConditionedEnv
from sierra.agent.hrl_agent import Manager, Worker
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.optim as optim
import numpy as np

# --- HRL Hyperparameters ---
NUM_GOALS = 4 # 0: GATHER_WOOD, 1: CRAFT_AXE, 2: EXPLORE, 3: SURVIVE
MANAGER_UPDATE_FREQ = 10

def train_hrl_agent(total_timesteps=100000):
    # --- Initialize Environments ---
    base_env = SierraEnv()
    goal_conditioned_envs = [GoalConditionedEnv(base_env, goal) for goal in range(NUM_GOALS)]

    # --- Initialize Manager and Workers ---
    obs_dim = base_env.get_observation_dim()
    manager = Manager(obs_dim, NUM_GOALS)
    manager_optimizer = optim.Adam(manager.parameters(), lr=1e-4)

    workers = [Worker("MlpPolicy", DummyVecEnv([lambda: env]), verbose=0) for env in goal_conditioned_envs]

    # --- Training Loop ---
    obs = base_env.reset()
    for step in range(total_timesteps):
        # --- Manager Phase ---
        with torch.no_grad():
            goal = manager(torch.FloatTensor(obs)).argmax(dim=-1).item()

        # --- Worker Phase ---
        worker_env = goal_conditioned_envs[goal]
        worker = workers[goal]
        action, _ = worker.predict(obs)
        obs, intrinsic_reward, done, info = worker_env.step(action)

        # Train the selected worker
        worker.learn(total_timesteps=1) # Train for a single step

        # --- Update Manager ---
        if step % MANAGER_UPDATE_FREQ == 0:
            extrinsic_reward = base_env._calculate_reward(action)
            manager_loss = -torch.log_softmax(manager(torch.FloatTensor(obs)), dim=-1)[0, goal] * extrinsic_reward
            manager_optimizer.zero_grad()
            manager_loss.backward()
            manager_optimizer.step()

        if done:
            obs = base_env.reset()

    print("HRL agent training complete.")

if __name__ == "__main__":
    train_hrl_agent()