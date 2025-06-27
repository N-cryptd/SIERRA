
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sierra.environment.core import SierraEnv
from sierra.environment.goal_conditioned_env import GoalConditionedEnv
from sierra.agent.hrl_agent import Manager

# --- HRL Hyperparameters ---
# The number of steps the Worker takes for every one decision the Manager makes.
MANAGER_TIMESTEPS = 10
# The total number of environment steps to train for.
TOTAL_TRAINING_TIMESTEPS = 100000
# The number of goals the Manager can choose from.
NUM_GOALS = 4  # Corresponds to ["GATHER_WOOD", "CRAFT_AXE", "EXPLORE", "SURVIVE"]

def train_hrl_agent():
    print("--- Initializing HRL Training ---")

    # --- Initialize Environments ---
    # Base environment for the Manager
    base_env = DummyVecEnv([lambda: SierraEnv()])
    # A separate, goal-conditioned environment for the Worker
    worker_env = GoalConditionedEnv(SierraEnv(), goal=0) # Start with a default goal
    worker_vec_env = DummyVecEnv([lambda: worker_env])

    # --- Initialize Manager and Worker ---
    obs_dim = base_env.get_attr("get_observation_dim")[0]()
    manager = Manager(obs_dim, NUM_GOALS)
    manager_optimizer = optim.Adam(manager.parameters(), lr=1e-4)

    # The Worker is a standard PPO agent
    worker = PPO("MlpPolicy", worker_vec_env, verbose=0, n_steps=2048, batch_size=64, n_epochs=10)

    # --- Training Loop ---
    print("--- Starting Training Loop ---")
    obs = base_env.reset()
    total_steps = 0

    while total_steps < TOTAL_TRAINING_TIMESTEPS:
        # --- Manager Phase: Select a goal ---
        with torch.no_grad():
            # Flatten the observation dictionary for the Manager
            flat_obs = np.concatenate([v.flatten() for v in obs.values()])
            goal_logits = manager(torch.FloatTensor(flat_obs).unsqueeze(0))
            goal_distribution = torch.distributions.Categorical(logits=goal_logits)
            goal = goal_distribution.sample().item()
        
        # Set the worker's goal
        worker_env.goal = goal
        print(f"Step {total_steps}: Manager selected goal -> {worker_env.goal_space[goal]}")

        # --- Worker Phase: Execute for k steps ---
        cumulative_extrinsic_reward = 0
        worker_obs = obs

        for _ in range(MANAGER_TIMESTEPS):
            # Worker predicts an action based on the current observation
            action, _ = worker.predict(worker_obs, deterministic=True)
            
            # The action is executed in the *base* environment
            obs, extrinsic_reward, done, info = base_env.step(action)
            
            # The worker's environment is updated to reflect the new state
            worker_env.env.agent = base_env.get_attr("agent")[0]
            worker_env.env.grid = base_env.get_attr("grid")[0]
            # ... (a full sync would be needed for all state variables)
            
            # The worker learns based on the *intrinsic* reward from its goal-conditioned env
            intrinsic_reward = worker_env._calculate_intrinsic_reward(worker_obs, obs, action, info[0])
            worker.learn(total_timesteps=1) # This is a simplification; normally we'd collect experience and then learn

            worker_obs = obs
            cumulative_extrinsic_reward += extrinsic_reward[0]
            total_steps += 1

            if done:
                obs = base_env.reset()
                break

        # --- Manager Update Phase ---
        # The manager is rewarded based on the sum of extrinsic rewards the worker achieved
        manager_loss = -goal_distribution.log_prob(torch.tensor(goal)) * cumulative_extrinsic_reward
        
        manager_optimizer.zero_grad()
        manager_loss.backward()
        manager_optimizer.step()

        print(f"Step {total_steps}: Manager updated with cumulative reward: {cumulative_extrinsic_reward:.2f}")

    print("--- HRL Training Complete ---")
    # In a real scenario, we would save the manager and worker models here.

if __name__ == "__main__":
    train_hrl_agent()
