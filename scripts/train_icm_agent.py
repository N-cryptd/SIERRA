

import gymnasium as gym
from sierra.environment.core import SierraEnv
from sierra.agent.icm import ICM
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import os
import numpy as np
import torch

class ICMCallback(BaseCallback):
    def __init__(self, icm, icm_optimizer, verbose=0):
        super(ICMCallback, self).__init__(verbose)
        self.icm = icm
        self.icm_optimizer = icm_optimizer

    def _on_step(self) -> bool:
        # Get the latest transition
        state = self.locals['obs_tensor']
        next_state = self.locals['new_obs_tensor']
        action = self.locals['actions']

        # Calculate intrinsic reward
        predicted_next_state_feat, next_state_feat, predicted_action = self.icm(state, next_state, action)
        intrinsic_reward = (next_state_feat - predicted_next_state_feat).pow(2).mean(dim=1)

        # Add intrinsic reward to the environment's reward
        self.locals['rewards'] += intrinsic_reward.detach().numpy()

        # Update ICM
        forward_loss = nn.functional.mse_loss(predicted_next_state_feat, next_state_feat.detach())
        inverse_loss = nn.functional.cross_entropy(predicted_action.logits, action)
        icm_loss = forward_loss + inverse_loss

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        return True

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a PPO agent with ICM on SierraEnv.')
parser.add_argument('--total_timesteps', type=int, default=10000,
                    help='Total number of training timesteps')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='Learning rate for the agent')
parser.add_argument('--log_dir', type=str, default="./tensorboard_logs/",
                    help='Directory for TensorBoard logs')
parser.add_argument('--save_path', type=str, default="./trained_icm_model",
                    help='Path to save the trained model')

args = parser.parse_args()

# Create log directory if it doesn't exist
os.makedirs(args.log_dir, exist_ok=True)

# Wrap the environment
env = DummyVecEnv([lambda: SierraEnv()])

# ICM setup
input_dim = env.get_attr("get_observation_dim")[0]()

action_dim = env.action_space.n
icm = ICM(input_dim, action_dim)
icm_optimizer = torch.optim.Adam(icm.parameters(), lr=args.learning_rate)

# Instantiate the agent
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=args.learning_rate, tensorboard_log=args.log_dir)

# Train the agent
callback = ICMCallback(icm, icm_optimizer)
model.learn(total_timesteps=args.total_timesteps, callback=callback)

# Save the trained agent
model.save(args.save_path)

print(f"ICM agent training complete. Model saved to {args.save_path}")

