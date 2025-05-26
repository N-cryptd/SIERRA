import gymnasium as gym
from sierra.environment.core import SierraEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import os
import numpy as np

class SimpleRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SimpleRewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.rewards.append(self.locals['infos'][0]['episode']['r'])
        return True
    
    def get_mean_reward(self):
        if not self.rewards:
            return 0
        return np.mean(self.rewards)

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a basic agent on SierraEnv.')
parser.add_argument('--total_timesteps', type=int, default=10000,
                    help='Total number of training timesteps')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='Learning rate for the agent')
parser.add_argument('--n_steps', type=int, default=2048,
                    help='Number of steps to run for each environment per update')
parser.add_argument('--clip_range', type=float, default=0.2,
                    help='Clipping parameter for PPO')
parser.add_argument('--log_dir', type=str, default="./tensorboard_logs/",
                    help='Directory for TensorBoard logs')
parser.add_argument('--save_path', type=str, default="./trained_model",
                    help='Path to save the trained model')
parser.add_argument('--load_path', type=str, default=None,
                    help='Path to load a pre-trained model')

args = parser.parse_args()

# Create log directory if it doesn't exist
os.makedirs(args.log_dir, exist_ok=True)

# Wrap the environment
env = DummyVecEnv([lambda: SierraEnv()])

# Instantiate the agent
if args.load_path:
    model = PPO.load(args.load_path, env=env)
else:
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0, # Set to 0 for tuning
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        clip_range=args.clip_range,
        tensorboard_log=args.log_dir
    )

# Train the agent
callback = SimpleRewardCallback()
model.learn(total_timesteps=args.total_timesteps, callback=callback)

# Save the trained agent
model.save(args.save_path)

# The last line of output should be the mean reward
print(callback.get_mean_reward())