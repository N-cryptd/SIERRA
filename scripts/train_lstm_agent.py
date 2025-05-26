
import gymnasium as gym
from sierra.environment.core import SierraEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a recurrent PPO agent on SierraEnv.')
parser.add_argument('--total_timesteps', type=int, default=10000,
                    help='Total number of training timesteps')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='Learning rate for the agent')
parser.add_argument('--log_dir', type=str, default="./tensorboard_logs/",
                    help='Directory for TensorBoard logs')
parser.add_argument('--save_path', type=str, default="./trained_lstm_model.zip",
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
    model = PPO("MlpLstmPolicy", env, verbose=1, learning_rate=args.learning_rate, tensorboard_log=args.log_dir)

# Train the agent
model.learn(total_timesteps=args.total_timesteps)

# Save the trained agent
model.save(args.save_path)

print(f"Recurrent PPO agent training complete. Model saved to {args.save_path}")
