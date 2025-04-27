import gymnasium as gym
from sierra.environment.core import SierraEnv
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a basic agent on SierraEnv.')
parser.add_argument('--total_timesteps', type=int, default=10000,
                    help='Total number of training timesteps')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='Learning rate for the agent')
parser.add_argument('--log_dir', type=str, default="./tensorboard_logs/",
                    help='Directory for TensorBoard logs')
parser.add_argument('--save_path', type=str, default="./trained_model",
                    help='Path to save the trained model')
parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC', 'DQN'],
                    help='Stable Baselines3 algorithm to use (PPO, SAC, DQN)')

args = parser.parse_args()

# Create log directory if it doesn't exist
os.makedirs(args.log_dir, exist_ok=True)

# Wrap the environment
env = DummyVecEnv([lambda: SierraEnv()])

# Instantiate the agent
if args.algo == 'PPO':
    model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=args.learning_rate, tensorboard_log=args.log_dir)
elif args.algo == 'SAC':
    # SAC specific hyperparameters - using reasonable defaults for now
    model = SAC("MultiInputPolicy", env, verbose=1, learning_rate=args.learning_rate, tensorboard_log=args.log_dir, buffer_size=100000, learning_starts=1000, tau=0.005, gamma=0.99)
elif args.algo == 'DQN':
    # DQN specific hyperparameters - using reasonable defaults for now
    model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=args.learning_rate, tensorboard_log=args.log_dir, buffer_size=100000, learning_starts=1000, gamma=0.99, exploration_final_eps=0.01)
else:
    raise ValueError(f"Unknown algorithm: {args.algo}")

# Train the agent
model.learn(total_timesteps=args.total_timesteps)

# Save the trained agent
model.save(args.save_path)

print(f"Basic agent training complete. Model saved to {args.save_path}")