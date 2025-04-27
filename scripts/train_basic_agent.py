import gymnasium as gym
from sierra.environment.core import SierraEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap the environment
env = DummyVecEnv([lambda: SierraEnv()])

# Instantiate the agent
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

print("Basic agent training complete.")