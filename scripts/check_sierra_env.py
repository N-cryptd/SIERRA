import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from sierra.environment.core import SierraEnv

# Instantiate the environment
env = SierraEnv()

# Check the environment
print("Running Gymnasium environment check...")
check_env(env.unwrapped)

print("Environment check passed!")