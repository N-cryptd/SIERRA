import gymnasium as gym
from sierra.environment.core import SierraEnv

# Instantiate the environment
env = SierraEnv()

print("Starting environment test...")

# Run for a few steps
num_steps = 100

observation, info = env.reset()

for step in range(num_steps):
    # Sample a random action
    action = env.action_space.sample()

    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Optionally render the environment
    # env.render()

    # Print step information
    print(f"Step {step + 1}: Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")

    # Check if episode finished
    if terminated or truncated:
        print("Episode finished!")
        observation, info = env.reset()

# Close the environment
env.close()

print("Environment test finished.")