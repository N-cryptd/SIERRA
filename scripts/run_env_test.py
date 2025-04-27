import gymnasium as gym
from sierra.environment.core import SierraEnv

# Instantiate the environment
env = SierraEnv()

print("Starting environment test...")

# Run for a few steps
num_steps = 500

observation, info = env.reset()
print(f"Initial Observation: {observation}")

for step in range(num_steps):
    # Sample a random action
    action = env.action_space.sample()

    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Optionally render the environment
    # env.render()

    # Print step information
    print(f"Step {step + 1}: Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")
    # Print detailed observation information
    if isinstance(observation, dict):
        agent_obs = observation.get('agent')
        resource_obs = observation.get('resource') # Use 'resource' key

        if agent_obs is not None:
            print(f"  Agent Position: {agent_obs[:2]}") # Assuming first two elements are position
            if len(agent_obs) > 2: # Check if hunger/thirst are included
                print(f"  Agent Hunger: {agent_obs[2]}")
            if len(agent_obs) > 3: # Check if thirst is included
                print(f"  Agent Thirst: {agent_obs[3]}")

        if resource_obs is not None:
            # Check if it's a single resource or multiple
            if resource_obs.ndim == 1: # Single resource (e.g., array([x, y]))
                 print(f"  Resource Location: {resource_obs}")
            else: # Multiple resources (e.g., array([[x1, y1], [x2, y2]]))
                 print(f"  Resource Locations: {resource_obs}")

        # Print dynamic environment states
        time_of_day = observation.get('time_of_day')
        current_weather = observation.get('current_weather')
        current_season = observation.get('current_season')

        if time_of_day is not None:
            print(f"  Time of Day: {time_of_day}")
        if current_weather is not None:
            print(f"  Current Weather: {current_weather}")
        if current_season is not None:
            print(f"  Current Season: {current_season}")
    else:
        print(f"  Observation: {observation}")

    # Check if episode finished
    if terminated or truncated:
        print("Episode finished!")
        observation, info = env.reset()

# Close the environment
env.close()

print("Environment test finished.")