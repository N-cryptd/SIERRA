import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Assuming SierraEnv is located at sierra.environment.core
# You might need to adjust the import path based on your project structure
try:
    from sierra.environment.core import SierraEnv
except ImportError:
    print("Error: Could not import SierraEnv. Make sure the sierra package is in your PYTHONPATH.")
    exit()

def evaluate_model(model_path, num_episodes=100):
    """
    Evaluates a trained Stable Baselines3 model on the Sierra environment.

    Args:
        model_path (str): Path to the trained model file.
        num_episodes (int): Number of episodes to run for evaluation.
    """
    # Create the environment
    # Use DummyVecEnv for simplicity, switch to SubprocVecEnv for parallel evaluation
    env = DummyVecEnv([lambda: SierraEnv()])

    print(f"Loading model from {model_path}")
    try:
        if "ppo" in model_path.lower():
            model = PPO.load(model_path, env=env)
        elif "dqn" in model_path.lower():
            from stable_baselines3 import DQN
            model = DQN.load(model_path, env=env)
        else:
            print(f"Error: Unknown model type in path: {model_path}")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    episode_rewards = []
    episode_lengths = []

    print(f"Starting evaluation for {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] # Assuming reward is a list/array from VecEnv
            episode_length += 1
            if isinstance(done, np.ndarray):
                 done = done[0] # Assuming done is a list/array from VecEnv


        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{num_episodes} finished with reward: {total_reward:.2f} and length: {episode_length}")


    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print("\n--- Evaluation Results ---")
    print(f"Mean Episode Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print("------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the SIERRA environment.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (.zip).")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run for evaluation.")

    args = parser.parse_args()

    evaluate_model(args.model_path, args.num_episodes)