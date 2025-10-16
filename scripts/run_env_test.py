"""Smoke test script for the SIERRA environment."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on the Python path when executed directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sierra.environment.core import SierraEnv


def main(num_steps: int = 100) -> None:
    env = SierraEnv()
    observation, _ = env.reset()

    print("Starting environment test...")
    print(f"Initial Observation Keys: {list(observation.keys())}")

    for step in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step + 1}: Action={action}, Reward={reward:.3f}, "
            f"Terminated={terminated}, Truncated={truncated}, NewCell={info.get('new_cell_visited')}"
        )

        if terminated or truncated:
            print("Episode finished, resetting environment...")
            observation, _ = env.reset()

    env.close()
    print("Environment test finished.")


if __name__ == "__main__":
    main()
