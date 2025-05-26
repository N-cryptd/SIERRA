
import subprocess
import json

CURRICULUM = [
    {
        "name": "Stage 1: Basic Survival",
        "config": {
            "max_threats": 0,
            "crafting": False,
        },
        "timesteps": 10000,
    },
    {
        "name": "Stage 2: Tool Use & Crafting",
        "config": {
            "max_threats": 0,
            "crafting": True,
        },
        "timesteps": 20000,
    },
    {
        "name": "Stage 3: Hazard Avoidance",
        "config": {
            "max_threats": 2,
            "crafting": True,
        },
        "timesteps": 30000,
    },
]

def run_curriculum():
    for i, stage in enumerate(CURRICULUM):
        print(f"--- Starting {stage['name']} ---")

        config_path = f"/tmp/curriculum_config_{i}.json"
        with open(config_path, "w") as f:
            json.dump(stage["config"], f)

        command = [
            "python",
            "scripts/train.py",
            f"--total_timesteps={stage['timesteps']}",
            f"--config_path={config_path}",
        ]

        if i > 0:
            command.append(f"--load_path=./trained_model_stage_{i-1}.zip")

        subprocess.run(command)

        print(f"--- Completed {stage['name']} ---")

if __name__ == "__main__":
    run_curriculum()
