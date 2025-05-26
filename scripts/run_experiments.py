import subprocess
import pandas as pd

EXPERIMENTS = {
    "baseline": {
        "description": "A baseline PPO agent without any enhancements.",
        "script": "scripts/train.py",
        "args": {
            "--total_timesteps": "10000",
        }
    },
    "icm": {
        "description": "PPO agent with Intrinsic Curiosity Module (ICM).",
        "script": "scripts/train_icm_agent.py",
        "args": {
            "--total_timesteps": "10000",
        }
    },
    "curriculum": {
        "description": "PPO agent trained with a curriculum learning strategy.",
        "script": "scripts/run_curriculum.py",
        "args": {}
    },
    "lstm": {
        "description": "Recurrent PPO agent with LSTM policy.",
        "script": "scripts/train_lstm_agent.py",
        "args": {
            "--total_timesteps": "10000",
        }
    },
}

def run_experiments():
    results = []

    for name, experiment in EXPERIMENTS.items():
        print(f"--- Running Experiment: {name} ---")
        print(experiment["description"])

        command = ["python", experiment["script"]]
        for arg, value in experiment["args"].items():
            command.append(f"{arg}={value}")

        process = subprocess.run(command, capture_output=True, text=True)
        mean_reward = float(process.stdout.strip().split("\n")[-1])

        results.append({
            "name": name,
            "description": experiment["description"],
            "mean_reward": mean_reward,
        })

        print(f"--- Completed Experiment: {name} ---")

    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    print("\nExperiment results saved to experiment_results.csv")

    # Generate report
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["name"], df["mean_reward"])
    ax.set_ylabel("Mean Reward")
    ax.set_title("Experiment Results")

    with PdfPages("experiment_report.pdf") as pdf:
        pdf.savefig(fig)

    print("Experiment report saved to experiment_report.pdf")

if __name__ == "__main__":
    run_experiments()
