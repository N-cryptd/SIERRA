
import optuna
import subprocess

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 2048, 8192)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    command = [
        "python",
        "scripts/train.py",
        f"--learning_rate={learning_rate}",
        f"--n_steps={n_steps}",
        f"--clip_range={clip_range}",
    ]

    process = subprocess.run(command, capture_output=True, text=True)
    
    # Extract the mean reward from the output of the training script
    mean_reward = float(process.stdout.strip().split("\n")[-1])

    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

