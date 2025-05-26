
import torch
import torch.nn as nn
from stable_baselines3 import PPO

class Manager(nn.Module):
    def __init__(self, input_dim, num_goals):
        super(Manager, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_goals)
        )

    def forward(self, x):
        return self.network(x)

class Worker(PPO):
    def __init__(self, policy, env, **kwargs):
        super(Worker, self).__init__(policy, env, **kwargs)
