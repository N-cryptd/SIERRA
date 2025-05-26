
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ICM(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ICM, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(128 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, state, next_state, action):
        state_feat = self.feature_extractor(state)
        next_state_feat = self.feature_extractor(next_state)

        # Inverse model
        action_logits = self.inverse_model(torch.cat((state_feat, next_state_feat), dim=1))
        predicted_action = Categorical(logits=action_logits)

        # Forward model
        action_one_hot = nn.functional.one_hot(action, num_classes=self.inverse_model[-1].out_features).float()
        predicted_next_state_feat = self.forward_model(torch.cat((state_feat, action_one_hot), dim=1))

        return predicted_next_state_feat, next_state_feat, predicted_action
