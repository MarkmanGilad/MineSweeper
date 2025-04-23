import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Graphics import *

# Parameters
input_size = ROW * COL   # the action is -1 in the state itself.
output_size = 1 # Value of square
gamma = 0.98
 

class DQN (nn.Module):
    def __init__(self, input_channels: int, row: int, col: int, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 10x10 → 10x10
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 10x10 → 10x10
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # 10x10 → 10x10
        self.MSELoss = nn.MSELoss()

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, row, col)
            dummy = self.conv1(dummy)
            dummy = F.relu(dummy)
            dummy = self.conv2(dummy)
            dummy = F.relu(dummy)
            self.flattened_size = dummy.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.output = nn.Linear(128, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))         # [B, 32, H, W]
        x = F.relu(self.conv2(x))         # [B, 64, H, W]
        x = x.view(x.size(0), -1)         # Flatten to [B, *]
        x = F.relu(self.fc1(x))           # [B, 128]
        return self.output(x)             # [B, 1]
    
    def loss (self, Q_values, rewards, Q_next_Values, dones ):
        Q_new = rewards.to(self.device) + gamma * Q_next_Values * (1- dones.to(self.device))
        return self.MSELoss(Q_values, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states):
        return self.forward(states).to(self.device)