import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Graphics import *

# Parameters
input_size = (ROW * COL) + (ROW * COL * 3) + 2# Board ROW*COL + action (x,y)  
layer1 = 512
layer2 = 256
layer3 = 128
layer4 = 64

output_size = 1# Value of square
gamma = 0.98
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1) 
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer2, layer3)
        self.linear4 = nn.Linear(layer3, layer4)
        # self.linear5 = nn.Linear(layer4, layer5)
        self.output = nn.Linear(layer4, output_size)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        x = self.linear4(x)
        x = F.leaky_relu(x)
        # x = self.linear5(x)
        # x = F.leaky_relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_values, rewards, Q_next_Values, dones ):
        Q_new = rewards.to(self.device) + gamma * Q_next_Values * (1- dones.to(self.device))
        return self.MSELoss(Q_values, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states, actions):
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action).to(self.device)