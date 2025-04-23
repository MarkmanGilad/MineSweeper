import math
import random
from typing import Any
import torch
import torch.nn as nn
import numpy as np
from DQN_CNN import DQN
from State import State
from Minesweeper import Minesweeper
from Graphics import *

env = Minesweeper(State())
# epsilon Greedy
epsilon_start = 1
epsilon_final = 0.01
epsiln_decay = 1000

# epochs = 1000
# batch_size = 64
gamma = 0.99 
MSELoss = nn.MSELoss()

class DQN_Agent:
    def __init__(self, parametes_path = None, train = False, env= Minesweeper, graphics: Graphics = None) -> None:
        self.DQN = DQN(1, ROW, COL)
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train = train
        self.env = env

    def train (self, train):
          self.train = train
          if train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_state_action(self, state, action):
        state_action = state.board.copy()
        state_action[action] = -1
        return torch.from_numpy(state_action).to(dtype=torch.float32).unsqueeze(0)

    def get_state_action_from_tensor(self, state, action):
        state_action = state.squeeze()
        state_action[action] = -1
        return state_action.unsqueeze(0)

    def get_all_state_actions (self, state, actions):
        state_actions = []
        for action in actions:
            state_actions.append(self.get_state_action(state, action))
        return torch.stack(state_actions)

    def get_all_state_actions_from_tensor (self, state, actions):
        state_actions = []
        for action in actions:
            state_actions.append(self.get_state_action_from_tensor(state.clone(), action))
        return torch.stack(state_actions)

    def get_action (self, state = State(), epoch = 0, events= None, train = False):
        
        actions = env.legal_action(state) 

        epsilon = self.epsilon_greedy(epoch)
        rnd = random.random()
        if train and rnd < epsilon:
            return random.choice(actions)
        
        states_tensor = self.get_all_state_actions(state, actions)
                
        with torch.no_grad():
            Q_values = self.DQN(states_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Q_Values (self, states, actions):
        rows = actions[:, 0]  # shape: [B]
        cols = actions[:, 1]  # shape: [B]
        states[torch.arange(actions.shape[0]), 0, rows, cols] = -1
        Q_values = self.DQN(states)
        return Q_values

    def get_actions_Values (self, states):
        actions_list = []
        values_list = []
        for state in states:
            actions = env.legat_actions_from_state_tensor(state)
            states_tensor = self.get_all_state_actions_from_tensor(state, actions)
                
            with torch.no_grad():
                Q_values = self.DQN(states_tensor).squeeze()
            max_value, max_index  = torch.max(Q_values, dim=0)
            actions_list.append(actions[max_index])
            values_list.append(max_value)
        
        actions_tensor = torch.tensor(actions_list)
        value_tensor = torch.stack(values_list).unsqueeze(1)
        
        return actions_tensor, value_tensor


    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        res = max(final, start - (start - final) * (epoch / decay))
        return res
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None) -> Any:
        return self.get_action(state)