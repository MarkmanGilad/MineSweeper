import random
from Graphics import *
import numpy as np
import torch
import torch.nn as nn

ROW = 6
COL = 6
#i = r1 = 16 j = r2 = 30
class State:
    def __init__(self, state = np.zeros((ROW,COL),dtype=np.float32)) -> None:
        self.board = state

    # def toTensor(self):
    #     board_tensor = torch.tensor(self.board, dtype=torch.float32)
    #     feature_tensor = torch.zeros((ROW, COL, 3))  # Additional features
    
    #     for i in range(ROW):
    #         for j in range(COL):
    #             if self.board[i, j] == 0:  # Hidden cell
    #                 feature_tensor[i, j, 0] = self.count_hidden_around(i, j)
    #                 feature_tensor[i, j, 1] = self.calc_mine_probability(i, j)
    #             elif self.board[i, j] > 1:  # Revealed cell
    #                 feature_tensor[i, j, 2] = self.board[i, j]
    
    #     return torch.cat([board_tensor.flatten(), feature_tensor.flatten()])
    ################################################
    def count_hidden_around(self, i, j):
        count = 0
        for x in range(-1, 2):  # Check rows i-1, i, i+1
            for y in range(-1, 2):  # Check columns j-1, j, j+1
                if x == 0 and y == 0:  # Skip the current cell
                    continue
                row = i + x
                col = j + y
                if 0 <= row < ROW and 0 <= col < COL:  # Check if within bounds
                    if self.board[row, col] == 0:  # Check if cell is hidden
                        count += 1
        return count
    
###############################################################
    def calc_mine_probability(self, i, j):
        if self.board[i, j] != 0:  
            return 0.0
    
        total_mines = 0
        total_hidden = 0
    
    # Count total mines and hidden cells in adjacent cells
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:  
                    continue
                row = i + x
                col = j + y
                if 0 <= row < ROW and 0 <= col < COL:  # Check if within bounds
                    if self.board[row, col] == -2:  # Adjacent mine
                       total_mines += 1
                    elif self.board[row, col] == 0:  # Adjacent hidden cell
                        total_hidden += 1
    
        if total_hidden == 0:
            return 0.0  # No hidden cells, so probability is 0
    
        return total_mines / total_hidden
    

    def toState(self, state  ,device = torch.device('cpu')) -> tuple:
        return np.array(state.reshape((ROW,COL)))
    
    def copy(self):
        return State(self.board.copy())

    def toTensor (self):
        state_tensor = torch.from_numpy(self.board).to(dtype=torch.float32)
        normalized_state = state_tensor / 10
        return normalized_state.unsqueeze(0)
