import time
from Graphics import *
from State import State
import numpy as np
import random

Mine_Count = 6


class Minesweeper:

    def __init__(self, world = State) -> None:
        self.world = world
        self.world.board = np.zeros((ROW,COL))
        self.state = self.initState()

    def set_init_World (self):
        self.world = State(self.initBoard())
        

    def set_init_state(self):
        self.state = self.initState()

    def initState (self):
        state = State()
        state.board = np.zeros((ROW,COL))
        return state

    def legal(self,state:State,action):
        if state.board[action] == 0:
            return True
        return False
    
    # def legal_flag(self,state:State,action):
    #     if state.board[action] in [0, -1]:
    #         return True
    #     return False
    
    def legal_action(self,state:State):
        lst = []

        for i in range(ROW):
            for j in range(COL):
                if state.board[i,j] == 0:
                    lst.append((i,j))
        return lst
    
        # rows, cols = np.where(state.board == 0)
        # return list(zip(rows, cols))

    def legat_actions_from_state_tensor(self, state):
        rows, cols = np.where(state.squeeze() == 0)
        return list(zip(rows, cols))

    def GetScore(self, state=None):
        if not state:
            state = self.state
        score = 0
        for i in range(ROW):
            for j in range(COL):
                if state.board[i,j] != 0 and state.board[i,j] != -2:
                    score += 1
                    
        return score
    
    def end_of_game (self):
        self.state = self.world
        self.state.board = self.world.board
        print("end")
        return True

    def initBoard (self):
        bombcounter = 0
        while bombcounter < Mine_Count:
            r1 = random.randint(0, ROW - 1 )
            r2 = random.randint(0, COL - 1)
            if self.world.board[r1,r2] != -2:    
                self.world.board[r1,r2] = -2
                bombcounter += 1
        for i in range(ROW):
            for j in range(COL):
                if self.world.board[i,j] != -2:
                    self.world.board[i,j] = self.initNumbersAroundBomb(i,j)
        return self.world.board
    
    

    def initOpenZeros(self, x, y):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0:
                    continue
                row = x + i
                col = y + j
                if not (0 <= row < ROW and 0 <= col < COL):
                    continue
                if self.state.board[row][col] == 0:
                    self.state.board[row][col] = self.world.board[row][col]

                    if self.world.board[row][col] == 1:
                        self.initOpenZeros(row, col)   
    ####################################                        
    def initNumbersAroundBomb(self,x,y):
        b = 1
        for i in range(-1,2):
             for j in range(-1,2):
                row = x + i
                col = y + j
                if 0 > row or row >= ROW or 0 > col or col >= COL:
                    continue
                if self.world.board[row,col] == -2:
                    b += 1
        return b
    ######################

    def move(self, action): 
        reward = 0
        x,y = action
        if self.world.board[action] == -2:     # bomb   
            return -1, True  
        
        if self.world.board[action] == 1:
            self.initOpenZeros(x,y)

        if self.state.board[action] == 0:
            self.state.board[action] = self.world.board[action]


        if self.IsBoardComplete() == True:
            return 5, True
        return self.calculateRewards(x,y), False
    
    ################

    def IsBoardComplete(self):
        a = 0
        for i in range(ROW):
            for j in range(COL):
                if self.state.board[i,j] == 0 or self.state.board[i,j] == -1:
                    a += 1
        if a == Mine_Count:
            print("GAME WON")
            return True
        return False



    # def calculateRewards(self, x,y):
    #     reward = 0.5
    #     revealed_count = 0
    #     for i in range(-1,2):
    #         for j in range(-1,2):
    #             row = x + i
    #             col = y + j
    #             if 0 > row or row >= ROW or 0 > col or col >= COL:
    #                 continue
    #             elif self.state.board[row, col] > 1:  # Checks for numbers (not hidden or empty)
    #                 revealed_count += 1  
        
    #     reward += revealed_count * 0.5

         
    #     return reward

    # def restart(self):
    #     self.world.board = np.zeros((ROW,COL))
    #     self.world = State(self.initBoard())
    #     self.state = self.initState()
    def calculateRewards(self, x, y):
        reward = 0.5
        revealed_count = 0
        safe_count = 0

        for i in range(-1, 2):
            for j in range(-1, 2):
                row = x + i
                col = y + j
                if 0 > row or row >= ROW or 0 > col or col >= COL:
                    continue
                elif self.state.board[row, col] > 1:  # Checks for numbers (not hidden or empty)
                    revealed_count += 1
                elif self.state.board[row, col] == 0:  # Checks for hidden cells
                   safe_count += 1
    
        reward += revealed_count * 0.5
        reward += safe_count * 0.1  # Additional reward for safe moves
    
        return reward
    def restart(self):
        self.world.board = np.zeros((ROW, COL))
        self.world = State(self.initBoard())
        self.state = self.initState()
    
        # Ensure the first move is safe
        # safe_cells = []
        # for i in range(ROW):
        #     for j in range(COL):
        #         if self.world.board[i, j] != -2:
        #             safe_cells.append((i, j))
        # if safe_cells:
        #     first_move = random.choice(safe_cells)
        #     self.state.board[first_move] = self.world.board[first_move]

    def RestartWithSameBoard(self):
        self.state = self.initState()

    def FirstMove(self):
        for i in range(ROW):
            for j in range(COL):
                if self.state.board[i,j] != 0:
                    return False
        return True
        
    def Flags(self,action):
        if self.state.board[action] == self.world.board[action]:
            return 
        if self.state.board[action] == -1:
            self.state.board[action] = 0
        else:
            self.state.board[action] = -1