from Minesweeper import Minesweeper
from State import State
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from DQN_Agent import DQN_Agent
import numpy as np


PATH = 'D:\MineSweeper\Tester.py'
# PATH=None
env = Minesweeper(State())
player = DQN_Agent(1, env=env, parametes_path=PATH, train=True)
# player = Random_Agent(1, env,graphics=None)

num = 1000

def main ():

    
        
    for n in range(num):
        state = State()
        while not env.end_of_game(state):
            action = player.get_action(state=state)
            state, _ = env.next_state(state,action)
        if env.end_of_game():
            score = env.GetScore()
        state =  np.zeros((16,30))
        print(n, end = "\r")
    print()
    print(score) 

