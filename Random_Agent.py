import random
import time
from Graphics import *
from Minesweeper import Minesweeper


class Random_Agent:
    def __init__(self, env, graphics):
        self.env: Minesweeper = env
        self.graphics = graphics


    def get_action (self, events, state):
        for event in events:
            r1 = random.randint(0, ROW)
            r2 = random.randint(0, COL)
            action = r1,r2
            #time.sleep(0.05)
            if self.env.legal(state, action):
                return action

        return None
    
    def __call__(self, events):
        return self.get_action(events,self.env)
