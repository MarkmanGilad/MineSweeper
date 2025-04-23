import time
from Graphics import *
from Minesweeper import Minesweeper

class Human_Agent:
    def __init__(self, env, graphics):
        self.env = env
        self.graphics = graphics


    def get_action (self, events, ):
        
        for event in events:
            
            if pygame.mouse.get_pressed()[0]:
                time.sleep(0.05)
                pos = event.pos
                action = self.graphics.calc_row_col(pos)
                if action == "OUT":
                    return None
                elif self.env.legal(self.env.state, action):
                    return action
            elif pygame.mouse.get_pressed()[2]:
                time.sleep(0.05)
                pos = event.pos
                action = self.graphics.calc_row_col(pos)
                self.env.Flags(action)
                

        return None
    
    def __call__(self, events, state = None):
        return self.get_action(events)
