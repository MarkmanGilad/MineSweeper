import time
import pygame
from Random_Agent import Random_Agent
from Graphics import Graphics
from State import State
from Human_Agent import Human_Agent
from Minesweeper import Minesweeper
from DQN_Agent import DQN_Agent

pygame.init()
graphics = Graphics()
env = Minesweeper()
env.set_init_World()
env.set_init_state()
dqn = DQN_Agent(env=env,graphics=graphics)
human = Human_Agent(env, graphics)
random = Random_Agent(env,graphics)


def main():
    player = human
    run = True

    
    while run:
        pygame.event.pump()
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False
        action = player(events=events, state=env.state)
        if action:
            _,cont = env.move(action)
            if cont :
                graphics.draw(env.state)  
                env.restart()
                
              
        graphics.draw(env.state)      
        pygame.display.flip()
        pygame.time.Clock().tick(60)
    pygame.quit()        

if __name__== '__main__':
    main()
