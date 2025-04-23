import pygame
import torch
from Graphics import *
from Minesweeper import Minesweeper
from DQN_Agent_CNN import DQN_Agent
from ReplayBuffer import *
import os
import wandb
import random


def main (chkpt):

    pygame.init()
    graphics = Graphics()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Minesweeper')
    # clock = pygame.time.Clock()


    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    TopSurface = pygame.Surface((WIDTH,TOP_HEIGHT))
    BottomSurface = pygame.Surface((WIDTH,MARGIN_TOP))
    TopImagePart = pygame.image.load('Images/topPart.png')
    OutLinePart = pygame.image.load('Images/border.png')
    hidden = pygame.image.load('Images/hidden.png')
    zero = pygame.image.load('Images/zero.png')
    one = pygame.image.load('Images/one.png')
    two = pygame.image.load('Images/two.png')
    three = pygame.image.load('Images/three.png')
    four = pygame.image.load('Images/four.png')
    five = pygame.image.load('Images/five.png')
    six = pygame.image.load('Images/six.png')
    seven = pygame.image.load('Images/seven.png')
    eight = pygame.image.load('Images/eight.png')
    bomb = pygame.image.load('Images/bomb.png')
    bombmark = pygame.image.load('Images/bombMark.png')
    wrong = pygame.image.load('Images/wrong.png')
    end = pygame.image.load('Images/placeholder.png')
    pygame.display.set_caption('MineSweeper')
    env = Minesweeper()

    num = chkpt
    best_score = 0
    best_step = 0
    ####### params ############
    player = DQN_Agent()
    player_hat = DQN_Agent()
    player_hat.DQN = player.DQN.copy()
    batch_size = 50
    buffer = ReplayBuffer(path=None)
    learning_rate = 1e-4
    ephocs = 100000
    start_epoch = 0
    C = 50
    loss = torch.tensor([0])
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[5000*1000, 10000*1000, 15000*1000], gamma=0.5)
    wandb.init(
        project= "Minesweeper",
        id = f"RunNumber{num}",

        config={
            "name": f"Minesweeper{num}",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "scheduler": scheduler,
            "C": C,
        }
    )
    

    #region ######## checkpoint Load ############
    checkpoint_path = f"Data/checkpoint{num}.pth"
    buffer_path = f"Data/buffer{num}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player.DQN.load_state_dict(checkpoint['model_state_dict'])
        player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        buffer = torch.load(buffer_path)
        losses = checkpoint['loss']
        scores = checkpoint['scores']
        avg_score = checkpoint['avg_score']
    player.DQN.train()
    player_hat.DQN.eval()
    # endregion
    #################################

    for epoch in range(start_epoch, ephocs):
        # if epoch == 0:
        #     env.restart()
        # else:
        #     env.RestartWithSameBoard()
        env.restart()
        end_of_game = False
        state = env.state.copy()
        step = 0
        while not end_of_game:
            print (step, end='\r')
            step += 1

            events = pygame.event.get()
            pygame.event.pump()
            for event in events:
                if event.type == pygame.QUIT:
                    return
            
            ############## Sample Environement #########################
            action = player.get_action(state=env.state, epoch=epoch, train=True)
            reward, done = env.move(action=action)
            
            next_state = env.state.copy()
            buffer.push(state.toTensor(), torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                        next_state.toTensor(), torch.tensor(done, dtype=torch.float32))
            if done:
                best_score = max(best_score, env.GetScore())
                best_step = max(best_step, step)
                break

            state = next_state

            i,j = action
            BottomSurface.blit(OutLinePart,(0,0))
            screen.blit(BottomSurface,(0,90))
            TopSurface.blit(TopImagePart,(0,0))
            screen.blit(TopSurface,(0,0))
            graphics.draw(state)
            graphics.revealSquares(state,i,j)
            pygame.display.update()
            # clock.tick(FPS)

            if len(buffer) < 100:
                continue
            ############## Train ################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player.get_Q_Values(states, actions)                 # DDQN
            _, Q_hat_Values = player_hat.get_actions_Values(next_states)

            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())
        
        #########################################
        score = env.GetScore()
        wandb.log({"loss": loss,  "score": score, "step": step })
        print (f'epoch: {epoch} loss: {loss.item():.7f} step: {step} best_steps: {best_step} ' \
               f'score: {score} best_score: {best_score}')
        step = 0
        if epoch % 10 == 0:
            scores.append(score)
            losses.append(loss.item())

        avg = (avg * (epoch % 10) + score) / (epoch % 10 + 1)
        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            print (f'average score last 10 games: {avg} ')
            avg = 0

        if epoch % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.DQN.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses,
                'scores':scores,
                'avg_score': avg_score
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)

if __name__ == "__main__":
    if not os.path.exists("Data/checkpoit_num"):
        torch.save(101, "Data/checkpoit_num")    
    
    chkpt = torch.load("Data/checkpoit_num", weights_only=False)
    chkpt += 1
    torch.save(chkpt, "Data/checkpoit_num")    
    main (chkpt)
    
    
  