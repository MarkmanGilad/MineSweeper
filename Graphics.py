import pygame
import numpy as np
from State import State
# from Human_Agent import Human_Agent

MARGIN = 14
MARGIN_TOP = 454
SQUARE = 22
TOP_HEIGHT = 90
WIDTH = 692
HEIGHT = 468
ROW = 6
COL = 6
class Graphics:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        self.TopSurface = pygame.Surface((WIDTH,TOP_HEIGHT))
        self.BottomSurface = pygame.Surface((WIDTH,MARGIN_TOP))
        self.TopImagePart = pygame.image.load('Images/topPart.png')
        self.OutLinePart = pygame.image.load('Images/border.png')
        self.hidden = pygame.image.load('Images/hidden.png')
        self.zero = pygame.image.load('Images/zero.png')
        self.one = pygame.image.load('Images/one.png')
        self.two = pygame.image.load('Images/two.png')
        self.three = pygame.image.load('Images/three.png')
        self.four = pygame.image.load('Images/four.png')
        self.five = pygame.image.load('Images/five.png')
        self.six = pygame.image.load('Images/six.png')
        self.seven = pygame.image.load('Images/seven.png')
        self.eight = pygame.image.load('Images/eight.png')
        self.bomb = pygame.image.load('Images/bomb.png')
        self.bombmark = pygame.image.load('Images/bombMark.png')
        self.wrong = pygame.image.load('Images/wrong.png')
        self.end = pygame.image.load('Images/placeholder.png')
        pygame.display.set_caption('MineSweeper')


    def draw(self, state):
        self.BottomSurface.blit(self.OutLinePart, (0, 0))  # Drawing grid background
        
        for i in range(ROW):
            for j in range(COL):
                self.revealSquares(state, i, j)  # Render tiles
                
        self.screen.blit(self.BottomSurface, (MARGIN, TOP_HEIGHT + MARGIN))  # Only blit after all tiles are updated
        pygame.display.flip()

    def revealSquares(self,state,i,j):
        if state.board[i,j] == 0:
            self.BottomSurface.blit(self.hidden,(i * SQUARE,j * SQUARE)) 
        elif state.board[i,j] == -2:
             self.BottomSurface.blit(self.bomb,(i * SQUARE,  j * SQUARE))
        elif state.board[i,j] == -1:
            self.BottomSurface.blit(self.bombmark,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 1:
            self.BottomSurface.blit(self.zero,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 2:
            self.BottomSurface.blit(self.one,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 3:
            self.BottomSurface.blit(self.two,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 4:
            self.BottomSurface.blit(self.three,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 5:
            self.BottomSurface.blit(self.four,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 6:
            self.BottomSurface.blit(self.five,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 7:
            self.BottomSurface.blit(self.six,(i * SQUARE,j * SQUARE))
        elif state.board[i,j] == 8:
            self.BottomSurface.blit(self.seven,(i * SQUARE, j * SQUARE))
        elif state.board[i,j] == 9:
            self.BottomSurface.blit(self.eight,(i * SQUARE,j * SQUARE))
    
    def calc_row_col (self, pos):
        x, y = pos
        row = (y - TOP_HEIGHT- MARGIN) // SQUARE
        col = (x - MARGIN) // SQUARE
        if 0 > row or row >= ROW or 0 > col or col >= COL:
            return "OUT"
        return col, row
    def end_game(self):
        self.end.blit(self.end,(0,0))





