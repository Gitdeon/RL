import numpy as np

class MonteCarloPlayer():
   def __init__(self, game):
      self.game = game

   def play(self, board):
      valids = self.game.getValidMoves(board, 1)
      

