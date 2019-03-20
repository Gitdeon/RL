import Arena
#from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame, display
import tictactoe.TicTacToePlayers
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = TicTacToeGame(3)

# all players
rp = RandomPlayer(g).play
hp = HumanTicTacToePlayer(g).play

arena_rp_hp = Arena.Arena(rp, hp, g, display=display)
print(arena_rp_hp.playGames(20, verbose=True))