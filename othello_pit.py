"""
Use this script to pit any two agents against each other, or play manually with
any agent.

Author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
import py_compile as cmp
from utils import *

from Arena import *
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *

from MCS import MCSPlayer
from Qlearning import QlearningPlayer
#from MCTS import MCTSPlayer

cmp.compile("othello/OthelloPlayers.py")

g = OthelloGame(4)

# p1 = MCSPlayer(g)
p1 = QlearningPlayer(g)
p1.train(10000)

p2 = GreedyOthelloPlayer(g)
# p2 = RandomPlayer(g)

print("Othello")
arena = Arena(p1.play, p2.play, g, display=display)
print(arena.playGames(1000, verbose=False))