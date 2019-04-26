"""
Use this script to pit any two agents against each other, or play manually with
any agent.

Author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
import py_compile as cmp
from utils import *

from Arena import *
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *

from MCS import MCSPlayer
from Qlearning import QlearningPlayer
#from MCTS import MCTSPlayer

cmp.compile("connect4/Connect4Players.py")

g = Connect4Game(6)

# p1 = MCSPlayer(g)
p1 = QlearningPlayer(g)
p1.train(60000)

p2 = RandomPlayer(g)
# p2 = OneStepLookaheadConnect4Player(g)

print("Connect 4")
arena = Arena(p1.play, p2.play, g, display=display)
print(arena.playGames(1000, verbose=False))