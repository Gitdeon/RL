"""
Use this script to pit any two agents against each other, or play manually with
any agent.

Author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
import py_compile as cmp
from utils import *

from Arena import *
from gobang.GobangGame import GobangGame, display
from gobang.GobangPlayers import *

from MCS import MCSPlayer
from Qlearning import QlearningPlayer
#from MCTS import MCTSPlayer

cmp.compile("gobang/GobangPlayers.py")

g = GobangGame(9)

# p1 = MCSPlayer(g)
p1 = QlearningPlayer(g)
p1.train(10000)

p2 = RandomPlayer(g)


print("Gobang")
arena = Arena(p1.play, p2.play, g, display=display)
print(arena.playGames(1000, verbose=False))