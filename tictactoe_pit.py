#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
from utils import *
import py_compile as cmp
import pickle as pc

from Arena import *
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
from MCS import MCSPlayer
from MCTS import MCTSPlayer
from Qlearning import QlearningPlayer

cmp.compile("tictactoe/TicTacToePlayers.py")
"""
Use this script to play any two agents against each other, or play manually with
any agent.
"""
g = TicTacToeGame()

# All players
# p1 = RandomPlayer(g)
# p1 = QlearningPlayer(g, "gamewon")
# p1 = MCSPlayer(g, "gamewon")
# p1.train(10000)

#MCTS:
mcts_arguments = dotdict({'MCTSiterations': 25, 'exploration_param': 1.0})
mcts = MCTSPlayer(g, mcts_arguments)
p1 = lambda x: np.argmax(mcts.getActionProb(x, temp=0))


# Save player so we can reuse it without training
# pc.dump(p1, open("mcs_tictactoe.rlp", "w"))

p2 = RandomPlayer(g)

arena_rp1_rp2 = Arena(p1, p2.play, g, display=display)
# print(arena_rp1_rp2.playGame(verbose=True))
print(arena_rp1_rp2.playGames(100, verbose=False))
