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

cmp.compile("tictactoe/TicTacToePlayers.py")
"""
Use this script to play any two agents against each other, or play manually with
any agent.
"""
g = TicTacToeGame()

# All players
mcsp1 = MCSPlayer(g, "gamewon")
mcsp1.train(100000)
# mcsp1 = RandomPlayer(g)

# Save player so we can reuse it without training
# pc.dump(mcsp1, open("mcs_tictactoe.rlp", "w"))

p2 = RandomPlayer(g)

arena_rp1_rp2 = Arena(mcsp1.play, p2.play, g, display=display)
# print(arena_rp1_rp2.playGame(verbose=True))
print(arena_rp1_rp2.playGames(1000, verbose=False))