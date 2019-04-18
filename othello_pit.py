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
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from MCS import MCSPlayer

cmp.compile("othello/OthelloPlayers.py")
"""
Use this script to play any two agents against each other, or play manually with
any agent.
"""
g = OthelloGame(4)

# All players
mcsp1 = MCSPlayer(g, "othellopieces")
mcsp1.train(10000)

# Save player so we can reuse it without training
# pc.dump(mcsp1, open("mcs_othello.rlp", "w"))

# p2 = GreedyOthelloPlayer(g)
p2 = RandomPlayer(g)

arena_rp1_rp2 = Arena(mcsp1.play, p2.play, g, display=display)
print(arena_rp1_rp2.playGames(1000, verbose=False))