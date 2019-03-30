#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
from utils import *
import py_compile as cmp

from Arena import *
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from MCS import MCSPlayer

cmp.compile("othello/OthelloPlayers.py")
"""
Use this script to play any two agents against each other, or play manually with
any agent.
"""
g = OthelloGame(8)

# all players
mcsp1 = MCSPlayer(g)
mcsp1.train(1000)

p2 = GreedyOthelloPlayer(g)

arena_rp1_rp2 = Arena(mcsp1.play, p2.play, g, display=display)
print(arena_rp1_rp2.playGame(verbose=True))