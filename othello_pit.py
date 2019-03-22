#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:40:18 2019

@author: Gideon
"""

import Arena
#from MCTS import MCTS
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
import numpy as np
from utils import *
import py_compile as cmp

cmp.compile("othello/OthelloPlayers.py")
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

arena_rp_rp2 = Arena.Arena(rp, rp2, g, display=display)
print(arena_rp_rp2.playGame(verbose=True))
