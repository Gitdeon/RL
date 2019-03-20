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

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
hp = HumanOthelloPlayer(g).play

arena_rp_hp = Arena.Arena(rp, hp, g, display=display)
print(arena_rp_hp.playGames(2, verbose=True))