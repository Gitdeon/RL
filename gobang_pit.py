#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:40:18 2019

@author: Gideon
"""

import Arena
#from MCTS import MCTS
from gobang.GobangGame import GobangGame, display
from gobang.GobangPlayers import *
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = GobangGame(9)

# all players
rp = RandomPlayer(g).play
hp = HumanGobangPlayer(g).play

arena_rp_hp = Arena.Arena(rp, hp, g, display=display)
print(arena_rp_hp.playGames(2, verbose=True))
