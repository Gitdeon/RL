import Arena
#from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = Connect4Game(6)

# all players
rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
hp = HumanConnect4Player(g).play

op=OneStepLookaheadConnect4Player(g).play

arena_rp_op = Arena.Arena(rp, op, g, display=display)
print(arena_rp_op.playGames(2, verbose=True))

#arena_rp_hp = Arena.Arena(rp, hp, g, display=display)
#print(arena_rp_hp.playGames(2, verbose=True))


