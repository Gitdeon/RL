"""
Use this script to pit any two agents against each other, or play manually with
any agent.

Author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
import py_compile as cmp
from utils import *

from Arena import *
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *

from MCS import MCSPlayer
from Qlearning import QlearningPlayer
from MCTS import MCTSPlayer

cmp.compile("othello/OthelloPlayers.py")
g = OthelloGame(4)
totalp1wins, totalp2wins, totaldraws = [], [], []

#using random seeds 8-18
for i in range(8,18):
    np.random.seed(i)
    
    #p1 = MCSPlayer(g)
    p1 = QlearningPlayer(g)
    p1.train(5000)
    
    #mcts_arguments = dotdict({'MCTSiterations': 100, 'exploration_param': 1.0})
    #mcts = MCTSPlayer(g, mcts_arguments)
    #p1 = lambda x: np.argmax(mcts.getProbability(x, temp=0))
    
    p2 = GreedyOthelloPlayer(g)
    #p2 = RandomPlayer(g)
    
    arena = Arena(p1.play, p2.play, g, display=display)
    p1wins, p2wins, draws = arena.playGames(100, verbose=False)
    totalp1wins.append(p1wins)
    totalp2wins.append(p2wins)
    totaldraws.append(draws)

print('Player 1 wins: ', np.mean(totalp1wins), ' Player 2 wins:' , np.mean(totalp2wins), ' Draws: ', np.mean(totaldraws), '\nStd. wins: ', np.std(totalp1wins), '\nStd. losses: ', np.std(totalp2wins))
    
