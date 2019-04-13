"""
@author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np
import pprint

from othello.OthelloGame import display

pp = pprint.PrettyPrinter(indent=4)

class GreedyOthelloPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		valids = self.game.getValidMoves(board, 1)
		candidates = []
		for a in range(self.game.getActionSize()):
			if valids[a]==0:
				continue
			nextBoard, _ = self.game.getNextState(board, 1, a)
			score = self.game.getScore(nextBoard, 1)
			candidates += [(-score, a)]
		candidates.sort()
		return candidates[0][1]

class MCSPlayer():
	def __init__(self, game, reward_mode):
		self.game = game
		self.reward_mode = reward_mode
		
		self.Q  = {} # Action-value dict
		self.R  = {} # Returns dict
		self.pi = {} # Policy
		self.epsilon = 0.6
	
	def generate_episode(self):
		episode = []
		rewards = []
		plays = [self.play, None, self.play]
		curPlayer = 1
		board = self.game.getInitBoard()
		
		while self.game.getGameEnded(board, curPlayer)==0:
			state = self.game.stringRepresentation(board*curPlayer)
			action = plays[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
			newBoard, curPlayer = self.game.getNextState(board, curPlayer, action)
			if self.reward_mode == "othellopieces":
				score = self.game.getScore(board, 1)
				newScore = self.game.getScore(newBoard, 1)
				reward = curPlayer*(newScore-score)
			elif self.reward_mode == "gamewon":
				reward = self.game.getGameEnded(newBoard, curPlayer)
			episode.append((state, board*curPlayer, action, curPlayer))
			rewards.append(reward)
			board = newBoard
			
			# print("newBoard=")
			# display(board)
			# print("score={}, newScore={}, reward={}\n\n".format(score, newScore, reward))
		
		return episode, rewards
	
	def train(self, n_episodes=1):
		for e in range(n_episodes):
			if e%100 == 0: print("Training... episode {}      ".format(e))
			# print("Episode {}".format(e))
			# Dynamic epsilon
			# self.epsilon = (1+e)**-0.7
			# self.epsilon = np.exp(-0.0005*e)*0.8
			self.epsilon = np.exp(-0.05*e)
			episode, rewards = self.generate_episode()
			
			for i in range(len(episode)):
				'''Note: the nature of these games disallows the occurence of the same
				state more than once per episode. Hence we can directly update pi
				after updating Q'''
				s, b, a, cp = episode[i]
				
				# Update Q
				r_sa = sum(rewards[i:]) * cp
				if s not in self.Q:
					self.Q[s] = [np.zeros(self.game.n**2+1), np.zeros(self.game.n**2+1)]
					# Assign large negative value to invalid actions
					self.Q[s][0] -= 10*np.logical_not(self.game.getValidMoves(b, 1))
				self.Q[s][1][a] += 1 # Number of times (s, a) has been encountered
				self.Q[s][0][a] += 1/self.Q[s][1][a] * (r_sa - self.Q[s][0][a])
				
				# Update pi
				Q_s = self.Q[s]
				# print(s)
				# print(Q_s)
				a_best = np.argmax(Q_s[0]) # Pick action with highest return
				# print(a_best)
				# print("\n")
				valids = self.game.getValidMoves(b, 1)
				v_a = [ai for ai in range(len(valids)) if valids[ai]==1]	# v_a is A(s)
				v_l = len(v_a)												# v_l is |A(s)|
				for a_ in v_a: self.pi[s][a_] = self.epsilon/v_l
				self.pi[s][a_best] += 1-self.epsilon
			# pp.pprint(self.pi)
			# wait = input()
	
	def play(self, board):
		s = self.game.stringRepresentation(board)
		valids = self.game.getValidMoves(board, 1)
		
		# Initial policy for yet unseen states
		if s not in self.pi:
			# print("Nog niet gezien:\n{}".format(s))
			probs = valids/len(valids[valids==1])
			self.pi[s] = probs
		
		probs = self.pi[s]
		action = np.random.choice(range(len(valids)), size=1, p=probs)[0]
		
		return action
		