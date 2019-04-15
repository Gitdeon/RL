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
	def __init__(self, game, reward_mode, two_policies=True):
		self.game = game
		self.reward_mode = reward_mode
		
		self.Q  = {} # Action-value dict
		self.R  = {} # Returns dict
		self.pi = {} # Policy
		self.epsilon = 0.6
		self.two_policies = two_policies
	
	def generate_episode_iterative(self):
		episode = []
		rewards = []
		plays = [self.play, None, self.play]
		curPlayer = 1
		board = self.game.getInitBoard()
		
		while self.game.getGameEnded(board, curPlayer) == 0:
			action = plays[curPlayer+1](curPlayer*board)
			newBoard, newCurPlayer = self.game.getNextState(board, curPlayer, action)
			if self.reward_mode == "othellopieces":
				score = self.game.getScore(board, 1)
				newScore = self.game.getScore(newBoard, 1)
				reward = newScore-score
			elif self.reward_mode == "gamewon":
				reward = self.game.getGameEnded(newBoard, 1)
			episode.append((curPlayer*board, action, curPlayer))
			rewards.append(reward)
			board = newBoard
			curPlayer = newCurPlayer
		
		return episode, rewards
	
	def train(self, n_episodes=1):
		for e in range(n_episodes):
			if e%100 == 0: print("Training... episode {}      ".format(e))
			# print("Episode {}".format(e))
			
			# Dynamic epsilon
			self.epsilon = np.exp(-0.0005*e)*0.8
			# self.epsilon = np.exp(-0.05*e)
			
			episode, rewards = self.generate_episode_iterative()
			
			for i in range(len(episode)):
				'''Note: the nature of these games disallows the occurence of the same
				state more than once per episode. Hence we can directly update pi
				after updating Q'''
				b, a, cp = episode[i]
				valids = self.game.getValidMoves(b, 1)
				s = self.game.stringRepresentation(b)
				
				# Update Q
				r_sa = sum(rewards[i:])*cp
				if s not in self.Q:
					self.Q[s] = [np.zeros(len(valids)), np.zeros(len(valids))]
					# Assign large negative value to invalid actions
					self.Q[s][0] -= 100*np.logical_not(valids)
				
				self.Q[s][1][a] += 1 # Number of times (s, a) has been encountered
				self.Q[s][0][a] += 1/self.Q[s][1][a]*(r_sa-self.Q[s][0][a])
				
				# Update pi
				Q_s = self.Q[s]
				a_best = np.argmax(Q_s[0]) # Pick action with highest return
				self.pi[s] = self.epsilon*valids/len(valids[valids==1])
				self.pi[s][a_best] += 1-self.epsilon
				# if a_best == len(valids)-1:
					# print("\nGepast")
				# print(b*cp)
				# print(cp)
				# print(Q_s[0])
	
	def play(self, board):
		s = self.game.stringRepresentation(board)
		valids = self.game.getValidMoves(board, 1)
		
		# Initial policy for yet unseen states
		if s not in self.pi:
			probs = valids/len(valids[valids==1])
			self.pi[s] = probs
		
		probs = self.pi[s]
		action = np.random.choice(range(len(valids)), size=1, p=probs)[0]
		if not(valids[action]==1):
			# print(board)
			# print(action)
			# pp.pprint(self.pi[s])
			raise ValueError("Valsgespeeld!")
		
		return action
		