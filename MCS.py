"""
@author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np

class MCSPlayer():
	def __init__(self, game):
		self.game = game
		
		self.Q  = {} # Action-value dict
		self.pi = {} # Policy
		self.epsilon = 1
		self.mu = 6 # Controls the decrease rate of epsilon
	
	def generate_episode_iterative(self):
		episode = []
		rewards = []
		curPlayer = 1
		board = self.game.getInitBoard()
		
		while self.game.getGameEnded(board, curPlayer) == 0:
			action = self.play(curPlayer*board)
			newBoard, newCurPlayer = self.game.getNextState(board, curPlayer, action)
			reward = self.game.getGameEnded(newBoard, 1)
			episode.append((curPlayer*board, action, curPlayer))
			rewards.append(reward)
			board = newBoard
			curPlayer = newCurPlayer
		
		return episode, rewards
	
	def train(self, n=1): # n is the number of training episodes
		for e in range(n):
			if e%100 == 0: print("Training... episode {}      ".format(e))
			
			# Dynamic epsilon
			self.epsilon = np.exp(-self.mu*e/n)*0.95+0.05
			
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
				a_best = np.argmax(self.Q[s][0]) # Pick action with highest return
				self.pi[s] = self.epsilon*valids/len(valids[valids==1])
				self.pi[s][a_best] += 1-self.epsilon
		print("MCS klaar")
	
	def play(self, board):
		s = self.game.stringRepresentation(board)
		valids = self.game.getValidMoves(board, 1)
		
		# Initial policy for yet unseen states
		if s not in self.pi:
			probs = valids/len(valids[valids==1])
			self.pi[s] = probs
		
		probs = self.pi[s]
		action = np.random.choice(range(len(valids)), size=1, p=probs)[0]
		
		return action
		