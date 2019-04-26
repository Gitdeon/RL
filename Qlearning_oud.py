"""
@author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np

class QlearningPlayer():
	def __init__(self, game, reward_mode="gamewon"):
		self.game = game
		self.reward_mode = reward_mode
		
		self.Q  = {} # Action-value dict
		self.pi = {} # Policy
		self.alpha = 0.2
		self.gamma = 0.9
		self.epsilon = 1
	
	def generate_episode_iterative(self):
		episode = []
		rewards = []
		
		return episode, rewards
	
	def train(self, n_episodes=1):
		for e in range(n_episodes):
			if e%100 == 0: print("Training... episode {}      ".format(e))
			
			# Dynamic epsilon
			self.epsilon = np.exp(-0.0010*e)*0.95+0.05
			
			# Work way down the episode
			curPlayer = 1
			board = self.game.getInitBoard()			
			while self.game.getGameEnded(board, curPlayer) == 0:
				a = self.play(curPlayer*board)
				newBoard, newCurPlayer = self.game.getNextState(board, curPlayer, a)				
				valids = self.game.getValidMoves(curPlayer*board, 1)
				s = self.game.stringRepresentation(curPlayer*board)
				
				# Let a fixed-policy (random) opponent play to create s' (denoted s_)
				newA = self.play_random(-curPlayer*newBoard)
				newNewBoard, _ = self.game.getNextState(newBoard, -curPlayer, newA)
				newNewValids = self.game.getValidMoves(curPlayer*newNewBoard, 1)
				s_ = self.game.stringRepresentation(curPlayer*newNewBoard)
				
				
				reward = self.game.getGameEnded(newNewBoard, 1)
				
				# Update Q
				r_sa = curPlayer*reward
				if s not in self.Q:
					self.Q[s ]  = np.zeros(len(valids))
					self.Q[s ] -= 100*np.logical_not(valids)
				if s_ not in self.Q:
					self.Q[s_]  = np.zeros(len(newNewValids))
					self.Q[s_] -= 100*np.logical_not(newNewValids)
				
				self.Q[s][a] += self.alpha*(r_sa+self.gamma*np.max(self.Q[s_])-self.Q[s][a])
				
				# Update pi
				a_best = np.argmax(self.Q[s]) # Pick action with highest return
				self.pi[s] = self.epsilon*valids/len(valids[valids==1])
				self.pi[s][a_best] += 1-self.epsilon
				
				board = newBoard
				curPlayer = newCurPlayer
		print("Qlearning klaar")
	
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
	
	def play_random(self, board):
		valids = self.game.getValidMoves(board, 1)
		probs = valids/len(valids[valids==1])
		
		action = np.random.choice(range(len(valids)), size=1, p=probs)[0]
		
		return action
		