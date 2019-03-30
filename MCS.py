"""
@author: Gideon Hanse, Dyon van Vreumingen
"""

import numpy as np

from othello.OthelloGame import display

class MCPlayer_oud():
	def __init__(self, game):
		self.game = game
		 
	def play(self, board):
		best_outcome = 0
		valids = self.game.getValidMoves(board, 1)
		for i in range(len(valids)):
			outcome = Arena.playGames(valids[i], 10);
			if outcome > best_outcome:
				best_outcome = outcome
				best_move = valids[i]
		
		return best_move

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
	def __init__(self, game):
		self.game = game
		
		self.Q  = {} # Action-value dict
		self.R  = {} # Returns dict
		self.pi = {} # Policy
		self.epsilon = 1
	
	def generate_episode(self):
		episode = []
		rewards = []
		plays = [self.play, None, self.play]
		curPlayer = 1
		board = self.game.getInitBoard()
		
		while self.game.getGameEnded(board, curPlayer)==0:
			state = self.game.stringRepresentation(board)
			score = self.game.getScore(board, 1)
			action = plays[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
			newBoard, curPlayer = self.game.getNextState(board, curPlayer, action)
			newScore = self.game.getScore(newBoard, 1)
			reward = (newScore-score)
			episode.append((state, board, action))
			rewards.append(reward)
			board = newBoard
			
			# print("newBoard=")
			# display(board)
			# print("score={}, newScore={}, reward={}\n\n".format(score, newScore, reward))
		
		return episode, rewards
	
	def train(self, n_episodes=1):
		for e in range(n_episodes):
			if e%100 == 0: print("Training... episode {}      ".format(e))
			self.epsilon = 1/(1+e) # Dynamic epsilon
			episode, rewards = self.generate_episode()
			
			for i in range(int(np.ceil(len(episode)/2))):
				'''Note: the nature of these games disallows the occurence of the same
				state more than once per episode. Hence we can directly update pi
				after updating Q'''
				s, b, a = episode[2*i]
				# Neemt alleen de zetten van wit mee. Moet een betere oplossing voor zijn...
				
				# Update returns
				r_sa = sum(rewards[i:]) # Return following action a in state s
				if (s, a) in self.R: self.R[(s, a)].append(r_sa)
				else: self.R[(s, a)] = [r_sa]
				
				# Update Q
				self.Q[(s, a)] = np.mean(self.R[(s, a)])
				
				# Update pi
				Q_s = {sa: r for sa, r in self.Q.items() if sa[0]==s}
				a_best = max(Q_s, key=Q_s.get)[1] # Pick action with highest return
				valids = self.game.getValidMoves(b, 1)
				v_a = [ai for ai in range(len(valids)) if valids[ai]==1]	# v_a is A(s)
				v_l = len(v_a)												# v_l is |A(s)|
				for a_ in v_a: self.pi[(s, a_)] = self.epsilon/v_l
				self.pi[(s, a_best)] += 1-self.epsilon
	
	def play(self, board):
		state = self.game.stringRepresentation(board)
		valids = self.game.getValidMoves(board, 1)
		
		# Initial policy for yet unseen states
		if state not in self.pi:
			probs = valids/len(valids[valids==1])
			self.pi[state] = probs
		
		probs = self.pi[state]
		action = np.random.choice(range(len(valids)), size=1, p=probs)[0]
		
		return action
		