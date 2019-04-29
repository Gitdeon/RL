import numpy as np
import math


class MCTSPlayer():

    def __init__(self, game, args):
        
        self.game = game
        self.args = args
        self.sa_actionvalue = {}  # action value for (state,action) dict
        self.sa_visits = {}       # amount of visits to (state,action) dict
        self.s_status = {}        # game status for board state dict
        self.pi = {}              # initial policy dict
        self.s_boardvisits = {}   # amount of visits for board state dict
        self.s_moves = {}         # possible moves for board state dict
    

    def getActionProb(self, canonicalBoard, temp=1):

        for i in range(self.args.MCTSiterations):
            self.iteration(canonicalBoard)
        
        #reorder the following block
        state = self.game.stringRepresentation(canonicalBoard)
        counts = [self.sa_visits[(state,action)] if (state,action) in self.sa_visits else 0 for action in range(self.game.getActionSize())]
        
        if temp == 0:
            bestaction = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestaction] = 1
            return probs #policy vector with probability of action i
        
        counts = [count ** (1. / temp) for count in counts]
        probs = [count / float(sum(counts)) for count in counts]
        return probs


    def iteration(self, canonicalBoard):
        
        state = self.game.stringRepresentation(canonicalBoard)
        
        if state not in self.s_status: #add state outcome if not added yet
            self.s_status[state] = self.game.getGameEnded(canonicalBoard, 1)
        
        if self.s_status[state] != 0: #if node is terminal node
            return -self.s_status[state]

        if state not in self.pi:
            validmoves = self.game.getValidMoves(canonicalBoard, 1)
            self.pi[state] = np.random.uniform(-1,1,len(validmoves))
            value = np.random.uniform(-1,1,1)
            self.pi[state] = self.pi[state] * validmoves   #masking invalid moves?
            state_policy_sum = np.sum(self.pi[state])

            if state_policy_sum > 0:
                self.pi[state] /= state_policy_sum #normalize
            else:
                #All moves are masked, workaround:
                self.pi[state] = self.pi[state] + validmoves
                self.pi[state] /= np.sum(self.pi[state])
            
            self.s_moves[state] = validmoves
            self.s_boardvisits[state] =0
            return -value

        validmoves = self.s_moves[state]
        current_best = -float('inf')
        best_action = -1

        for action in range(self.game.getActionSize()):
            if validmoves[action]:
                if (state, action) in self.sa_actionvalue:
                    uct = self.sa_actionvalue[(state, action)] + self.args.exploration_param * self.pi[state][action] * math.sqrt(self.s_boardvisits[state]) / (1 + self.sa_visits[(state, action)])
                else:
                    uct = self.args.exploration_param * self.pi[state][action] * math.sqrt(self.s_boardvisits[state] + 1e-8)
                if uct > current_best:
                    current_best = uct
                    best_action = action
        action = best_action
        next_state, next_player = self.game.getNextState(canonicalBoard, 1, action)
        next_state = self.game.getCanonicalForm(next_state, next_player)

        value = self.iteration(next_state)

        if (state, action) in self.sa_actionvalue:
            self.sa_actionvalue[(state, action)] = (self.sa_visits[(state, action)] * self.sa_actionvalue[(state, action)] + value) / (self.sa_visits[(state, action)] + 1)
            self.sa_visits[(state, action)] += 1
        else:
            self.sa_actionvalue[(state, action)] = value
            self.sa_visits[(state, action)] = 1

        self.s_boardvisits[state] += 1
        return -value


