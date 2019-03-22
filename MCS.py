class MCPlayer():
    def __init__(self, game):
        self.game = game
         
    def play(self, board):
        best_outcome = 0
        valids = self.game.getValidMoves(board, 1)
        for i in range(len(valids)):
             outcome = Arena.playGames(valids[i],10);
             if outcome > best_outcome
                 best_outcome = outcome
                 best_move = valids[i]
     return best_move

