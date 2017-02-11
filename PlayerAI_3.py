import random
from BaseAI_3 import BaseAI

spacePremium = 10
scorePremium = 1

class PlayerAI(BaseAI):
    def getMove(self, grid):
        moves = grid.getAvailableMoves()
        scores=[(move, self.assessPotentialMove(grid, move)) for move in moves]
        rankedMoves = sorted(scores, key=lambda move: move[1], reverse=True)
        return rankedMoves[0][0]
        # return randint(0, 3)

    def assessPotentialMove(self, grid, move):
        grid2 = grid.clone()
        if grid2.move(move):
            return self.scoreGrid(grid2)
        else:
            return 0.0

    def scoreGrid(self, grid):
        emptyCells = len(grid.getAvailableCells())
        maxCell = grid.getMaxTile()
        return (emptyCells * spacePremium) + (maxCell * scorePremium)
