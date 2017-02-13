import sys
import time
from random import randint

from BaseAI_3 import BaseAI
from Grid_3 import vecIndex, directionVectors

space_weight = 1
score_weight = 1
compactability_weight = 0
monotonicity_weight = 1
deadline_offset = 0.1
max_depth = 10


class PlayerAI(BaseAI):
    def __init__(self):
        self.deadline = None
        self.max_depth = 0

    def __del__(self):
        print("Player AI shutting down")
        print("max depth: ", self.max_depth)

    def getMove(self, grid):
        self.deadline = time.process_time() + deadline_offset
        moves = grid.getAvailableMoves()
        child_grids = [(self.gen_grid(move, grid), move) for move in moves]
        # create a list of tuple(score, grid, move)
        assessed_children = sorted(
            [(self.score_grid(child[0], sys.maxsize, -1 * sys.maxsize, True, ), child[0], child[1]) for child in
             child_grids], key=lambda m: m[0], reverse=True)
        return assessed_children[0][2]
        # return randint(0, 3)

    def gen_grid(self, move, grid):
        result = grid.clone()
        if result.move(move):
            return result
        else:
            return None

    def terminal_test(self, grid):
        return not grid.canMove()

    def utility(self, grid):
        empty_cells = len(grid.getAvailableCells())
        max_cell = grid.getMaxTile()
        empty_cells_norm = empty_cells / (grid.size * grid.size)
        high_score_norm = max_cell / 2048
        monotonicity_norm = self.calculate_monotonicity(grid)
        #number_of_adjacent_duplicates = self.count_adjacent_dups(grid) / ((grid.size * grid.size) / 2)
        # penalise grids where a high score was got at the expense of losing space
        return (empty_cells_norm * space_weight) \
               + (high_score_norm * score_weight) \
               + (monotonicity_norm * monotonicity_weight)#\
               #+ (number_of_adjacent_duplicates * compactability_premium)
        # return (empty_cells * spacePremium) + (max_cell * scorePremium)

    def count_adjacent_dups(self, grid):
        count = 0
        for x in range(grid.size):
            for y in range(grid.size):
                if self.cell_has_adjacent_dups(grid, x, y):
                    count += 1
        return count / 2

    def score_grid(self, grid, alpha, beta, is_maximiser, depth=0):
        if time.process_time() > self.deadline or self.terminal_test(grid) or depth >= max_depth:
            return self.utility(grid)
        self.max_depth = max(self.max_depth, depth)
        if is_maximiser:
            result = -1 * sys.maxsize
            moves = self.getMaximizerMoves(grid)
            for move in moves:
                next = grid.clone()
                if next.move(move):
                    result = max(result, self.score_grid(next, alpha, beta, not is_maximiser, depth + 1))
                    alpha = max(alpha, result)
                    if beta <= alpha:
                        break
            return result
        else:
            result = sys.maxsize
            moves = self.getMaximizerMoves(grid)
            for move in moves:
                next = grid.clone()
                if next.move(move):
                    #self.simulate_insert_randon_tile(next)
                    result = min(result, self.score_grid(next, alpha, beta, not is_maximiser, depth + 1))
                    alpha = min(alpha, result)
                    if beta <= alpha:
                        break
            return result

    def simulate_insert_randon_tile(self, grid):
        tileValue = self.getNewTileValue()
        cells = grid.getAvailableCells()
        cell = cells[randint(0, len(cells) - 1)]
        grid.setCellValue(cell, tileValue)

    def getNewTileValue(self):
        if randint(0, 99) < 100 * 0.9:
            return 2
        else:
            return 4

    def getMaximizerMoves(self, grid):
        return grid.getAvailableMoves()

    def getMinimizerMoves(self, grid):
        cells = grid.getAvailableCells()
        new_grids = []
        possibleNewTiles = [2, 4]

        for cell in cells:
            for new_value in possibleNewTiles:
                new_grid = grid.clone()
                new_grid.setCellValue(cell, new_value)
                new_grids.append(new_grid)
        return new_grids

    @staticmethod
    def cell_has_adjacent_dups(grid, x, y):
        checking_moves = set(vecIndex)
        for i in checking_moves:
            move = directionVectors[i]
            adjCellValue = grid.getCellValue((x + move[0], y + move[1]))
            if adjCellValue is None:
                return False
            if adjCellValue == 0:
                break
            if adjCellValue == grid.getCellValue((x, y)):
                return True
        return False

    def calculate_monotonicity(self, grid):
        result = 0
        for row in range(grid.size):
            is_monotonic = True
            for col in range(grid.size):
                a = grid.getCellValue((row, col))
                b = grid.getCellValue((row, col+1))
                is_monotonic = is_monotonic and a is not None and b is not None and (a < b)
            if is_monotonic:
                result += 1
        for col in range(grid.size):
            is_monotonic = True
            for row in range(grid.size):
                a = grid.getCellValue((row, col))
                b = grid.getCellValue((row+1, col))
                is_monotonic = is_monotonic and a is not None and b is not None and (a < b)
                if is_monotonic:
                    result += 1
        return result / (2 * grid.size)
