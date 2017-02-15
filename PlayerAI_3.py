import sys
import time
import math
import collections

from BaseAI_3 import BaseAI

space_weight = 1.0
score_weight = 1.0
compactability_weight = 1.0
monotonicity_weight = 1.0
smoothness_weight = 1.0
deadline_offset = 0.1
max_depth = 40
plus_infinity = float(sys.maxsize)
minus_infinity = -1.0 * plus_infinity


class Utility:
    def __init__(self, space_score, max_val_score, monotonicity_score, smoothness_score, division_factor):
        self.division_factor = division_factor
        self.smoothness_score = smoothness_score
        self.monotonicity_score = monotonicity_score
        self.max_val_score = max_val_score
        self.space_score = space_score

    @property
    def cumulative_score(self, offset=10, mon_range=20):
        return (self.space_score * space_weight + self.max_val_score * score_weight + \
                (self.monotonicity_score + offset) / mon_range * monotonicity_weight + \
                self.smoothness_score * smoothness_weight) / self.division_factor


class PlayerAI(BaseAI):
    def __init__(self):
        self.deadline = None
        self.max_depth = 0

    def __del__(self):
        print("Player AI shutting down")
        print("max depth: ", self.max_depth)

    def getMove(self, grid):
        self.deadline = time.process_time() + deadline_offset
        moves = self.getMaximizerMoves(grid)
        child_grids = [(self.gen_grid(move, grid), move) for move in moves]
        # create a list of tuple(score, grid, move)
        assessed_children = sorted(
            [(self.score_grid(child[0], plus_infinity, minus_infinity, True, len(moves)), child[0], child[1]) for child
             in
             child_grids], key=lambda m: m[0], reverse=True)
        return assessed_children[0][2]
        # return randint(0, 3)

    def score_grid(self, grid, alpha, beta, is_maximiser, num_moves, depth=0):
        if time.process_time() > self.deadline or self.terminal_test(grid) or depth >= max_depth:
            s = self.utility(grid, num_moves, is_maximiser)
            return s.cumulative_score

        self.max_depth = max(self.max_depth, depth)
        all_moves = [self.gen_grid(m, grid) for m in self.getMaximizerMoves(grid)] \
            if is_maximiser \
            else self.getMinimizerMoves(grid)
        all_scores = [self.score_grid(g if is_maximiser else g[0],
                                      alpha,
                                      beta,
                                      not is_maximiser,
                                      num_moves,
                                      depth + 1)
                      for g in all_moves]

        if is_maximiser:
            result = minus_infinity

            for s in all_scores:
                result = max(result, s)
                alpha = max(alpha, result)
                if beta <= alpha:
                    break
            return result
        else:
            result = plus_infinity
            # adjust the weights on each of the possible moves to reflect the probability that it will happen do this
            #  by summing all the probs across the board and then dividing each say there are 10 spaces left on the
            # board. each will get a 2 and a 4 move generated for it, for a total of 20 possible moves 10 will have
            # prob P_2 == 90% 10 will have prob P_4 == 10%., so the total prob per cell is 100%, and the prob of each
            #  branch is The sum of probabilities will always add up to the number of possible moves, so each move
            # can just be divided by that to get its overall probability
            for s in all_scores:
                result = min(result, s)
                alpha = min(alpha, result)
                if beta <= alpha:
                    break
            return result

    def gen_grid(self, move, grid):
        result = grid.clone()
        if result.move(move):
            return result
        else:
            return None

    def terminal_test(self, grid):
        return not grid.canMove()

    def utility(self, grid, num_moves, is_maximising):
        # if is not maximising, then the utility needs to be weighted by the probablility of the move occurring since
        # the minimisers moves are probabilistic
        space_score = (len(grid.getAvailableCells()) / (grid.size * grid.size))
        max_val_score = (grid.getMaxTile() / 2048)
        monotonicity_score = self.calculate_monotonicity2(grid)
        smoothness_score = self.calculate_smoothness(grid)
        division_factor = (float(num_moves) if not is_maximising else 1.0)
        return Utility(space_score, max_val_score, monotonicity_score, smoothness_score, division_factor)

    def getMaximizerMoves(self, grid):
        moves = grid.getAvailableMoves()
        # always reject up(??) unless it is the only option to move
        if len(moves) == 1:
            return moves
        if moves.count(0) > 0: # 0 is the value used for DOWN in grid
            moves.remove(0)
        return moves


    def getMinimizerMoves(self, grid):
        cells = grid.getAvailableCells()
        new_grids = []
        possibleNewTiles = [2, 4]

        for cell in cells:
            for new_value in possibleNewTiles:
                new_grid = grid.clone()
                new_grid.setCellValue(cell, new_value)
                new_grids.append((new_grid, 0.9 if new_value == 2 else 0.1))
        return new_grids

    def calculate_monotonicity(self, grid):
        result = 0.0

        for row in range(0, grid.size - 1):
            for col in range(0, grid.size - 1):
                if grid.getCellValue((row, col)) <= grid.getCellValue((row, col + 1)):
                    result += 1.0
        for cols in range(0, grid.size - 1):
            for rows in range(0, grid.size - 1):
                if grid.getCellValue((row, col)) <= grid.getCellValue((row + 1, col)):
                    result += 1.0

        # for x in range(0, grid.size - 1):
        #     for y in range(0, grid.size - 1):
        #         xy = grid.getCellValue((x, y))
        #         x1y = grid.getCellValue((x + 1, y))
        #         xy1 = grid.getCellValue((x, y + 1))
        #         result += 1.0 if (xy <= x1y) else 0.0
        #         result += 1.0 if (xy <= xy1) else 0.0

        # maximum possible score for this would be when every cell scores both upwards as well as across
        # one row and one col can't be tested since they have no neighbours above or across, so the total
        # possible tests is (grid.size-1)^2
        max_possible_score = pow(grid.size, 2)
        return result / float(max_possible_score)

    def calculate_monotonicity2(self, grid):
        # first get cumulative scores for all rows
        row_scores = sum([self.vector_monotonicity(v) for v in grid.map])
        col_scores = sum([self.vector_monotonicity(self.get_column(grid, col)) for col in range(0, grid.size)])
        return row_scores + col_scores

    def get_column(self, grid, col_index):
        return [grid.getCellValue((x, col_index)) for x in range(0, grid.size)]

    def vector_monotonicity(self, vec):
        acc = 0.0
        for x in range(1, len(vec)):
            v1 = math.log(vec[x - 1]) / math.log(2) if vec[x - 1] > 0 else 0
            v2 = math.log(vec[x]) / math.log(2) if vec[x] > 0 else 0
            acc += (v2 - v1)
        return acc

    def calculate_smoothness(self, grid):
        result = 0.0

        for row in range(0, grid.size - 1):
            for col in range(0, grid.size - 1):
                c1 = grid.getCellValue((row, col))
                c2 = grid.getCellValue((row, col+1))
                if c1 == 0:  # skip the zeros, since this is a kinda of test of mergeability
                    pass
                elif c1 == c2:
                    result += 1.0

        for cell in range(0, grid.size - 1):
            for row in range(0, grid.size - 1):
                c1 = grid.getCellValue((row, col))
                c2 = grid.getCellValue((row+1, col))
                if c1 == 0:
                    pass
                elif c1 == c2:
                    result += 1.0

        max_possible_score = pow(grid.size-1, 2)*2
        return result / float(max_possible_score)
