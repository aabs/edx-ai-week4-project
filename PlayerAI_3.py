import sys
import time
import math

from BaseAI_3 import BaseAI
from Grid_3 import Grid

deadline_offset = 0.09
max_depth = 8
plus_infinity = float(sys.maxsize)
minus_infinity = -1.0 * plus_infinity
orientation = [EAST, SOUTH, SOUTHEAST, NORTHEAST] = range(4)


# http://www.wikihow.com/Beat-2048#Step_by_Step_Strategy_Guide_sub
# http://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

class SafeDict(dict):
    def __missing__(self, key):
        return None


class AlgorithmWeights:
    def __init__(self, space_weight=1.0
                 , score_weight=1.0
                 , monotonicity_weight=1.0
                 , smoothness_weight=1.0):
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight
        self.score_weight = score_weight
        self.space_weight = space_weight

class PlayerAI(BaseAI):
    def __init__(self):
        self.deadline = None
        self.max_depth = 0
        self.moves = []
        self.weights = AlgorithmWeights()
        self.memo_monotonicity_scores = SafeDict([])
        self.memo_smoothness_scores = SafeDict([])
        # self.snake = [10, 8, 7, 6.5, .5, .7, 1, 3, -.5, -1.5, -1.8, -2, -3.8, -3.7, -3.5, -3]
        # self.snake = [x * 10 for x in self.snake]
        self.snake = self.compute_kernel()

    def set_weights(self, space_weight=1.0
                    , smoothness_weight=1.0
                    ):
        self.weights = AlgorithmWeights(space_weight
                                        , smoothness_weight)
        print("weights",
              space_weight
              , smoothness_weight)
    def compute_kernel(self) -> list:
        width = 4
        weight = 2.5
        result = [0] * int(math.pow(width, 2))
        min_val = math.pow(2, weight)
        max_val = math.pow((2*(width-1))+2, weight)
        total_range = max_val - min_val
        for x in range(0, width):
            for y in range(0, width):
                idx = (y * width) + x
                result[idx] = math.pow(x + y + 2, weight) - (total_range/2)
        return result
    def __del__(self):
        print("Player AI shutting down")
        print("max depth: ", self.max_depth)

    def getMove(self, grid):
        self.reset_stats()
        self.deadline = time.clock() + deadline_offset
        moves = self.getMaximizerMoves(grid)
        if len(moves) == 1:
            # no point messing about, just make the move
            self.moves.append(moves[0])
            return moves[0]

        child_grids = [(self.gen_grid(move, grid), move) for move in moves]
        # create a list of tuple(score, grid, move)
        assessed_children = sorted(
            [(self.score_grid(child, minus_infinity, plus_infinity, True, len(moves), 2), child[0], child[1]) for child
             in
             child_grids], key=lambda m: m[0], reverse=True)
        chosen_move = assessed_children[0][2]
        self.moves.append(chosen_move)
        return chosen_move

    def score_grid(self, gm, alpha, beta, is_maximiser, num_moves, depth=max_depth):
        (grid, originating_move) = gm

        self.max_depth = max(self.max_depth, depth)

        if depth == 0 or time.clock() >= self.deadline or self.terminal_test(grid):
            # return self.utility(grid)
            return self.utility4(grid)
            # return s.cumulative_score

        if is_maximiser:
            result = minus_infinity

            for move in self.getMaximizerMoves(grid):
                child_grid = self.gen_grid(move, grid)
                s = self.score_grid((child_grid, move),
                                    alpha,
                                    beta,
                                    not is_maximiser,
                                    num_moves,
                                    depth - 1)
                # is this result better than anything I've seen on this node so far?
                result = max(result, s)
                # is this result better than anything I've seen on any node previously visited?
                alpha = max(alpha, result)

                # is this branch better than the worst that the minimiser can force me to?
                if beta <= alpha:
                    # if yes, then we can expect the minimiser to avoid this branch on principle.
                    break
            return result
        else:
            result = plus_infinity
            sub_moves = self.getMinimizerMoves(grid)

            for move in sub_moves:
                (child_grid, _) = move
                s = self.score_grid((child_grid, move),
                                    alpha,
                                    beta,
                                    not is_maximiser,
                                    num_moves,
                                    depth - 1)
                result = min(result, s)
                beta = min(beta, result)
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

    def getMaximizerMoves(self, grid):
        moves = grid.getAvailableMoves()
        # always reject up(??) unless it is the only option to move
        # if len(moves) == 1:
        #     return moves
        # if moves.count(0) > 0:  # 0 is the value used for DOWN in grid
        #     moves.remove(0)
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

    def get_grid_score(self, grid: Grid, scores: SafeDict) -> float:
        repr_hash = hash(str(grid.map))
        return scores[repr_hash]

    def set_grid_score(self, grid: Grid, scores: SafeDict, score: float):
        repr_hash = hash(str(grid.map))
        scores[repr_hash] = score

    def reset_stats(self):
        self._highest_scoring_move = 0.0
        self._lowest_scoring_position = 0.0
        self._move_chosen = None
        self._grid_chosen = None

    def utility4(self, grid: Grid):
        cells = len(grid.getAvailableCells())
        return cells * cells * self.weights.space_weight + self.dot_product(grid) * self.weights.smoothness_weight

    def dot_product(self, grid):
        return sum([a * b for a, b in zip(self.snake, self.grid_to_list(grid))])

    def grid_to_list(self, grid):
        return grid.map[0] + grid.map[1] + grid.map[2] + grid.map[3]
