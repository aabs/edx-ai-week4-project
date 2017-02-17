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
    def __init__(self, space_weight=3.0
                 , score_weight=1.0
                 , monotonicity_weight=1.0
                 , smoothness_weight=6.0):
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight
        self.score_weight = score_weight
        self.space_weight = space_weight


class Utility:
    def __init__(self, weights, space_score, max_val_score, monotonicity_score, smoothness_score, division_factor):
        self.weights = weights
        self.division_factor = division_factor
        self.smoothness_score = smoothness_score
        self.monotonicity_score = monotonicity_score
        self.max_val_score = max_val_score
        self.space_score = space_score

    @property
    def cumulative_score(self, offset=10, mon_range=20):
        return (self.space_score * self.weights.space_weight + self.max_val_score * self.weights.score_weight + \
                (self.monotonicity_score + offset) / mon_range * self.weights.monotonicity_weight + \
                self.smoothness_score * self.weights.smoothness_weight)


class AuditPoint:
    def __init__(self, highest_scoring_move, lowest_scoring_move, move_chosen, grid_chosen, branches_ignored,
                 max_depth):
        self.grid_chosen = grid_chosen
        self.move_chosen = move_chosen
        self.lowest_scoring_move = lowest_scoring_move
        self.highest_scoring_move = highest_scoring_move
        self.branches_ignored = branches_ignored
        self.max_depth = max_depth


class PlayerAI(BaseAI):
    def __init__(self):
        self.deadline = None
        self.max_depth = 0
        self.moves = []

        # fields used for keeping track of the player AIs behaviour

        self._highest_scoring_move = 0.0
        self._lowest_scoring_position = 0.0
        self._move_chosen = None
        self._grid_chosen = None
        self._branches_ignored = 0
        self.weights = AlgorithmWeights()
        self.memo_monotonicity_scores = SafeDict([])
        self.memo_smoothness_scores = SafeDict([])
        self.snake = [10, 8, 7, 6.5, .5, .7, 1, 3, -.5, -1.5, -1.8, -2, -3.8, -3.7, -3.5, -3]

    def set_weights(self, space_weight=2.0
                    , score_weight=1.0
                    , monotonicity_weight=1.0
                    , smoothness_weight=1.0
                    ):
        self.weights = AlgorithmWeights(space_weight
                                        , score_weight
                                        , monotonicity_weight
                                        , smoothness_weight)
        print("weights",
              space_weight
              , score_weight
              , monotonicity_weight
              , smoothness_weight)

    def audit(self, assessed_children):
        self._highest_scoring_move = assessed_children[0][0]
        self._lowest_scoring_move = assessed_children[len(assessed_children) - 1][0]
        self._move_chosen = assessed_children[0][2]
        self._grid_chosen = assessed_children[0][1]
        self._audit(self._highest_scoring_move, self._lowest_scoring_position, self._move_chosen, self._grid_chosen,
                    self._branches_ignored, self.max_depth)

    def _audit(self, highest_scoring_move, lowest_scoring_move, move_chosen, grid_chosen, branches_ignored, max_depth):
        self.transcript.append(
            AuditPoint(highest_scoring_move, lowest_scoring_move, move_chosen, grid_chosen, branches_ignored,
                       max_depth))

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
                free = len(child_grid.getAvailableCells())
                s = self.score_grid((child_grid, move),
                                    alpha,
                                    beta,
                                    not is_maximiser,
                                    num_moves,
                                    depth - 1)
                result = max(result, s)
                alpha = max(alpha, result)
                if beta <= alpha:
                    self._branches_ignored += 1
                    break
            return result
        else:
            result = plus_infinity
            sub_moves = self.getMinimizerMoves(grid)
            for move in sub_moves:
                (child_grid, prob) = move
                s = self.score_grid((child_grid, move),
                                    alpha,
                                    beta,
                                    not is_maximiser,
                                    num_moves,
                                    depth - 1)
                result = min(result, s)
                alpha = min(alpha, result)
                if beta <= alpha:
                    self._branches_ignored += 1
                    break
            # adjust the weights on each of the possible moves to reflect the probability that it will happen do this
            #  by summing all the probs across the board and then dividing each say there are 10 spaces left on the
            # board. each will get a 2 and a 4 move generated for it, for a total of 20 possible moves 10 will have
            # prob P_2 == 90% 10 will have prob P_4 == 10%., so the total prob per cell is 100%, and the prob of each
            #  branch is The sum of probabilities will always add up to the number of possible moves, so each move
            # can just be divided by that to get its overall probability
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
        self._branches_ignored = 0

    def utility4(self, grid: Grid):
        cells = len(grid.getAvailableCells())
        return cells * self.weights.space_weight + self.dot_product(grid) * self.weights.smoothness_weight

    def dot_product(self, grid):
        return sum([a * b for a, b in zip(self.snake, self.grid_to_list(grid))])

    def grid_to_list(self, grid):
        return grid.map[0] + grid.map[1] + grid.map[2] + grid.map[3]
