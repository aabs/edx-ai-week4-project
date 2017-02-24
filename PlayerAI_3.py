import sys
import time

import multiprocessing

from AlgWeights import AlgorithmWeights
from BaseAI_3 import BaseAI
from FastGrid import FastGrid
from Grid_3 import Grid
from Util import Util
from UtilityCalculation import CompositeUtilityCalculator

deadline_offset = 0.1  # mandated solution timeout for exercise is .1 secs
max_depth_allowed = 4  # how deep to search for solutions

# some constants for initialising alpha and beta values in minimax
plus_infinity = float(sys.maxsize)
minus_infinity = -1.0 * plus_infinity


# some resources
# http://www.wikihow.com/Beat-2048#Step_by_Step_Strategy_Guide_sub
# http://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048




class PlayerAI(BaseAI):
    def __init__(self):
        self.deadline = None  # used to set a timeout on the exploration of possible moves
        self.moves = []
        self.util_engine = CompositeUtilityCalculator(AlgorithmWeights(free_space_weight=3.0
                                                                       , monotonicity_weight=1.0
                                                                       , roughness_weight=0.0
                                                                       , max_tile_weight=1.0))
        # self.kernel = [[10, 8, 7, 6.5], [.5, .7, 1, 3], [-.5, -1.5, -1.8, -2], [-3.8, -3.7, -3.5, -3]]
        # self.kernel = [math.exp(x) for x in self.kernel]
        self.kernel = Util.compute_kernel(create_snake=False, ramp_amplification=1.5)

    def set_weights(self, space_weight=1.0
                    , monotonicity_weight=3.0
                    , roughness_weight=-3.0
                    , max_tile_weight=1.0
                    ):
        self.util_engine.weights = AlgorithmWeights(space_weight
                                                    , monotonicity_weight
                                                    , roughness_weight
                                                    , max_tile_weight)
        print("weights",
              space_weight
              , monotonicity_weight
              , roughness_weight
              , max_tile_weight)

    def getMove(self, slow_grid: Grid):
        grid = FastGrid(slow_grid)
        self.deadline = time.perf_counter() + deadline_offset
        choice = None

        result_queue = multiprocessing.Queue()
        args=[]
        for m in grid.moves:
            s, g = grid.move(m)
            assert s, "moves must be valid"
            args.append((g, m, result_queue))
        jobs = [multiprocessing.Process(target=self.start_ab_search, group=None, args=mc) for mc in args]
        for job in jobs: job.start()
        for job in jobs: job.join()
        results = [result_queue.get() for mc in args]

        for r in results:
            move, score = r

            if choice is None or score > choice[1]:
                choice = r
        self.moves.append(choice)
        return choice[0]

    def start_ab_search(self, grid: FastGrid, move, result_queue):
        score = self.alphabeta_search((grid, move),
                                      minus_infinity,
                                      plus_infinity,
                                      True,
                                      max_depth_allowed)
        result_queue.put((move, score))

    def alphabeta_search(self, gm, alpha, beta, is_maximiser, depth):
        (grid, originating_move) = gm

        if depth == 0 or self.terminal_test(grid) or time.perf_counter() >= self.deadline:
            return self.util_engine.compute_utility(grid)

        if is_maximiser:
            result = minus_infinity

            for move in grid.moves:
                ok, child_grid = grid.move(move)
                assert ok, "move should have been valid"
                s = self.alphabeta_search((child_grid, move),
                                          alpha,
                                          beta,
                                          False,
                                          depth - 1)
                # is this result better than anything I've seen on this node so far?
                result = max(result, s)
                # is this result better than anything I've seen on any node previously visited?
                alpha = max(alpha, result)

                # is this branch better than the worst that the minimiser can force me to?
                if beta <= alpha:
                    # if yes, then we can expect the minimiser to avoid this branch on principle.
                    return result
            return result
        else:
            result = plus_infinity
            sub_moves = self.getMinimizerMoves(grid)

            for minmove in sub_moves:
                (child_grid, prob) = minmove
                s = self.alphabeta_search((child_grid, None),
                                          alpha,
                                          beta,
                                          True,
                                          depth - 1)
                result = min(result, s)
                beta = min(beta, result)
                if beta <= alpha:
                    return result
            return result

    def terminal_test(self, grid: FastGrid):
        return not grid.canMove()

    def getMaximizerMoves(self, grid):
        moves = grid.get_available_moves()
        return moves

    def getMinimizerMoves(self, grid):
        # get the most likely cells
        cells = grid.get_available_cells()
        new_grids = []
        # possible_new_tiles = [2, 4]
        possible_new_tiles = [2, 4]
        for cell in cells:
            for new_value in possible_new_tiles:
                new_grid = grid.clone()
                new_grid.setCellValue(cell, new_value)
                new_grids.append((new_grid, 1.0))
        return new_grids


        # def utility6(self, grid: FastGrid):
        #     hash_key = self.compute_grid_hash_key(grid)
        #     ce = self.cache_grid_scores[hash_key]
        #     if ce is not None:
        #         ce._replace(hit_count=ce.hit_count + 1)
        #         self.cache_grid_scores[hash_key] = CacheEntry(str_repr=ce.str_repr, score=ce.score, hash_key=ce.hash_key,
        #                                                       hit_count=ce.hit_count + 1)
        #         return ce.score
        #     r = 0.0
        #
        #     if self.weights.max_tile_weight != 0.0:
        #         max_tile = grid.getMaxTile()
        #         r += max_tile * self.weights.max_tile_weight
        #     if self.weights.monotonicity_weight != 0.0:
        #         r += self.mono3(grid) * self.weights.monotonicity_weight
        #     if self.weights.roughness_weight != 0.0:
        #         r += self.roughness_fast(grid) * self.weights.roughness_weight
        #     if self.weights.free_space_weight != 0.0:
        #         space = len(grid.getAvailableCells())
        #         space_ = (1.0 / space ** 0.9) if space > 0.0 else 1.0
        #         crampedness = self.weights.free_space_weight * min(1,
        #                                                            1 - space_)  # this figure grows geometrically as space dwindles
        #         r *= crampedness  # cramped boards are to be avoided at all costs. Penalise them heavily
        #     ce = self.create_cache_entry(grid, r)
        #     self.cache_grid_scores[hash_key] = ce
        #     return r
        #
        # def utility5(self, grid: FastGrid):
        #     r = 0.0
        #     if self.weights.free_space_weight != 0.0:
        #         cells = len(grid.getAvailableCells())
        #         max_tile = grid.getMaxTile()
        #         r += cells * math.log(max_tile, 2) * self.weights.free_space_weight
        #     if self.weights.max_tile_weight != 0.0:
        #         max_tile = grid.getMaxTile()
        #         r += max_tile * self.weights.max_tile_weight
        #     if self.weights.monotonicity_weight != 0.0:
        #         r += self.dot_product(grid) * self.weights.monotonicity_weight
        #     if self.weights.roughness_weight != 0.0:
        #         r += self.clustering(grid) * self.weights.roughness_weight
        #     return r
        #
        # def utility4(self, grid: FastGrid):
        #     r = 0.0
        #     if self.weights.free_space_weight != 0.0:
        #         cells = len(grid.getAvailableCells())
        #         r += cells * self.weights.free_space_weight
        #     if self.weights.max_tile_weight != 0.0:
        #         max_tile = grid.getMaxTile()
        #         r += max_tile * self.weights.max_tile_weight
        #     if self.weights.monotonicity_weight != 0.0:
        #         r += self.dot_product(grid) * self.weights.monotonicity_weight
        #     if self.weights.roughness_weight != 0.0:
        #         r += self.roughness_fast(grid) * self.weights.roughness_weight
        #     return r
