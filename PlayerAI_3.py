import math
import sys
import time
import itertools
from collections import namedtuple

from AlgWeights import AlgorithmWeights
from BaseAI_3 import BaseAI
from Grid_3 import Grid
from SafeDict import SafeDict

deadline_offset = 1.1  # mandated solution timeout for exercise is .1 secs
max_depth_allowed = 4  # how deep to search for solutions

# some constants for initialising alpha and beta values in minimax
plus_infinity = float(sys.maxsize)
minus_infinity = -1.0 * plus_infinity

# some resources
# http://www.wikihow.com/Beat-2048#Step_by_Step_Strategy_Guide_sub
# http://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

CacheEntry = namedtuple('CacheEntry', ['hash_key', 'str_repr', 'score', 'hit_count'])


class PlayerAI(BaseAI):
    def __del__(self):
        x = sorted(self.cache_grid_scores.items(), key=lambda v: v[1].hit_count, reverse=True)
        for ce in x[0:5]:
            print(ce)
    def __init__(self):
        self.deadline = None  # used to set a timeout on the exploration of possible moves
        self.max_depth_reached_so_far = 0
        self.moves = []
        self.weights = AlgorithmWeights(free_space_weight=4.0
                                        , monotonicity_weight=0.5
                                        , roughness_weight=1.0
                                        , max_tile_weight=1.0)
        self.cache_grid_scores = SafeDict([])
        # self.kernel = [[10, 8, 7, 6.5], [.5, .7, 1, 3], [-.5, -1.5, -1.8, -2], [-3.8, -3.7, -3.5, -3]]
        # self.kernel = [math.exp(x) for x in self.kernel]
        self.kernel = self.compute_kernel(create_snake=False, ramp_amplification=1.5)

    def create_cache_entry(self, g: Grid, score=0.0) -> CacheEntry:
        return CacheEntry(hash_key=self.compute_grid_hash_key(g), hit_count=1, score=score, str_repr=str(g.map))

    def compute_grid_hash_key(self, g: Grid) -> str:
        return str(g.map)

    def set_weights(self, space_weight=1.0
                    , monotonicity_weight=3.0
                    , roughness_weight=-3.0
                    , max_tile_weight=1.0
                    ):
        self.weights = AlgorithmWeights(space_weight
                                        , monotonicity_weight
                                        , roughness_weight
                                        , max_tile_weight)
        print("weights",
              space_weight
              , monotonicity_weight
              , roughness_weight
              , max_tile_weight)

    def compute_kernel(self, width=4, ramp_amplification: float = None, create_snake: bool = False) -> list:
        weight = ramp_amplification if ramp_amplification is not None else 1.0
        if create_snake:
            r = [weight * x for x in range(16, -16, -2)]
            tmp = self.list_to_grid(r)
            tmp[1] = list(reversed(tmp[1]))
            tmp[3] = list(reversed(tmp[3]))
            return self.grid_to_list(tmp)

        else:
            r = [0.0] * (width * width)
            min_val = math.pow(2, weight)
            max_val = math.pow((2 * (width - 1)) + 2, weight)
            total_range = max_val - min_val
            for row in range(0, width):
                for col in range(0, width):
                    idx = (row * width) + col
                    r[idx] = math.pow(row + col + 2, weight) - (total_range / 2) - min_val
            return r

    def getMove(self, grid):
        # str_moves = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.deadline = time.clock() + deadline_offset
        moves = self.getMaximizerMoves(grid)
        if len(moves) == 1:
            # no point messing about, just make the move
            self.moves.append(moves[0])
            return moves[0]

        child_grids = [(self.gen_grid(move, grid), move, str(grid.map)) for move in moves]
        # create a list of tuple(score, grid, move)
        choice = None
        for m in child_grids:
            (child, move, s) = m
            score = self.alphabeta_search((child, move), minus_infinity, plus_infinity, True, max_depth_allowed)
            # print("%s (%f): %s"%(str_moves[move], score, s))
            if choice is None or score > choice[0]:
                choice = (score, move)
        self.moves.append(choice)
        return choice[1]

    def alphabeta_search(self, gm, alpha, beta, is_maximiser, depth):
        (grid, originating_move) = gm
        self.max_depth_reached_so_far = max(self.max_depth_reached_so_far, depth)

        if depth == 0 or self.terminal_test(grid): #  or time.clock() >= self.deadline
            return self.utility6(grid)

        if is_maximiser:
            result = minus_infinity

            for move in self.getMaximizerMoves(grid):
                child_grid = self.gen_grid(move, grid)
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
        return moves

    def getMinimizerMoves(self, grid):
        # get the most likely cells
        cells = grid.getAvailableCells()
        new_grids = []
        # possibleNewTiles = [2, 4]
        possibleNewTiles = [2, 4]
        for cell in cells:
            for new_value in possibleNewTiles:
                new_grid = grid.clone()
                new_grid.setCellValue(cell, new_value)
                new_grids.append((new_grid, 1.0))
        return new_grids

    def dot_product(self, grid):
        return sum([a * b for a, b in zip(self.kernel, self.grid_to_list(grid.map))])

    def grid_to_list(self, g):
        return g[0] + g[1] + g[2] + g[3]

    def list_to_grid(self, l):
        return [l[0:4], l[4:8], l[8:12], l[12:16]]

    def roughness_grid(self, grid, width, height):
        """Assumes something constructed like this:
        M = [[0,1,2], [3,4,5], [6,7,8]]
        that can be indexed like this M[0][2]"""

        r = [[0.0] * width for _ in range(height)]

        for y in range(0, height):
            for x in range(0, width):
                sum = 0.0
                cnt = 0.0
                rxlo = max(0, x - 1)
                rxhi = min(width - 1, x + 1)
                rylo = max(0, y - 1)
                ryhi = min(height - 1, y + 1)
                v = math.log(grid[x][y], 2) if grid[x][y] != 0.0 else 0.0
                for i in range(rylo, ryhi + 1):
                    for j in range(rxlo, rxhi + 1):
                        if i != y or j != x:
                            n1 = math.log(grid[i][j], 2) if grid[i][j] != 0.0 else 0.0
                            sum += abs(v - n1)
                            cnt += 1.0
                r[x][y] = sum / cnt if cnt > 0.0 else 0.0
        return r

    def roughness(self, grid, width, height):
        sum = 0.0
        r = self.roughness_grid(grid, width, height)
        for y in range(0, height):
            for x in range(0, width):
                sum += r[x][y]
        return sum / (width * height)

    # a simple smoothness function from here: http://artent.net/2014/04/07/an-ai-for-2048-part-4-evaluation-functions/
    def roughness_fast(self, grid: Grid):
        (less, more, eq) = (0, 1, 2)
        sign = None
        changed_signs = 0
        for row in grid.map:
            for x in range(0, 3):
                # first classify the sign of the two nums being considered
                if row[x] < row[x + 1]:
                    sign_new = less  # where False means we were on less
                elif row[x] == row[x + 1]:
                    sign_new = eq
                else:
                    sign_new = more
                # if this is the first comparison on the line then there is no chang score possible
                # so just record the sign and then move on
                if sign is None:
                    sign = sign_new
                    continue
                # if we get here then we are not on the first comparison so we can do additions
                if sign_new != eq and sign != sign_new:
                    changed_signs += 1
                sign = sign_new
        return changed_signs

    # https://raw.githubusercontent.com/datumbox/Game-2048-AI-Solver/master/src/com/datumbox/opensource/ai/AIsolver.java
    def clustering(self, grid: Grid):
        clusteringScore = 0
        neighbors = [-1, 0, 1]
        for i in range(grid.size):
            for j in range(grid.size):
                v = grid.map[i][j]
                if v == 0:
                    continue
                clusteringScore -= v
                numOfNeighbors = 0
                acc = 0
                for k in neighbors:
                    x = i + k
                    if x < 0 or x >= grid.size:
                        continue
                    for l in neighbors:
                        y = j + l
                        if y < 0 or y >= grid.size:
                            continue
                        if grid.map[x][y] > 0:
                            numOfNeighbors += 1
                            acc += abs(v - grid.map[x][y])
                clusteringScore += acc / numOfNeighbors
        return clusteringScore

    def utility6(self, grid: Grid):
        hash_key = self.compute_grid_hash_key(grid)
        ce = self.cache_grid_scores[hash_key]
        if ce is not None:
            ce._replace(hit_count=ce.hit_count + 1)
            self.cache_grid_scores[hash_key] = CacheEntry(str_repr=ce.str_repr, score=ce.score, hash_key=ce.hash_key, hit_count=ce.hit_count + 1)
            return ce.score
        r = 0.0

        if self.weights.max_tile_weight != 0.0:
            max_tile = grid.getMaxTile()
            r += max_tile * self.weights.max_tile_weight
        if self.weights.monotonicity_weight != 0.0:
            r += self.mono3(grid) * self.weights.monotonicity_weight
        if self.weights.roughness_weight != 0.0:
            r += self.roughness_fast(grid) * self.weights.roughness_weight
        if self.weights.free_space_weight != 0.0:
            space = len(grid.getAvailableCells())
            space_ = (1.0 / space ** 0.9) if space > 0.0 else 1.0
            crampedness = self.weights.free_space_weight * min(1,
                                                               1 - space_)  # this figure grows geometrically as space dwindles
            r *= crampedness  # cramped boards are to be avoided at all costs. Penalise them heavily
        ce = self.create_cache_entry(grid, r)
        self.cache_grid_scores[hash_key] = ce
        return r

    def utility5(self, grid: Grid):
        r = 0.0
        if self.weights.free_space_weight != 0.0:
            cells = len(grid.getAvailableCells())
            max_tile = grid.getMaxTile()
            r += cells * math.log(max_tile, 2) * self.weights.free_space_weight
        if self.weights.max_tile_weight != 0.0:
            max_tile = grid.getMaxTile()
            r += max_tile * self.weights.max_tile_weight
        if self.weights.monotonicity_weight != 0.0:
            r += self.dot_product(grid) * self.weights.monotonicity_weight
        if self.weights.roughness_weight != 0.0:
            r += self.clustering(grid) * self.weights.roughness_weight
        return r

    def utility4(self, grid: Grid):
        r = 0.0
        if self.weights.free_space_weight != 0.0:
            cells = len(grid.getAvailableCells())
            r += cells * self.weights.free_space_weight
        if self.weights.max_tile_weight != 0.0:
            max_tile = grid.getMaxTile()
            r += max_tile * self.weights.max_tile_weight
        if self.weights.monotonicity_weight != 0.0:
            r += self.dot_product(grid) * self.weights.monotonicity_weight
        if self.weights.roughness_weight != 0.0:
            r += self.roughness_fast(grid) * self.weights.roughness_weight
        return r

    def mono3(self, grid: Grid):
        def not_zero(g: Grid, x, y):
            return g.getCellValue((x, y)) != 0.0

        def get(g: Grid, a, b):
            v = g.map[a][b]
            return math.log(v, 2) if v != 0.0 else 0.0

        totals = [0, 0, 0, 0]

        # // up/down direction
        for x in range(4):
            current = 0
            neighbour = current + 1
            while neighbour < 4:
                while neighbour < 4 and not not_zero(grid, x, neighbour):
                    neighbour += 1
                if neighbour >= 4:
                    neighbour -= 1
                current_value = get(grid, x, current)
                nextValue = get(grid, x, neighbour)
                if current_value > nextValue:
                    totals[0] += nextValue - current_value
                elif nextValue > current_value:
                    totals[1] += current_value - nextValue
                current = neighbour
                neighbour += 1
                #
        # // left/right direction
        for y in range(4):
            current = 0
            neighbour = current + 1
            while neighbour < 4:
                while neighbour < 4 and not not_zero(grid, neighbour, y):
                    neighbour += 1
                if neighbour >= 4:
                    neighbour -= 1
                current_value = get(grid, current, y)
                nextValue = get(grid, neighbour, y)
                if current_value > nextValue:
                    totals[2] += nextValue - current_value
                elif nextValue > current_value:
                    totals[3] += current_value - nextValue
                current = neighbour
                neighbour += 1
        return max(totals[0], totals[1]) + max(totals[2], totals[3])
