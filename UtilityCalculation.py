import math

import array

from AlgWeights import AlgorithmWeights
from Caching import GridCache
from Grid_3 import Grid
from Util import Util


class UtilityCalculator:
    def compute_utility(self, grid: Grid) -> float:
        pass


class MaxTileCalculator:
    def compute_utility(self, grid: Grid) -> float:
        return grid.getMaxTile()


class CompositeUtilityCalculator(UtilityCalculator):
    def __init__(self, weights: AlgorithmWeights = None):
        self.weights = weights
        if self.weights is None:
            self.weights = AlgorithmWeights()
        self.score_cache = GridCache()
        self.free_space_calculator = FreeSpaceCalculator()
        self.roughness_calculator = RoughnessCalculator()
        self.monotonicity_calculator = MonotonicityCalculator()
        self.max_tile_calculator = MaxTileCalculator()

    def compute_utility(self, grid: Grid) -> float:
        if self.score_cache.contains_grid(grid):
            return self.score_cache[grid]
        r = 0.0

        if self.weights.max_tile_weight != 0.0:
            max_tile = self.max_tile_calculator.compute_utility(grid)
            r += max_tile * self.weights.max_tile_weight
        if self.weights.monotonicity_weight != 0.0:
            r += self.weights.monotonicity_weight * self.monotonicity_calculator.compute_utility(grid)
        if self.weights.roughness_weight != 0.0:
            r += self.weights.roughness_weight * self.roughness_calculator.compute_utility(grid)
        if self.weights.free_space_weight != 0.0:
            r *= self.weights.free_space_weight * self.free_space_calculator.compute_utility(grid)
        self.score_cache[grid] = r
        return r


class FreeSpaceCalculator(UtilityCalculator):
    def __init__(self, weight=1.0):
        self.weight = weight

    def compute_utility(self, grid: Grid):
        space = len(grid.getAvailableCells())
        space_ = (1.0 / space ** 0.9) if space > 0.0 else 1.0
        result = self.weight * min(1,
                                   1 - space_)  # this figure grows geometrically as space dwindles

        return result


class RoughnessCalculator(UtilityCalculator):
    def compute_utility(self, grid: Grid):
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
        # def roughness_grid(self, grid, width, height):
        #     """Assumes something constructed like this:
        #     M = [[0,1,2], [3,4,5], [6,7,8]]
        #     that can be indexed like this M[0][2]"""
        #
        #     r = [[0.0] * width for _ in range(height)]
        #
        #     for y in range(0, height):
        #         for x in range(0, width):
        #             sum = 0.0
        #             cnt = 0.0
        #             rxlo = max(0, x - 1)
        #             rxhi = min(width - 1, x + 1)
        #             rylo = max(0, y - 1)
        #             ryhi = min(height - 1, y + 1)
        #             v = math.log(grid[x][y], 2) if grid[x][y] != 0.0 else 0.0
        #             for i in range(rylo, ryhi + 1):
        #                 for j in range(rxlo, rxhi + 1):
        #                     if i != y or j != x:
        #                         n1 = math.log(grid[i][j], 2) if grid[i][j] != 0.0 else 0.0
        #                         sum += abs(v - n1)
        #                         cnt += 1.0
        #             r[x][y] = sum / cnt if cnt > 0.0 else 0.0
        #     return r
        #
        # def roughness(self, grid, width, height):
        #     sum = 0.0
        #     r = self.roughness_grid(grid, width, height)
        #     for y in range(0, height):
        #         for x in range(0, width):
        #             sum += r[x][y]
        #     return sum / (width * height)


class MonotonicityCalculator(UtilityCalculator):
    """
calculate the monotonicity of a grid


Original Performance (100000 calls using UtilityCalculatorTests.test_profile_monotonicity_calculator
             16433615 function calls in 9.892 seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       100000    3.115    0.000    9.892    0.000 UtilityCalculation.py:51(compute_utility)
      4533510    2.160    0.000    3.245    0.000 UtilityCalculation.py:55(get)
      2400000    1.190    0.000    1.190    0.000 Grid_3.py:172(crossBound)
      2400000    1.176    0.000    2.366    0.000 Grid_3.py:175(getCellValue)
      2400000    1.109    0.000    3.475    0.000 UtilityCalculation.py:52(not_zero)
      4400103    1.084    0.000    1.084    0.000 {built-in method math.log}
       200000    0.058    0.000    0.058    0.000 {built-in method builtins.max}
            1    0.000    0.000    0.000    0.000 ABTestingBase.py:20(stop_profiling)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

This needs to be no more than 0.09 s i.e. a 100 fold improvement is required to get the depth needed

    """

    def compute_utility(self, grid: Grid):
        a = Util.grid_to_array(grid)
        totals = array.array('i', [0, 0, 0, 0])

        # // up/down direction
        for x in range(4):
            current = 0
            neighbour = current + 1
            while neighbour < 4:
                while neighbour < 4 and a[(neighbour * 4) + x] == 0:
                    neighbour += 1
                if neighbour >= 4:
                    neighbour -= 1
                current_value = a[(current * 4) + x]  # get_val(a, x, current)
                next_value = a[(neighbour * 4) + x]  # get_val(a, x, neighbour)
                if current_value > next_value:
                    totals[0] += next_value - current_value
                elif next_value > current_value:
                    totals[1] += current_value - next_value
                current = neighbour
                neighbour += 1
                #
        # // left/right direction
        for y in range(4):
            current = 0
            neighbour = current + 1
            while neighbour < 4:
                while neighbour < 4 and a[(y * 4) + neighbour] == 0:
                    neighbour += 1
                if neighbour >= 4:
                    neighbour -= 1
                current_value = a[(y * 4) + current]  # get_val(a, current, y)
                next_value = a[(y * 4) + neighbour]  # get_val(a, neighbour, y)
                if current_value > next_value:
                    totals[2] += next_value - current_value
                elif next_value > current_value:
                    totals[3] += current_value - next_value
                current = neighbour
                neighbour += 1
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def compute_utility_original(self, grid: Grid):
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


class ClusteringCalculator(UtilityCalculator):
    # https://raw.githubusercontent.com/datumbox/Game-2048-AI-Solver/master/src/com/datumbox/opensource/ai/AIsolver.java

    def compute_utility(self, grid: Grid) -> float:
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
