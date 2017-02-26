import logging

import FastGrid
from AlgWeights import AlgorithmWeights
from Caching import GridCache
from KernelCalculator import KernelCalculator
from UtilityCalculation import UtilityCalculator, FreeSpaceCalculator, RoughnessCalculator, MonotonicityCalculator, \
    MaxTileCalculator, ClusteringCalculator

log = logging.getLogger('app' + __name__)

class CompositeUtilityCalculator(UtilityCalculator):
    def __init__(self, weights: AlgorithmWeights = None):
        self.weights = weights
        if self.weights is None:
            self.weights = AlgorithmWeights()
        self.use_cache = None
        if self.use_cache:
            self.score_cache = GridCache()
        self.free_space_calculator = FreeSpaceCalculator()
        self.roughness_calculator = RoughnessCalculator()
        self.monotonicity_calculator = MonotonicityCalculator()
        self.max_tile_calculator = MaxTileCalculator()
        self.kernel_calculator = KernelCalculator()
        self.clustering_calculator = ClusteringCalculator()
        log.debug("Composite Calculator.")
        log.debug("weights %s", repr(self.weights))

    def compute_utility(self, grid: FastGrid) -> float:
        if self.use_cache and self.score_cache.contains_grid(grid):
            score = self.score_cache[grid]
            log.debug("cache hit for %d : %0.3f", hash(grid), score)
            return score
        r = 0.0
        r += self.calc_tile_score(grid)
        r += self.calc_monotonicity_score(grid)
        r += self.calc_roughness_score(grid)
        r += self.calc_kernel_score(grid)
        r += self.calc_clustering_score(grid)
        r += self.calc_space_score(grid)
        if self.use_cache:
            self.score_cache[grid] = r
        return r

    def calc_clustering_score(self, g: FastGrid):
        if self.weights.clustering_weight != 0.0:
            return self.clustering_calculator.compute_utility(g) * self.weights.clustering_weight
        return 0.0

    def calc_tile_score(self, g: FastGrid):
        if self.weights.max_tile_weight != 0.0:
            max_tile = self.max_tile_calculator.compute_utility(g)
            return max_tile * self.weights.max_tile_weight
        return 0.0


    def calc_monotonicity_score(self, g: FastGrid):
        if self.weights.monotonicity_weight != 0.0:
            x = self.monotonicity_calculator.compute_utility(g)
            return x * self.weights.monotonicity_weight
        return 0.0


    def calc_roughness_score(self, g: FastGrid):
        if self.weights.roughness_weight != 0.0:
            return self.weights.roughness_weight * self.roughness_calculator.compute_utility(g)
        return 0.0


    def calc_kernel_score(self, g: FastGrid):
        if self.weights.kernel_weight != 0.0:
            return self.weights.kernel_weight * self.kernel_calculator.compute_utility(g)
        return 0.0


    def calc_space_score(self, g: FastGrid):
        if self.weights.free_space_weight != 0.0:
            return self.weights.free_space_weight * self.free_space_calculator.compute_utility(g)
        return 1.0