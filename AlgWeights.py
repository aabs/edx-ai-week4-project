class AlgorithmWeights:
    def __init__(self, free_space_weight=2.0
                 , monotonicity_weight=1.0
                 , roughness_weight=-1.0
                 , max_tile_weight=0.0):
        self.max_tile_weight = max_tile_weight
        self.roughness_weight = roughness_weight
        self.monotonicity_weight = monotonicity_weight
        self.free_space_weight = free_space_weight
