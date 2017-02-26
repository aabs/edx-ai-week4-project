class AlgorithmWeights:
    def __init__(self, free_space_weight=2.0
                 , monotonicity_weight=0.0
                 , roughness_weight=0.0
                 , kernel_weight=1.0
                 , clustering_weight=0.0
                 , max_tile_weight=0.0):
        self.clustering_weight = clustering_weight
        self.kernel_weight = kernel_weight
        self.max_tile_weight = max_tile_weight
        self.roughness_weight = roughness_weight
        self.monotonicity_weight = monotonicity_weight
        self.free_space_weight = free_space_weight

    def __repr__(self):
        return """Weights:\n\tclustering = %f
\tkernel = %0.2f
\tmax_tile = %0.2f
\troughness = %0.2f
\tmonotonicity = %0.2f
\tfree_space = %0.2f""" % (
            self.clustering_weight,
            self.kernel_weight,
            self.max_tile_weight,
            self.roughness_weight,
            self.monotonicity_weight,
            self.free_space_weight
        )
