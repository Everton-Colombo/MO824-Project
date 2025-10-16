from typing import List, Tuple
import numpy as np

Node = Tuple[float, float]

class AgcspInstance:

    def __init__(self, grid_nodes: List[Node], obstacle_nodes: List[Node], sprayer_length: float):
        self.grid_nodes = np.array(grid_nodes)
        
        self.sprayer_length = sprayer_length
        for obs in obstacle_nodes:
            if obs not in grid_nodes:
                raise ValueError(f"Obstacle node {obs} is not in grid nodes.")
        self.obstacle_nodes = np.array(obstacle_nodes)

        self._perform_precomputations()

    def _perform_precomputations(self):
        self.node_count = len(self.grid_nodes)
        
        # Bounding box shape (used by evalutor for coverage calculations):
        self.min_coords = np.min(self.grid_nodes, axis=0)
        max_coords = np.max(self.grid_nodes, axis=0)
        self.bounding_box_shape = tuple(max_coords - self.min_coords + 1)

        # Validity mask (true when nodes from the bounding box are valid grid nodes) (also used by evaluator for coverage calculations):
        self.validity_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        shifted_nodes = self.grid_nodes - self.min_coords
        self.validity_mask[shifted_nodes[:, 0], shifted_nodes[:, 1]] = True
        
        # Obstacle mask (true when nodes from the bounding box are obstacles):
        self.obstacle_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        shifted_obstacles = self.obstacle_nodes - self.min_coords
        self.obstacle_mask[shifted_obstacles[:, 0], shifted_obstacles[:, 1]] = True
        