from typing import List, Tuple
import numpy as np

Node = Tuple[float, float]

class AgcspInstance:

    def __init__(self, grid_nodes: List[Node], obstacle_nodes: List[Node], sprayer_length: float, max_turn_angle: float):
        """
        Initializes the AgcspInstance with grid nodes, obstacle nodes, and sprayer length.
        
        grid_nodes: List of all nodes in the grid, including both field and obstacle nodes.
        obstacle_nodes: List of nodes that are obstacles.
        """
        
        self.grid_nodes = np.array(grid_nodes)
    
        # Convert to sets of tuples for proper comparison. This must be done in case
        # nodes are given as numpy arrays.
        grid_set = set(map(tuple, grid_nodes))
        obstacle_set = set(map(tuple, obstacle_nodes))
        
        # Validate obstacles are in grid
        for obs in obstacle_set:
            if obs not in grid_set:
                raise ValueError(f"Obstacle node {obs} is not in grid nodes.")
        
        self.obstacle_nodes = np.array(obstacle_nodes)
        
        field_set = grid_set - obstacle_set
        self.field_nodes = np.array(list(field_set))
        
        self.sprayer_length = sprayer_length
        self.max_turn_angle = max_turn_angle
        
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
        