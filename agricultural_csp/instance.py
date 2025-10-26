from typing import List, Tuple
from scipy.ndimage import distance_transform_edt
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
        self.obstacle_nodes = np.array(obstacle_nodes)
        self.sprayer_length = sprayer_length
        self.max_turn_angle = max_turn_angle
    
        # Convert to sets of tuples for proper comparison. This must be done in case
        # nodes are given as numpy arrays.
        grid_set = set(map(tuple, grid_nodes))
        obstacle_set = set(map(tuple, obstacle_nodes))
        for obs in obstacle_set:
            if obs not in grid_set:
                raise ValueError(f"Obstacle node {obs} is not in grid nodes.")
        
        self._original_field_nodes = np.array(list(grid_set - obstacle_set))
        
        self._perform_precomputations()

    def _perform_precomputations(self):
        """Run all preprocessing steps."""
        self._compute_bounding_box_and_masks()
        self._compute_visitable_area()
        self._compute_coverable_area()
        self._filter_field_nodes()
        self._update_obstacles_with_removed_nodes()

    def _compute_bounding_box_and_masks(self):
        """Compute bounding box, validity mask and obstacle mask."""
        self.node_count = len(self.grid_nodes)
        self.min_coords = np.min(self.grid_nodes, axis=0)
        max_coords = np.max(self.grid_nodes, axis=0)
        self.bounding_box_shape = tuple(max_coords - self.min_coords + 1)

        self.validity_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        shifted_nodes = self.grid_nodes - self.min_coords
        self.validity_mask[shifted_nodes[:, 0], shifted_nodes[:, 1]] = True

        self.obstacle_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        if len(self.obstacle_nodes) > 0:
            shifted_obstacles = self.obstacle_nodes - self.min_coords
            self.obstacle_mask[shifted_obstacles[:, 0], shifted_obstacles[:, 1]] = True

    def _compute_visitable_area(self):
        """Compute which nodes can be visited by the sprayer."""
        R = self.sprayer_length / 2.0
        if np.any(self.obstacle_mask):
            distances_to_obstacle = distance_transform_edt(~self.obstacle_mask)
            self._visitable_mask = self.validity_mask & (distances_to_obstacle > R)
        else:
            self._visitable_mask = self.validity_mask.copy()

    def _compute_coverable_area(self):
        """Compute which nodes can be covered by the sprayer from visitable nodes."""
        edt_input_mask = ~self._visitable_mask
        distance_to_visitable_area = distance_transform_edt(edt_input_mask)
        self._coverable_mask = distance_to_visitable_area <= (self.sprayer_length / 2.0)

    def _filter_field_nodes(self):
        """Filter original field nodes to keep only coverable ones."""
        original_shifted = self._original_field_nodes - self.min_coords
        is_coverable = self._coverable_mask[original_shifted[:, 0], original_shifted[:, 1]]
        self.field_nodes = self._original_field_nodes[is_coverable]

        self.target_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        if len(self.field_nodes) > 0:
            shifted_targets = self.field_nodes - self.min_coords
            self.target_mask[shifted_targets[:, 0], shifted_targets[:, 1]] = True
        self.target_node_count = len(self.field_nodes)

        self._removed_nodes = np.array(list(set(map(tuple, self._original_field_nodes)) - set(map(tuple, self.field_nodes))))

    def _update_obstacles_with_removed_nodes(self):
        """Add removed nodes as obstacles to simplify future computations."""
        if len(self._removed_nodes) == 0:
            return

        if self.obstacle_nodes.size == 0:
            self.obstacle_nodes = self._removed_nodes
        else:
            self.obstacle_nodes = np.vstack([self.obstacle_nodes, self._removed_nodes])

        shifted_removed = self._removed_nodes - self.min_coords
        self.obstacle_mask[shifted_removed[:, 0], shifted_removed[:, 1]] = True
