from typing import List, Tuple
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.spatial import KDTree

Node = Tuple[float, float]

class AgcspInstance:

    def __init__(self, grid_nodes: List[Node], obstacle_nodes: List[Node], sprayer_length: float, max_turn_angle: float, adaptive_sampling: bool = True):
        """
        Initializes the AgcspInstance with grid nodes, obstacle nodes, and sprayer length.
        
        grid_nodes: List of all nodes in the grid, including both field and obstacle nodes.
        obstacle_nodes: List of nodes that are obstacles.
        """
        
        self.grid_nodes_original = np.array(grid_nodes)
        self.obstacle_nodes_original = np.array(obstacle_nodes)

        self.sprayer_length = sprayer_length
        self.max_turn_angle = max_turn_angle

        self.adaptive_sampling = adaptive_sampling

        # Convert to sets of tuples for proper comparison. This must be done in case
        # nodes are given as numpy arrays.
        grid_set = set(map(tuple, self.grid_nodes_original))
        obstacle_set = set(map(tuple, self.obstacle_nodes_original))
        for obs in obstacle_set:
            if obs not in grid_set:
                raise ValueError(f"Obstacle node {obs} is not in grid nodes.")
        
        self._original_field_nodes = np.array(list(grid_set - obstacle_set))
        
        self._perform_precomputations()

    def _perform_precomputations(self):
        """Run all preprocessing steps."""
        self._compute_bounding_box_and_masks()
        self._compute_visitable_area_dense()
        self._compute_coverable_area()
        self._filter_field_nodes()
        self._update_obstacles_with_removed_nodes()
        self._compute_visitable_area_densa_e_kdtree()
        if self.adaptive_sampling:
            self._apply_adaptive_sampling()
        self._compute_visitable_area_esparsa_e_kdtree()

    def _compute_bounding_box_and_masks(self):
        """Compute bounding box, validity mask and obstacle mask."""
        self.node_count = len(self.grid_nodes_original)
        self.min_coords = np.min(self.grid_nodes_original, axis=0)
        max_coords = np.max(self.grid_nodes_original, axis=0)
        self.bounding_box_shape = tuple((max_coords - self.min_coords + 1).astype(int))

        self.validity_mask_dense = np.zeros(self.bounding_box_shape, dtype=bool)
        shifted_nodes = (self.grid_nodes_original - self.min_coords).astype(int)
        self.validity_mask_dense[shifted_nodes[:, 0], shifted_nodes[:, 1]] = True

        self.obstacle_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        if len(self.obstacle_nodes_original) > 0:
            shifted_obstacles = (self.obstacle_nodes_original - self.min_coords).astype(int)
            self.obstacle_mask[shifted_obstacles[:, 0], shifted_obstacles[:, 1]] = True
            
        self.validity_mask = self.validity_mask_dense.copy()
        self.grid_nodes = self.grid_nodes_original.copy()
        self.obstacle_nodes = self.obstacle_nodes_original.copy()

    def _compute_visitable_area_densa_e_kdtree(self):
        """Compute which nodes can be visited by the sprayer."""
        R = self.sprayer_length / 2.0
        
        if np.any(self.obstacle_mask):
            distances_to_obstacle = distance_transform_edt(~self.obstacle_mask)
            self._visitable_mask_dense = self.validity_mask_dense & (distances_to_obstacle > R)
        else:
            self._visitable_mask_dense = self.validity_mask_dense.copy()

        visitable_indices_dense = np.argwhere(self._visitable_mask_dense)
        self.visitable_nodes_array_dense = visitable_indices_dense + self.min_coords

        if self.visitable_nodes_array_dense.size > 0:
            self.visitable_kdtree_dense = KDTree(self.visitable_nodes_array_dense)
        else:
            self.visitable_kdtree_dense = None

    def _compute_visitable_area_esparsa_e_kdtree(self):
        """Compute which SPARSE nodes can be visited."""
        R = self.sprayer_length / 2.0
        
        if np.any(self.obstacle_mask):
            distances_to_obstacle = distance_transform_edt(~self.obstacle_mask)
            self._visitable_mask_esparsa = self.validity_mask & (distances_to_obstacle > R)
        else:
            self._visitable_mask_esparsa = self.validity_mask.copy()

        visitable_indices_esparsos = np.argwhere(self._visitable_mask_esparsa)
    
        self.visitable_nodes_array_esparso = visitable_indices_esparsos + self.min_coords

        if self.visitable_nodes_array_esparso.size > 0:
            self.visitable_kdtree_esparso = KDTree(self.visitable_nodes_array_esparso)
        else:
            self.visitable_kdtree_esparso = None

    def _compute_visitable_area_dense(self):
        """Calculates the DENSA _visitable_mask for coverage calculation."""
        R = self.sprayer_length / 2.0
        if np.any(self.obstacle_mask):
            distances_to_obstacle = distance_transform_edt(~self.obstacle_mask)
            self._visitable_mask = self.validity_mask_dense & (distances_to_obstacle > R)
        else:
            self._visitable_mask = self.validity_mask_dense.copy()

    def _compute_coverable_area(self):
        """Compute which nodes can be covered by the sprayer from visitable nodes."""
        edt_input_mask = ~self._visitable_mask
        distance_to_visitable_area = distance_transform_edt(edt_input_mask)
        self._coverable_mask = distance_to_visitable_area <= (self.sprayer_length / 2.0)

    def _filter_field_nodes(self):
        """Filter original field nodes to keep only coverable ones."""
        original_shifted = (self._original_field_nodes - self.min_coords).astype(int)
        
        valid_indices_mask = (original_shifted[:, 0] >= 0) & (original_shifted[:, 0] < self.bounding_box_shape[0]) & \
                             (original_shifted[:, 1] >= 0) & (original_shifted[:, 1] < self.bounding_box_shape[1])
        
        original_shifted_valid = original_shifted[valid_indices_mask]
        original_field_nodes_valid = self._original_field_nodes[valid_indices_mask]

        is_coverable = self._coverable_mask[original_shifted_valid[:, 0], original_shifted_valid[:, 1]]
        self.field_nodes = original_field_nodes_valid[is_coverable]

        self.target_mask = np.zeros(self.bounding_box_shape, dtype=bool)
        if len(self.field_nodes) > 0:
            shifted_targets = (self.field_nodes - self.min_coords).astype(int)
            self.target_mask[shifted_targets[:, 0], shifted_targets[:, 1]] = True
        self.target_node_count = len(self.field_nodes)

        removed_nodes_set = set(map(tuple, self._original_field_nodes)) - set(map(tuple, self.field_nodes))
        self._removed_nodes = np.array(list(removed_nodes_set))

    def _update_obstacles_with_removed_nodes(self):
        """Add removed nodes as obstacles to simplify future computations."""
        if len(self._removed_nodes) == 0:
            return

        if self.obstacle_nodes.size == 0:
            self.obstacle_nodes = self._removed_nodes
        else:
            self.obstacle_nodes = np.vstack([self.obstacle_nodes_original, self._removed_nodes])

        shifted_removed = (self._removed_nodes - self.min_coords).astype(int)
        
        valid_indices_mask = (shifted_removed[:, 0] >= 0) & (shifted_removed[:, 0] < self.bounding_box_shape[0]) & \
                             (shifted_removed[:, 1] >= 0) & (shifted_removed[:, 1] < self.bounding_box_shape[1])
        
        shifted_removed_valid = shifted_removed[valid_indices_mask]
        
        if shifted_removed_valid.size > 0:
            self.obstacle_mask[shifted_removed_valid[:, 0], shifted_removed_valid[:, 1]] = True

    def _apply_adaptive_sampling(self):
        """
        Filters the self.validity_mask to create a non-uniform grid.
        Maintains high density near obstacles and low density (sampled)
        in open areas.
        """

        original_node_count = np.sum(self.validity_mask_dense)
        print(f"Grid Adaptativo: Densidade original ({original_node_count})")

        R = self.sprayer_length / 2.0

        obstacle_safety_radius = R * 1.5
        if np.any(self.obstacle_mask):
            distances_to_obstacle = distance_transform_edt(~self.obstacle_mask)
            obstacle_high_density_mask = (distances_to_obstacle <= obstacle_safety_radius) & self.validity_mask_dense
        else:
            obstacle_high_density_mask = np.zeros_like(self.validity_mask_dense, dtype=bool)

        shape = self.validity_mask_dense.shape
        border_edges_mask = np.zeros(shape, dtype=bool)
        border_edges_mask[0, :] = True
        border_edges_mask[-1, :] = True
        border_edges_mask[:, 0] = True
        border_edges_mask[:, -1] = True

        distances_to_border = distance_transform_edt(~border_edges_mask)
        
        border_turn_radius = R * 1.0 
        
        border_high_density_mask = (distances_to_border <= border_turn_radius) & self.validity_mask_dense
        
        high_density_mask = obstacle_high_density_mask | border_high_density_mask

        step_size = int(max(2, R / 2.0))
        shape = self.validity_mask_dense.shape
        low_density_mask = np.zeros(shape, dtype=bool)

        low_density_mask[::step_size, ::step_size] = True

        low_density_mask &= ~high_density_mask
        
        low_density_mask &= self.validity_mask_dense

        self.validity_mask = high_density_mask | low_density_mask

        new_indices = np.argwhere(self.validity_mask)
        self.grid_nodes = new_indices + self.min_coords
        self.node_count = len(self.grid_nodes)

        print(f"Grid Adaptativo: Nova densidade (reduzida) ({self.node_count} nós)")
