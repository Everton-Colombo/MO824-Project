from scipy.ndimage import distance_transform_edt
from skimage.draw import line
import numpy as np

from instance import *
from solution import *

class AgcspEvaluator:
    
    def __init__(self, instance: AgcspInstance):
        self.instance = instance
        self.alpha = 1.0  # Weight for coverage proportion
        self.beta = 1.0   # Weight for travelled distance
        self.gamma = 1.0  # Weight for maneuver complexity penalty
    
    @cache_on_solution
    def objfun(self, solution: AgcspSolution) -> float:
        coverage_proportion = self.coverage_proportion(solution)
        travelled_distance = self.path_length(solution)
        manouver_complexity_penalty = self.manouver_complexity_penalty(solution)

        return (self.alpha * (1 - coverage_proportion) + self.beta * travelled_distance + self.gamma * manouver_complexity_penalty)

    @cache_on_solution
    def path_length(self, solution: AgcspSolution) -> float:
        dist = 0.0
        for i in range(len(solution.path) - 1):
            u, v = solution.path[i], solution.path[i + 1]
            dist += np.linalg.norm(u - v)

        return dist
    
    @cache_on_solution
    def manouver_complexity_penalty(self, solution: AgcspSolution) -> float:
        """
        Calculates the maneuver complexity penalty based on the angles between consecutive path segments.
        The penalty is higher for sharper turns (larger angles).
        
        Formula: P_M(S) = sum(1 + cos(theta_i)) for i from 2 to k-1
        where theta_i is the angle at point i between segments (i-1, i) and (i, i+1)
        """
        if len(solution.path) < 3:
            return 0.0
        
        penalty = 0.0
        for i in range(1, len(solution.path) - 1):
            p_prev = np.array(solution.path[i - 1])
            p_curr = np.array(solution.path[i])
            p_next = np.array(solution.path[i + 1])
            
            # Calculate vectors
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # Calculate norms
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            penalty += 1 - cos_theta
            
        return penalty

    @cache_on_solution
    def coverage_proportion(self, solution: AgcspSolution) -> float:
        """
        Efficiently calculates the proportion of covered nodes.
        Also sets a flag in the solution if any obstacle is hit.
        """

        final_coverage_mask = self._get_coverage_mask(solution.path)
        solution.cache['hits_obstacle'] = np.any(final_coverage_mask & self.instance.obstacle_mask)
        
        num_covered_nodes = np.sum(final_coverage_mask)
        return num_covered_nodes / self.instance.node_count
    
    def get_covered_nodes_list(self, path_points: List[Node]) -> np.ndarray:
        """
        Calculates the covered nodes for a given path.
        This is method is more for visualization/debugging purposes, as it returns the actual list of covered nodes.
        For the optimization process, use calculate_coverage_proportion instead.
        """

        final_coverage_mask = self._get_coverage_mask(path_points)

        # Convert covered indices back to the original coordinate system
        covered_indices_shifted = np.argwhere(final_coverage_mask)
        covered_nodes = covered_indices_shifted + self.instance.min_coords

        return covered_nodes

    def _get_coverage_mask(self, path_points: List[Node]) -> np.ndarray:
        """
        Calculates the coverage mask for a given path.
        This is done using a distance transform approach, which requires a rectangular grid.
        """
        path_arr = np.array(path_points)
        shifted_path = path_arr - self.instance.min_coords

        rectangular_coverage = self._get_rectangular_coverage(
            self.instance.bounding_box_shape,
            shifted_path,
            self.instance.sprayer_length
        )

        final_coverage_mask = rectangular_coverage & self.instance.validity_mask
        return final_coverage_mask
    
    @staticmethod
    def _get_rectangular_coverage(grid_shape, path_points, sprayer_length):
        """
        Helper function for the core distance transform calculation.
        """
        path_grid = np.zeros(grid_shape, dtype=bool)
        # Mark the path on the grid
        for i in range(len(path_points) - 1):
            p1, p2 = path_points[i], path_points[i+1]
            rr, cc = line(r0=p1[0], c0=p1[1], r1=p2[0], c1=p2[1])
            valid_idx = (rr >= 0) & (rr < grid_shape[0]) & (cc >= 0) & (cc < grid_shape[1])
            path_grid[rr[valid_idx], cc[valid_idx]] = True
        
        distances = distance_transform_edt(~path_grid)
        return distances <= (sprayer_length / 2.0)

    def evaluate_removal_delta(self, solution: AgcspSolution, node_idx: int) -> float:
        """ Calculates the delta when removing one node. """
        path = np.array(solution.path)
        if node_idx < 0 or node_idx >= len(path):
            raise ValueError("Node index for removal is out of bounds.")

        old_cost = self.objfun(solution)
        old_distance = solution.cache.get("path_length", 0.0)
        old_mcp = solution.cache.get("manouver_complexity_penalty", 0.0)

        delta_distance = 0.0
        if len(path) <= 1:
            new_distance = 0.0
        elif node_idx == 0:
            delta_distance = -np.linalg.norm(path[0] - path[1])
        elif node_idx == len(path) - 1:
            delta_distance = -np.linalg.norm(path[-2] - path[-1])
        else:
            p_prev, p_curr, p_next = path[node_idx - 1], path[node_idx], path[node_idx + 1]
            dist_removed = np.linalg.norm(p_prev - p_curr) + np.linalg.norm(p_curr - p_next)
            dist_added = np.linalg.norm(p_prev - p_next)
            delta_distance = dist_added - dist_removed
        new_distance = old_distance + delta_distance

        delta_mcp = 0.0
        if node_idx > 0 and node_idx < len(path) - 1:
            delta_mcp -= self._calculate_angle_penalty_at_node(path[node_idx-1], path[node_idx], path[node_idx+1])
        if node_idx > 1:
            delta_mcp -= self._calculate_angle_penalty_at_node(path[node_idx-2], path[node_idx-1], path[node_idx])
            if node_idx < len(path) - 1:
                delta_mcp += self._calculate_angle_penalty_at_node(path[node_idx-2], path[node_idx-1], path[node_idx+1])
        if node_idx < len(path) - 2:
            delta_mcp -= self._calculate_angle_penalty_at_node(path[node_idx], path[node_idx+1], path[node_idx+2])
            if node_idx > 0:
                delta_mcp += self._calculate_angle_penalty_at_node(path[node_idx-1], path[node_idx+1], path[node_idx+2])
        new_mcp = old_mcp + delta_mcp
        
        new_path_for_coverage = np.delete(path, node_idx, axis=0)
        temp_solution = AgcspSolution(new_path_for_coverage)
        new_coverage = self.coverage_proportion(temp_solution)

        new_cost = (self.alpha * (1 - new_coverage) + self.beta * new_distance + self.gamma * new_mcp)
                    
        return new_cost - old_cost

    def evaluate_swap_delta(self, solution: AgcspSolution, idx1: int, idx2: int) -> float:
        """ Calculates the delta when swapping two nodes. """
        path = np.array(solution.path)
        if idx1 < 0 or idx1 >= len(solution.path) or idx2 < 0 or idx2 >= len(solution.path):
            raise ValueError("Node indices are out of bounds.")
        if idx1 == idx2:
            return 0.0

        old_cost = self.objfun(solution)
        old_distance = solution.cache.get("path_length", 0.0)

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        p1, p2 = path[idx1], path[idx2]

        p1_prev = path[idx1 - 1] if idx1 > 0 else None
        p1_next = path[idx1 + 1] if idx1 < len(path) - 1 else None

        p2_prev = path[idx2 - 1] if idx2 > 0 else None
        p2_next = path[idx2 + 1] if idx2 < len(path) - 1 else None

        dist_removed = 0
        if p1_prev is not None: dist_removed += np.linalg.norm(p1_prev - p1)
        if p1_next is not None: dist_removed += np.linalg.norm(p1 - p1_next)
        if p2_prev is not None and p2_prev is not p1: dist_removed += np.linalg.norm(p2_prev - p2)
        if p2_next is not None: dist_removed += np.linalg.norm(p2 - p2_next)

        dist_added = 0
        if p1_prev is not None: dist_added += np.linalg.norm(p1_prev - p2)
        if p1_next is not None: dist_added += np.linalg.norm(p2 - p1_next)
        if p2_prev is not None and p2_prev is not p1: dist_added += np.linalg.norm(p2_prev - p1)
        if p2_next is not None: dist_added += np.linalg.norm(p1 - p2_next)

        if idx2 == idx1 + 1:
            dist_removed = (np.linalg.norm(p1_prev - p1) if p1_prev is not None else 0) + np.linalg.norm(p1 - p2) + (np.linalg.norm(p2 - p2_next) if p2_next is not None else 0)
            dist_added = (np.linalg.norm(p1_prev - p2) if p1_prev is not None else 0) + np.linalg.norm(p2 - p1) + (np.linalg.norm(p1 - p2_next) if p2_next is not None else 0)

        delta_distance = dist_added - dist_removed
        new_distance = old_distance + delta_distance

        new_path = path.copy()
        new_path[idx1], new_path[idx2] = new_path[idx2].copy(), new_path[idx1].copy()
        
        temp_solution = AgcspSolution(new_path)
        new_coverage = self.coverage_proportion(temp_solution)
        new_mcp = self.manouver_complexity_penalty(temp_solution)

        new_cost = (self.alpha * (1 - new_coverage) +
                    self.beta * new_distance +
                    self.gamma * new_mcp)
                    
        return new_cost - old_cost