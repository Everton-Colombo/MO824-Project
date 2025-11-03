from scipy.ndimage import distance_transform_edt
from skimage.draw import line
import numpy as np

from .instance import *
from .solution import *

class AgcspEvaluator:
    
    def __init__(self, instance: AgcspInstance):
        self.instance = instance
        self.alpha = self.instance.node_count  # Weight for coverage proportion
        self.beta = 1.0   # Weight for travelled distance
        self.gamma = 1.0  # Weight for maneuver complexity penalty
    
    #region Objective Function and Components
    @cache_on_solution
    def objfun(self, solution: AgcspSolution) -> float:
        coverage_proportion = self.coverage_proportion(solution)
        travelled_distance = self.path_length(solution)
        manouver_complexity_penalty = self.manouver_complexity_penalty(solution)

        return (self.alpha * (1 - coverage_proportion) + self.beta * travelled_distance + self.gamma * manouver_complexity_penalty)

    def objfun_components(self, solution: AgcspSolution) -> tuple[float, float, float]:
        """
        Returns the three components of the objective function separately.
        Returns: (coverage_penalty, distance, maneuver_penalty)
        """
        coverage_proportion = self.coverage_proportion(solution)
        travelled_distance = self.path_length(solution)
        manouver_complexity_penalty = self.manouver_complexity_penalty(solution)
        
        coverage_penalty = self.alpha * (1 - coverage_proportion)
        distance_penalty = self.beta * travelled_distance
        maneuver_penalty = self.gamma * manouver_complexity_penalty
        
        return (coverage_penalty, distance_penalty, maneuver_penalty)

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
        
        Formula: P_M(S) = sum(1 - cos(theta_i)) for i from 2 to k-1
        where theta_i is the angle at point i between segments (i-1, i) and (i, i+1)
        """
        if len(solution.path) < 3:
            return 0.0
        
        penalty = 0.0
        for i in range(1, len(solution.path) - 1):
            p_prev = np.array(solution.path[i - 1])
            p_curr = np.array(solution.path[i])
            p_next = np.array(solution.path[i + 1])
            
            penalty += self._calculate_angle_penalty_at_node(p_prev, p_curr, p_next)
            
        return penalty

    @cache_on_solution
    def coverage_proportion(self, solution: AgcspSolution) -> float:
        """
        Efficiently calculates the proportion of covered nodes.
        Also sets a flag in the solution if any obstacle is hit.
        """
        
        if len(solution.path) == 0:
            solution.cache['hits_obstacle'] = False
            return 0.0

        final_coverage_mask = self._coverage_mask(solution)
        solution.cache['hits_obstacle'] = np.any(final_coverage_mask & self.instance.obstacle_mask)
        
        num_covered_nodes = np.sum(final_coverage_mask)
        return num_covered_nodes / len(self.instance.field_nodes)
    
    @cache_on_solution
    def hits_obstacle(self, solution: AgcspSolution) -> bool:
        if len(solution.path) == 0:
            return False
                
        final_coverage_mask = self._coverage_mask(solution)
        return np.any(final_coverage_mask & self.instance.obstacle_mask)

    def get_covered_nodes_list(self, path_points: List[Node]) -> np.ndarray:
        """
        Calculates the covered nodes for a given path.
        This is method is more for visualization/debugging purposes, as it returns the actual list of covered nodes.
        For the optimization process, use calculate_coverage_proportion instead.
        """

        coverage_mask_area = self._coverage_mask(path_points)
        covered_target_nodes_mask = coverage_mask_area & self.instance.target_mask
        covered_indices_shifted = np.argwhere(covered_target_nodes_mask)
        covered_nodes_coords = covered_indices_shifted + self.instance.min_coords

        return covered_nodes_coords

    @cache_on_solution
    def _coverage_mask(self, solution: AgcspSolution | List[Node]) -> np.ndarray:
        """
        Calculates the coverage mask for a given path.
        This is done using a distance transform approach, which requires a rectangular grid.
        """
        if isinstance(solution, AgcspSolution):
            path_arr = np.array(solution.path, dtype=int)
        else:
            path_arr = np.array(solution, dtype=int)

        if path_arr.size == 0:
            return np.zeros(self.instance.bounding_box_shape, dtype=bool)

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
    #endregion

    #region Path Validation    
    def _path_segment_hits_obstacle(self, p_start: Node, p_end: Node) -> bool:
        """
        Checks if the path segment between p_start and p_end collides with any obstacles, considering the width of the sprayer.
        """

        if np.array_equal(p_start, p_end):
            return True

        segment_path = np.array([p_start, p_end], dtype=int)
        shifted_segment = segment_path - self.instance.min_coords

        segment_coverage_mask = self._get_rectangular_coverage(
            self.instance.bounding_box_shape,
            shifted_segment,
            self.instance.sprayer_length
        )

        collision_mask = segment_coverage_mask & self.instance.obstacle_mask
        
        if np.any(collision_mask):
            return False

        return True
    
    #endregion

    #region Delta Evaluation Methods
    def evaluate_removal_delta(self, solution: AgcspSolution, node_idx: int, 
                               return_components: bool = False) -> float | tuple[float, float, float]:
        """ 
        Calculates the delta when removing one node.
        
        Args:
            solution: The current solution
            node_idx: The index of the node to remove
            return_components: If True, returns (coverage_delta, distance_delta, maneuver_delta)
                             If False, returns only total_delta (backward compatible)
        
        Returns:
            If return_components=False: float (total delta)
            If return_components=True: tuple of (coverage_delta, distance_delta, maneuver_delta)
        """
        path = np.array(solution.path)
        if node_idx < 0 or node_idx >= len(path):
            raise ValueError("Node index for removal is out of bounds.")

        # Check if removal creates a path segment that hits an obstacle
        if len(path) > 2 and 0 < node_idx < len(path) - 1:
            p_prev = path[node_idx - 1]
            p_next = path[node_idx + 1]
            # If the new direct segment would hit an obstacle, don't allow this removal
            if not self._path_segment_hits_obstacle(p_prev, p_next):
                inf_result = float('inf')
                return (inf_result, inf_result, inf_result) if return_components else inf_result

        # Get old values
        old_coverage_proportion = self.coverage_proportion(solution)
        old_distance = solution.cache.get("path_length", 0.0)
        old_mcp = solution.cache.get("manouver_complexity_penalty", 0.0)

        # Calculate distance delta
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

        max_angle_penalty = 1 - np.cos(np.radians(self.instance.max_turn_angle))

        # Calculate maneuver complexity delta
        delta_mcp = 0.0
        if node_idx > 0 and node_idx < len(path) - 1:
            delta_mcp -= self._calculate_angle_penalty_at_node(path[node_idx-1], path[node_idx], path[node_idx+1])
        if node_idx > 1:
            delta_mcp -= self._calculate_angle_penalty_at_node(path[node_idx-2], path[node_idx-1], path[node_idx])
            if node_idx < len(path) - 1:
                new_angle_prev = self._calculate_angle_penalty_at_node(path[node_idx-2], path[node_idx-1], path[node_idx+1])
                if new_angle_prev > max_angle_penalty:
                    inf_result = float('inf')
                    return (inf_result, inf_result, inf_result) if return_components else inf_result
                delta_mcp += new_angle_prev
        if node_idx < len(path) - 2:
            delta_mcp -= self._calculate_angle_penalty_at_node(path[node_idx], path[node_idx+1], path[node_idx+2])
            if node_idx > 0:
                new_angle_next = self._calculate_angle_penalty_at_node(path[node_idx-1], path[node_idx+1], path[node_idx+2])
                if new_angle_next > max_angle_penalty:
                    inf_result = float('inf')
                    return (inf_result, inf_result, inf_result) if return_components else inf_result
                delta_mcp += new_angle_next
        new_mcp = old_mcp + delta_mcp
        
        # Calculate coverage delta
        new_path_for_coverage = np.delete(path, node_idx, axis=0)
        temp_solution = AgcspSolution(new_path_for_coverage)
        new_coverage_proportion = self.coverage_proportion(temp_solution)
        delta_coverage_proportion = new_coverage_proportion - old_coverage_proportion

        # Calculate component deltas (with weights applied)
        delta_coverage_penalty = self.alpha * (-delta_coverage_proportion)  # Note: objfun uses (1 - coverage)
        delta_distance_penalty = self.beta * delta_distance
        delta_maneuver_penalty = self.gamma * delta_mcp
                    
        if return_components:
            return (delta_coverage_penalty, delta_distance_penalty, delta_maneuver_penalty)
        else:
            # Total delta
            return delta_coverage_penalty + delta_distance_penalty + delta_maneuver_penalty

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

    def _calculate_angle_penalty_at_node(self, p_prev: Node, p_curr: Node, p_next: Node) -> float:
        """ Calculates the maneuver penalty for a single node in the middle of a path. """
        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return 1 - cos_theta


    def evaluate_insertion_delta(self, solution: AgcspSolution, node_to_insert: Node, position: int, 
                                  return_components: bool = False) -> float | tuple[float, float, float]:
        """ 
        Calculates the delta in the objective function when inserting a new node.
        
        Args:
            solution: The current solution
            node_to_insert: The node to insert
            position: The position to insert the node
            return_components: If True, returns (coverage_delta, distance_delta, maneuver_delta)
                             If False, returns only total_delta (backward compatible)
        
        Returns:
            If return_components=False: float (total delta)
            If return_components=True: tuple of (coverage_delta, distance_delta, maneuver_delta)
        """
        if position < 0 or position > len(solution.path):
            raise ValueError("Insertion position is out of bounds.")
        
        path = np.array(solution.path)
        node_to_insert = np.array(node_to_insert)
        
        # Check if insertion creates path segments that hit obstacles
        if position > 0:
            p_prev = path[position - 1]
            # Check segment from previous node to inserted node
            if not self._path_segment_hits_obstacle(p_prev, node_to_insert):
                inf_result = float('inf')
                return (inf_result, inf_result, inf_result) if return_components else inf_result
        
        if position < len(path):
            p_next = path[position]
            # Check segment from inserted node to next node
            if not self._path_segment_hits_obstacle(node_to_insert, p_next):
                inf_result = float('inf')
                return (inf_result, inf_result, inf_result) if return_components else inf_result
        
        # Get old values
        old_coverage_proportion = self.coverage_proportion(solution)
        old_distance = solution.cache.get("path_length", 0.0)
        old_mcp = solution.cache.get("manouver_complexity_penalty", 0.0)

        # Calculate distance delta
        delta_distance = 0.0
        if len(path) == 0:
            pass
        elif position == 0:
            delta_distance = np.linalg.norm(node_to_insert - path[0])
        elif position == len(path):
            delta_distance = np.linalg.norm(path[-1] - node_to_insert)
        else:
            p_prev, p_next = path[position - 1], path[position]
            delta_distance = (
                np.linalg.norm(p_prev - node_to_insert)
                + np.linalg.norm(node_to_insert - p_next)
                - np.linalg.norm(p_prev - p_next)
            )
        new_distance = old_distance + delta_distance

        max_angle_penalty = 1 - np.cos(np.radians(self.instance.max_turn_angle))

        new_path = np.insert(path, position, node_to_insert, axis=0)

        # Calculate maneuver complexity delta
        old_local_mcp = 0.0
        if len(path) >= 3:
            if 1 <= position - 1 <= len(path) - 2:
                old_local_mcp += self._calculate_angle_penalty_at_node(
                    path[position - 2], path[position - 1], path[position]
                )
            if 1 <= position <= len(path) - 2:
                old_local_mcp += self._calculate_angle_penalty_at_node(
                    path[position - 1], path[position], path[position + 1]
                )

        new_local_mcp = 0.0
        if len(new_path) >= 3:
            for idx in (position - 1, position, position + 1):
                if 1 <= idx <= len(new_path) - 2:
                    angle_penalty = self._calculate_angle_penalty_at_node(
                        new_path[idx - 1], new_path[idx], new_path[idx + 1]
                    )
                    if angle_penalty > max_angle_penalty:
                        inf_result = float('inf')
                        return (inf_result, inf_result, inf_result) if return_components else inf_result
                    new_local_mcp += angle_penalty

        delta_mcp = new_local_mcp - old_local_mcp
        new_mcp = old_mcp + delta_mcp

        # Calculate coverage delta
        temp_solution = AgcspSolution(new_path)
        new_coverage_proportion = self.coverage_proportion(temp_solution)
        delta_coverage_proportion = new_coverage_proportion - old_coverage_proportion

        # Calculate component deltas (with weights applied)
        delta_coverage_penalty = self.alpha * (-delta_coverage_proportion)  # Note: objfun uses (1 - coverage)
        delta_distance_penalty = self.beta * delta_distance
        delta_maneuver_penalty = self.gamma * delta_mcp

        if return_components:
            return (delta_coverage_penalty, delta_distance_penalty, delta_maneuver_penalty)
        else:
            return delta_coverage_penalty + delta_distance_penalty + delta_maneuver_penalty


    def evaluate_2opt_delta(self, solution: AgcspSolution, i: int, j: int) -> float:
        """ Calculates the delta in the objective function when performing a 2-Opt move. """
        path = np.array(solution.path)
        
        if i >= j - 1 or i < 0 or j >= len(path) - 1:
            return 0.0

        old_cost = self.objfun(solution)
        old_distance = solution.cache.get("path_length", 0.0)

        p_i, p_i_plus_1 = path[i], path[i+1]
        p_j, p_j_plus_1 = path[j], path[j+1]

        dist_removed = np.linalg.norm(p_i - p_i_plus_1) + np.linalg.norm(p_j - p_j_plus_1)
        dist_added = np.linalg.norm(p_i - p_j) + np.linalg.norm(p_i_plus_1 - p_j_plus_1)
        
        delta_distance = dist_added - dist_removed
        new_distance = old_distance + delta_distance

        new_path = np.concatenate([path[:i+1], path[j:i:-1], path[j+1:]])
        temp_solution = AgcspSolution(new_path)
        new_coverage = self.coverage_proportion(temp_solution)
        
        new_mcp = self.manouver_complexity_penalty(temp_solution)

        new_cost = (self.alpha * (1 - new_coverage) + self.beta * new_distance + self.gamma * new_mcp)
                    
        return new_cost - old_cost
    #endregion
    
    def is_feasible(self, solution: AgcspSolution) -> bool:
        """ Checks if the solution is feasible (i.e., does not hit any obstacles). """
        
        # verify that no angle is too sharp (no turn sharper than max_turn_angle)
        for i in range(1, len(solution.path) - 1):
            p_prev = np.array(solution.path[i - 1])
            p_curr = np.array(solution.path[i])
            p_next = np.array(solution.path[i + 1])
            
            if self._calculate_angle_penalty_at_node(p_prev, p_curr, p_next) > (1 - np.cos(np.radians(self.instance.max_turn_angle))):
                return False

        return not self.hits_obstacle(solution) and self.coverage_proportion(solution) >= 0.98