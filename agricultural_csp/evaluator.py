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
    
    #region Objective function (and its components) evaluation methods:
    @cache_on_solution
    def objfun(self, solution: AgcspSolution) -> float:
        coverage_proportion = self.coverage_proportion(solution)
        travelled_distance = self.path_length(solution)
        manouver_complexity_penalty = self.manouver_complexity_penalty(solution)
        
        objfun =  (self.alpha * (1 - coverage_proportion) +
                   self.beta * travelled_distance +
                   self.gamma * manouver_complexity_penalty)
        return objfun

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

    #region Coverage evaluation methods:
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
    #endregion
    
    #endregion

    #region Neighborhood steps evaluation methods:
    


    def evaluate_removal_delta(self, solution: AgcspSolution, node_idx: int) -> float:
        # Item a) Removal Method
        if node_idx < 0 or node_idx >= len(solution.path):
            raise ValueError("Node index is out of bounds.")

        old_distance = solution.cache.get('path_length')
        old_coverage = solution.cache.get('coverage_proportion')
        old_mcp = solution.cache.get('manouver_complexity_penalty')

        new_distance, new_coverage, new_mcp = None, None, None

        if old_distance is None or old_coverage is None or old_mcp is None:
            raise ValueError("Solution must have been evaluated before calling this method.")
        
        node = solution.path[node_idx]
        prev_node = solution.path[node_idx - 1] if node_idx > 0 else None
        next_node = solution.path[node_idx + 1] if node_idx < len(solution.path) - 1 else None

        # Calculate new distance
        
        if prev_node is None and next_node is None:
            new_distance = 0.0
            new_coverage = 0.0
            new_mcp = 0.0
        elif prev_node is None:
            new_distance = old_distance - np.linalg.norm(node - next_node)
            new_coverage = self.coverage_proportion(AgcspSolution(solution.path[1:]))
            new_mcp = old_mcp - (1 - np.dot((next_node - node) / np.linalg.norm(next_node - node), (solution.path[2] - next_node) / np.linalg.norm(solution.path[2] - next_node))) if len(solution.path) > 2 else 0.0
        elif next_node is None:
            new_distance = old_distance - np.linalg.norm(prev_node - node)
            new_coverage = self.coverage_proportion(AgcspSolution(solution.path[:-1]))
            new_mcp = old_mcp - (1 - np.dot((node - prev_node) / np.linalg.norm(node - prev_node), (prev_node - solution.path[-3]) / np.linalg.norm(prev_node - solution.path[-3]))) if len(solution.path) > 2 else 0.0
        else:
            new_distance = (old_distance 
                            - np.linalg.norm(prev_node - node) 
                            - np.linalg.norm(node - next_node) 
                            + np.linalg.norm(prev_node - next_node))
            new_coverage = self.coverage_proportion(AgcspSolution(solution.path[:node_idx] + solution.path[node_idx+1:]))
            if len(solution.path) > 2:
                old_angle_penalty = (1 - np.dot((node - prev_node) / np.linalg.norm(node - prev_node), 
                                               (next_node - node) / np.linalg.norm(next_node - node)))
                new_angle_penalty = (1 - np.dot((next_node - prev_node) / np.linalg.norm(next_node - prev_node), 
                                               (solution.path[node_idx + 2] - next_node) / np.linalg.norm(solution.path[node_idx + 2] - next_node))) if node_idx + 2 < len(solution.path) else 0.0
                new_mcp = old_mcp - old_angle_penalty + new_angle_penalty
            else:
                new_mcp = 0.0

        new_objfun = self.alpha * new_distance + self.beta * new_coverage + self.gamma * new_mcp
        return new_objfun - solution.cache.get('objfun')

    def evaluate_edge_insertion_delta(self, solution: AgcspSolution, node: int, position: int) -> float:
        """
        Item b) Insertion Method.
        Takes a node that is already in the path and evaluates the cost of removing it from its current position
        and inserting it at the specified position.
        """
        
        if position < 0 or position > len(solution.path):
            raise ValueError("Position is out of bounds.")
        
        original_index = solution.path.index(node) # This will raise ValueError if not found
        
        if original_index == position or original_index == position - 1:
            return 0.0
        
        # Calculate cost of removing node from its original position
        if original_index == 0:
            removal_cost = self.instance._distances[node][solution.path[1]]
        elif original_index == len(solution.path) - 1:
            removal_cost = self.instance._distances[solution.path[-2]][node]
        else:
            removal_cost = (self.instance._distances[solution.path[original_index - 1]][node] +
                            self.instance._distances[node][solution.path[original_index + 1]] -
                            self.instance._distances[solution.path[original_index - 1]][solution.path[original_index + 1]])
        
        # Calculate cost of inserting node at the new position
        if position == 0:
            insertion_cost = self.instance._distances[node][solution.path[0]]
        elif position == len(solution.path):
            insertion_cost = self.instance._distances[solution.path[-1]][node]
        else:
            insertion_cost = (self.instance._distances[solution.path[position - 1]][node] +
                              self.instance._distances[node][solution.path[position]] -
                              self.instance._distances[solution.path[position - 1]][solution.path[position]])
        
        return removal_cost + insertion_cost
    #endregion