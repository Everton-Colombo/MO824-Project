from scipy.ndimage import distance_transform_edt
from skimage.draw import line
import numpy as np

from instance import *
from solution import *

class AgcspEvaluator:
    
    def __init__(self, instance: AgcspInstance):
        self.instance = instance
    
    #region Objective function (and its components) evaluation methods:
    @cache_on_solution
    def evaluate_objfun(self, solution: AgcspSolution) -> float:
        alpha, beta, gamma = 1.0, 1.0, 1.0 # Weights for the three components
        
        coverage_proportion = self.calculate_coverage_proportion(solution)
        travelled_distance = self._calculate_travelled_distance(solution)
        manouver_complexity_penalty = self._calculate_manouver_complexity_penalty(solution)
        
        objfun =  (alpha * (1 - coverage_proportion) +
                    beta * travelled_distance +
                    gamma * manouver_complexity_penalty)
        return objfun

    @cache_on_solution
    def _calculate_travelled_distance(self, solution: AgcspSolution) -> float:
        dist = 0.0
        for i in range(len(solution.path) - 1):
            u, v = solution.path[i], solution.path[i + 1]
            dist += np.linalg.norm(np.array(u) - np.array(v))    # TODO: Address conversion overhead

        return dist
    
    @cache_on_solution
    def _calculate_manouver_complexity_penalty(self, solution: AgcspSolution) -> float:
        """
        Calculates the maneuver complexity penalty based on the angles between consecutive path segments.
        The penalty is higher for sharper turns (larger angles).
        
        Formula: P_M(S) = sum(1 + cos(theta_i)) for i from 2 to k-1
        where theta_i is the angle at point i between segments (i-1, i) and (i, i+1)
        """

        if 'manouver_complexity_penalty' in solution.cache:
            return solution.cache['manouver_complexity_penalty']

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
            
            # Avoid division by zero (shouldn't happen with valid paths, but just in case)
            if norm_v1 > 0 and norm_v2 > 0:
                cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                penalty += 1 - cos_theta

        solution._cache['manouver_complexity_penalty'] = penalty
        return penalty

    #region Coverage evaluation methods:
    @cache_on_solution
    def calculate_coverage_proportion(self, solution: AgcspSolution) -> float:
        """
        Efficiently calculates the proportion of covered nodes.
        Also sets a flag in the solution if any obstacle is hit.
        """

        final_coverage_mask = self._get_coverage_mask(solution.path)
        solution._hits_obstacle = np.any(final_coverage_mask & self.instance.obstacle_mask)
        
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
    
    def evaluate_removal_delta(self, solution: AgcspSolution, node: int) -> float:
        # Item a) Removal Method
        if node not in solution.path:
            raise ValueError("Node to be removed is not in the solution path.")
        
        idx = solution.path.index(node)
        
        if idx == 0 or idx == len(solution.path) - 1:
            # If it's the first or last node, just remove it
            if idx == 0:
                # First node
                new_distance = solution.travelled_distance - self.instance._distances[node][solution.path[1]]
            else:
                # Last node
                new_distance = solution.travelled_distance - self.instance._distances[solution.path[-2]][node]
        else:
            # Remove the node and add the distance between its neighbors
            new_distance = solution.travelled_distance
            new_distance -= self.instance._distances[solution.path[idx - 1]][node]
            new_distance -= self.instance._distances[node][solution.path[idx + 1]]
            new_distance += self.instance._distances[solution.path[idx - 1]][solution.path[idx + 1]]

        return solution.travelled_distance - new_distance
    
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

    def is_feasible(self, solution: AgcspSolution) -> bool:
        covered = set()
        for node in solution.path:
            covered.update(self.instance.coverage.get(node, {node}))
        return len(covered) == len(self.instance.grid_nodes)
    #endregion