# src/csp/evaluator.py

from src.core.evaluator import Evaluator
from .instance import CspInstance
from .solution import CspSolution
import numpy as np

class CspEvaluator(Evaluator):
    """
    Evaluates a CspSolution based on a CspInstance. It calculates the total
    cost of a tour and checks its feasibility based on node coverage.
    """
    
    def __init__(self, instance: CspInstance):
        self.instance = instance
    
    def evaluate(self, solution: CspSolution) -> float:
        """
        Calculates the total distance of the tour and checks for full coverage.
        This is the main objective function called by the Tabu Search framework.
        """
        total_distance = 0
        for i in range(len(solution.path) - 1):
            u, v = solution.path[i], solution.path[i+1]
            total_distance += self.instance.get_distance(u, v)
        solution.cost = total_distance # Note: renamed from 'distancia_total'

        solution.is_feasible = self._check_full_coverage(solution)

        if not solution.is_feasible:
            solution.cost = float('inf')

        return solution.cost

    def evaluate_removal_cost(self, solution: CspSolution, move_info: dict) -> float:
        """
        Calculates the cost delta for the Removal move (Removing Method).
        The delta is negative if the cost decreases (a good move).
        """
        node_to_remove = move_info['node']
        
        # 1. Check feasibility: if the removal makes the solution infeasible, cost is infinite.
        temp_solution = solution.copy()
        temp_solution.path.remove(node_to_remove)
        if not self._check_full_coverage(temp_solution):
            return float('inf') # Invalid move!
        
        # 2. If feasible, calculate the distance delta.
        idx = solution.path.index(node_to_remove)
        p = solution.path

        # If the node to be removed is an endpoint of the tour (which is a closed path)
        if idx == 0 or idx == len(p) - 1:
             # In a closed tour p[0] == p[-1], so removing the start/end is like removing a node
             # from the middle, considering the connection p[-2] -> p[1]
             delta = self._distance(p[-2], p[1]) - (self._distance(p[-2], p[0]) + self._distance(p[0], p[1]))
        else:
            # Original connection: (idx-1) -> (idx) -> (idx+1)
            # New connection: (idx-1) -> (idx+1)
            old_dist = self._distance(p[idx-1], p[idx]) + self._distance(p[idx], p[idx+1])
            new_dist = self._distance(p[idx-1], p[idx+1])
            delta = new_dist - old_dist
        
        return delta

    def evaluate_addition_cost(self, solution: CspSolution, move_info: dict) -> float:
        """
        Calculates the cost delta for the Addition move (Adding Method).
        Inserts a node that is NOT in the route.
        """
        node_to_add = move_info['node']
        # The edge (u,v) where the node will be inserted is defined by 'position'
        # Inserting at 'position' i means it goes between (i-1) and (i)
        position = move_info['position']
        
        p = solution.path
        u = p[position - 1]
        v = p[position]

        # Original connection: u -> v
        # New connection: u -> node_to_add -> v
        old_dist = self._distance(u, v)
        new_dist = self._distance(u, node_to_add) + self._distance(node_to_add, v)
        delta = new_dist - old_dist
        
        return delta

    def evaluate_exchange_cost(self, solution: CspSolution, move_info: dict) -> float:
        """
        Calculates the delta for various exchange moves. Coverage feasibility
        usually doesn't change drastically, but it's ideal to check.
        """
        move_type = move_info.get('type', '2-opt') # 2-opt is the default if not specified

        # --- 2-Opt Move ---
        if move_type == '2-opt':
            i, j = move_info['indices']
            # For 2-Opt, coverage doesn't change since the set of visited nodes is the same.
            # The only change is in the distance.
            return self.evaluate_2opt_delta(solution, i, j)

        # --- Internal Swap (Swapping Method) ---
        elif move_type == 'swap':
            i, j = move_info['indices'] # Indices of the nodes to be swapped in the route
            p = solution.path
            
            # To simplify, we calculate the new route's total cost and subtract the old one.
            # A more efficient delta calculation is possible, but more complex.
            new_path = p.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]
            
            new_cost = 0
            for k in range(len(new_path) - 1):
                new_cost += self._distance(new_path[k], new_path[k+1])
            
            return new_cost - solution.travelled_distance

        # --- External Exchange (Exchanging Nodes Method) ---
        elif move_type == 'exchange_external':
            node_in = move_info['node_in']   # Node from outside the route
            node_out = move_info['node_out'] # Node from inside the route
            
            # 1. Check feasibility: does removing node_out and adding node_in maintain coverage?
            temp_solution = solution.copy()
            temp_solution.path.remove(node_out)
            temp_solution.path.append(node_in) # The exact position doesn't matter for the coverage check
            if not self._check_full_coverage(temp_solution):
                return float('inf') # Invalid move

            # 2. If feasible, calculate the distance delta
            idx_out = solution.path.index(node_out)
            p = solution.path
            
            prev_node = p[idx_out - 1]
            next_node = p[idx_out + 1]

            old_dist = self._distance(prev_node, node_out) + self._distance(node_out, next_node)
            new_dist = self._distance(prev_node, node_in) + self._distance(node_in, next_node)
            
            return new_dist - old_dist
        
        return float('inf') # Returns infinity if the move type is unknown

    def evaluate_2opt_delta(self, solution: CspSolution, i: int, j: int) -> float:
        p = solution.path
        old_dist = self.instance.get_distance(p[i-1], p[i]) + self.instance.get_distance(p[j], p[j+1])
        new_dist = self.instance.get_distance(p[i-1], p[j]) + self.instance.get_distance(p[i], p[j+1])
        return new_dist - old_dist

    def _check_full_coverage(self, solution: CspSolution) -> bool:
        """Checks if all nodes in the instance are covered by the route."""
        if not solution.path:
            return False

        all_nodes_to_cover = set(self.instance.get_all_nodes())
        nodes_covered_by_tour = set()
        
        # Iterates through the tour nodes and gathers all nodes they cover
        for node_in_path in set(solution.path):
            nodes_covered_by_tour.update(self.instance.coverage[node_in_path])
        
        # The solution is feasible if the set of covered nodes equals the total set of nodes
        return nodes_covered_by_tour == all_nodes_to_cover
    
    def _calculate_travelled_distance(self, solution: CspSolution):
        solution.travelled_distance = 0.0
        for i in range(len(solution.path) - 1):
            u, v = solution.path[i], solution.path[i + 1]
            solution.travelled_distance += self._distance(u, v)

    def _perpendicular_distance(self, point: tuple, line_start: tuple, line_end: tuple) -> float:
        point_to_check = np.array(point)
        start_edge = np.array(line_start)
        end_edge = np.array(line_end)

        line_vector = end_edge - start_edge
        point_to_start_vector = point_to_check - start_edge

        length_sq = np.dot(line_vector, line_vector)
        if length_sq == 0.0:
            return self._distance(point_to_check, line_start)

        aux = np.dot(point_to_start_vector, line_vector) / length_sq

        if aux < 0.0:
            closest_point = start_edge
        elif aux > 1.0:
            closest_point = end_edge
        else:
            closest_point = start_edge + aux * line_vector
        
        return np.linalg.norm(point_to_check - closest_point)

    def _distance(self, u: int, v: int) -> int:
        """Looks up the pre-computed distance between two nodes from the instance matrix."""
        if u is None or v is None:
            return 0
        return self.instance.get_distance(u, v)
    
    def evaluate_removal_delta(self, solution: CspSolution, node: tuple) -> float:
        # Item a) Removal Method
        if node not in solution.path:
            raise ValueError("Node to be removed is not in the solution path.")
        
        idx = solution.path.index(node)
        
        if idx == 0 or idx == len(solution.path) - 1:
            # If it's the first or last node, just remove it
            if idx == 0:
                # First node
                new_distance = solution.travelled_distance - self._distance(node, solution.path[1])
            else:
                # Last node
                new_distance = solution.travelled_distance - self._distance(solution.path[-2], node)
        else:
            # Remove the node and add the distance between its neighbors
            new_distance = solution.travelled_distance
            new_distance -= self._distance(solution.path[idx - 1], node)
            new_distance -= self._distance(node, solution.path[idx + 1])
            new_distance += self._distance(solution.path[idx - 1], solution.path[idx + 1])

        return solution.travelled_distance - new_distance

    def evaluate_edge_insertion_delta(self, solution: CspSolution, node: tuple, position: int) -> float:
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
        
        # Get the previous and next nodes (before removal)
        prev_lneighbor = solution.path[original_index - 1] if original_index > 0 else None
        prev_rneighbot = solution.path[original_index + 1] if original_index < len(solution.path) - 1 else None
        # Get the previous and next nodes (after insertion)
        new_lneighbor = solution.path[position] if position < len(solution.path) else None
        new_rneighbor = solution.path[position + 1] if position + 1 < len(solution.path) else None
        
        new_distance = solution.travelled_distance
        # Remove the node from its original position    
        new_distance -= self._distance(prev_lneighbor, node)
        new_distance -= self._distance(node, prev_rneighbot)
        new_distance += self._distance(prev_lneighbor, prev_rneighbot)
        # Insert the node at the new position
        new_distance -= self._distance(new_lneighbor, new_rneighbor)
        new_distance += self._distance(prev_lneighbor, node)
        new_distance += self._distance(node, new_rneighbor)

        return new_distance - solution.travelled_distance
    
    def evaluate_swap_delta(self, solution: CspSolution, node1: tuple, node2: tuple) -> float:
        """
        Item c) Swapping Method.
        Takes two nodes that are already in the path and evaluates the cost of swapping their positions.
        """
        
        if node1 not in solution.path or node2 not in solution.path:
            raise ValueError("Both nodes must be in the solution path.")
        
        if node1 == node2:
            return 0.0
        
        idx1, idx2 = solution.path.index(node1), solution.path.index(node2)
        
        if abs(idx1 - idx2) == 1:
            # Adjacent nodes
            lneighbor = solution.path[min(idx1, idx2) - 1] if min(idx1, idx2) > 0 else None
            rneighbor = solution.path[max(idx1, idx2) + 1] if max(idx1, idx2) < len(solution.path) - 1 else None
            
            new_distance = solution.travelled_distance
            new_distance -= self._distance(lneighbor, node1)
            new_distance -= self._distance(node1, node2)
            new_distance -= self._distance(node2, rneighbor)
            new_distance += self._distance(lneighbor, node2)
            new_distance += self._distance(node2, node1)
            new_distance += self._distance(node1, rneighbor)
            
            return new_distance - solution.travelled_distance
        else:
            # Non-adjacent nodes
            lneighbor1 = solution.path[idx1 - 1] if idx1 > 0 else None
            rneighbor1 = solution.path[idx1 + 1] if idx1 < len(solution.path) - 1 else None
            lneighbor2 = solution.path[idx2 - 1] if idx2 > 0 else None
            rneighbor2 = solution.path[idx2 + 1] if idx2 < len(solution.path) - 1 else None
            
            new_distance = solution.travelled_distance
            # Remove edges connected to node1 and node2
            new_distance -= self._distance(lneighbor1, node1)
            new_distance -= self._distance(node1, rneighbor1)
            new_distance -= self._distance(lneighbor2, node2)
            new_distance -= self._distance(node2, rneighbor2)
            # Add edges for swapped positions
            new_distance += self._distance(lneighbor1, node2)
            new_distance += self._distance(node2, rneighbor1)
            new_distance += self._distance(lneighbor2, node1)
            new_distance += self._distance(node1, rneighbor2)
            
            return new_distance - solution.travelled_distance