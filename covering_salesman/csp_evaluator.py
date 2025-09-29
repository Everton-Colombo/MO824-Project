from csp_instance import *
from csp_solution import *
import networkx as nx
import numpy as np

class CspEvaluator:
    
    def __init__(self, instance: CspInstance):
        self.instance = instance
    
    def evaluate_objfun(self, solution: CspSolution) -> float:
        if solution.travelled_distance is None:
            self._calculate_travelled_distance(solution)
        
        return solution.travelled_distance
    
    def _distance(self, u: tuple, v: tuple) -> float:
        if u is None or v is None:
            return 0.0
        return np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
    
    def _calculate_travelled_distance(self, solution: CspSolution):
        solution.travelled_distance = 0.0
        for i in range(len(solution.path) - 1):
            u, v = solution.path[i], solution.path[i + 1]
            solution.travelled_distance += self._distance(u, v)
    
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