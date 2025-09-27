from csp_instance import *
from csp_solution import *
import networkx as nx
import numpy as np

class CspEvaluator:
    
    def __init__(self, instance: CspInstance):
        self.instance = instance
    
    def evaluate_objfun(self, solution: CspSolution) -> float:
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

    def evaluate_edge_insertion_delta(self, solution: CspSolution, inserted_node: tuple, position: int) -> float:
        # Item b) Insertion Method
        
        if position < 0 or position > len(solution.path):
            raise ValueError("Position is out of bounds.")
        
        original_index = solution.path.index(inserted_node)
        prev_u = solution.path[original_index - 1] if original_index > 0 else None
        prev_v = solution.path[original_index + 1] if original_index < len(solution.path) - 1 else None

        new_distance = solution.travelled_distance
        new_distance -= self._distance(prev_u, inserted_node)
        new_distance -= self._distance(inserted_node, prev_v)
        new_distance += self._distance(prev_u, prev_v)
            
        if position > 0 and position < len(solution.path):
            # Inserting in the middle
            
            u = solution.path[position - 1]
            v = solution.path[position]
            
            new_distance -= self._distance(u, v)
            new_distance += self._distance(u, inserted_node)
            new_distance += self._distance(inserted_node, v)
            
        elif position == 0:
            # Inserting at the start
            v = solution.path[0]
            new_distance = solution.travelled_distance
            
            new_distance += self._distance(inserted_node, v)
            
        else:  # position == len(solution.path)
            # Inserting at the end
            u = solution.path[-1]
            new_distance = solution.travelled_distance
            new_distance += self._distance(u, inserted_node)

        return new_distance - solution.travelled_distance