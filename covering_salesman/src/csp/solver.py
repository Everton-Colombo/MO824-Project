import random
from collections import deque
from src.core.abstract_ts import AbstractTabuSearch
from src.csp.evaluator import CspEvaluator
from src.csp.solution import CspSolution

class TabuSearchCSP(AbstractTabuSearch):
    
    def __init__(self, evaluator: CspEvaluator, tenure: int, iterations: int, random_seed: int = 0):
        super().__init__(evaluator, tenure, iterations, random_seed)
    
    def _get_fake_element(self) -> None:
        """A tabu attribute can be a node (int) or an edge pair (tuple).
           None is used as a generic placeholder.
        """
        return None

    def make_tabu_list(self) -> deque:
        """The tabu list will store the attributes of a move."""
        return deque(maxlen=self.tenure)

    def constructive_heuristic(self) -> CspSolution:
        """
        Builds a small, coverage-focused route using a GRASP-like heuristic.
        """
        instance = self.obj_function.instance
        all_nodes_set = set(instance.get_all_nodes())
        
        start_node = random.choice(instance.get_all_nodes())
        current_path = [start_node, start_node]
        
        covered_nodes = instance.coverage.get(start_node, {start_node}).copy()

        while covered_nodes != all_nodes_set:
            best_candidate = None
            best_benefit_cost_ratio = -1.0

            candidate_nodes_to_add = all_nodes_set - set(current_path)
            
            if not candidate_nodes_to_add:
                print("WARNING: All nodes are in path, but coverage is still incomplete.")
                break

            for candidate_node in candidate_nodes_to_add:
                newly_covered_count = len(instance.coverage.get(candidate_node, {candidate_node}) - covered_nodes)

                if newly_covered_count == 0:
                    continue

                min_insertion_delta = float('inf')
                for i in range(1, len(current_path)):
                    u, v = current_path[i-1], current_path[i]
                    delta = (instance.get_distance(u, candidate_node) +
                             instance.get_distance(candidate_node, v) -
                             instance.get_distance(u, v))
                    if delta < min_insertion_delta:
                        min_insertion_delta = delta
                
                if min_insertion_delta < 1e-9:
                    benefit_cost_ratio = float('inf')
                else:
                    benefit_cost_ratio = newly_covered_count / min_insertion_delta

                if benefit_cost_ratio > best_benefit_cost_ratio:
                    best_benefit_cost_ratio = benefit_cost_ratio
                    best_candidate = candidate_node
            
            if best_candidate is None:
                 print("WARNING: No candidate improves coverage. Constructive heuristic terminated.")
                 break

            best_pos_to_insert = -1
            min_delta = float('inf')
            for i in range(1, len(current_path)):
                u, v = current_path[i-1], current_path[i]
                delta = (instance.get_distance(u, best_candidate) +
                         instance.get_distance(best_candidate, v) -
                         instance.get_distance(u, v))
                if delta < min_delta:
                    min_delta = delta
                    best_pos_to_insert = i
            
            current_path.insert(best_pos_to_insert, best_candidate)
            covered_nodes.update(instance.coverage.get(best_candidate, {best_candidate}))
        
        final_solution = CspSolution(path=current_path)
        self.obj_function.evaluate(final_solution)
        
        return final_solution
    
    def neighborhood_move(self):
        """
        Generates neighbors using multiple operators (2-Opt, Removal)
        and applies the best non-tabu move found (Best-Improving strategy).
        """
        best_delta = float('inf')
        best_move_info = None

        # --- Neighborhood 1: 2-Opt (Route Optimization) ---
        p = self.current_sol.path
        path_len = len(p)
        for i in range(path_len - 2):
            for j in range(i + 2, path_len - 1):
                delta = self.obj_function.evaluate_2opt_delta(self.current_sol, i, j)
                if delta < best_delta:
                    edge1 = tuple(sorted((p[i], p[i+1])))
                    edge2 = tuple(sorted((p[j], p[j+1])))
                    tabu_attr = ("2-opt", tuple(sorted((edge1, edge2))))
                    
                    is_aspirated = (self.current_sol.cost + delta) < self.best_sol.cost
                    if tabu_attr not in self.tabu_list or is_aspirated:
                        best_delta = delta
                        best_move_info = {'type': '2-opt', 'indices': (i, j), 'attr': tabu_attr}

        # --- Neighborhood 2: Node Removal (Route Shrinking) ---
        if len(self.current_sol.path) > 3:
            for i in range(1, len(p) - 1):
                node_to_remove = p[i]
                move_info = {'node': node_to_remove}
                delta = self.obj_function.evaluate_removal_cost(self.current_sol, move_info)

                if delta < best_delta:
                    tabu_attr = ("remove", node_to_remove)
                    is_aspirated = (self.current_sol.cost + delta) < self.best_sol.cost
                    if tabu_attr not in self.tabu_list or is_aspirated:
                        best_delta = delta
                        best_move_info = {'type': 'remove', 'node': node_to_remove, 'attr': tabu_attr}

        # --- Apply the best found move ---
        if best_move_info:
            move_type = best_move_info['type']
            
            if move_type == '2-opt':
                i, j = best_move_info['indices']
                self.current_sol.path[i+1:j+1] = reversed(self.current_sol.path[i+1:j+1])
            elif move_type == 'remove':
                self.current_sol.path.remove(best_move_info['node'])
            
            self.obj_function.evaluate(self.current_sol)
            self.add_to_tabu_list(best_move_info['attr'])

    # --- Unused Abstract Methods ---
    def make_candidate_list(self): pass
    def make_restricted_candidate_list(self): pass
    def update_candidate_list(self): pass
    def create_empty_solution(self): 
        return CspSolution()