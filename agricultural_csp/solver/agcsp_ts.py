import math
import numpy as np
from ..instance import *
from ..evaluator import *
from ..solution import *
from .abc_solver import Solver, TerminationCriteria, DebugOptions
import time
from collections import deque
from typing import Literal, Optional, List
from dataclasses import dataclass, field
import random

from .constructive_heuristics.base_heuristics import *


@dataclass
class PhasedOptimizationParams:
    phase_iterations: List[int] = (100, 100, 100)   # Number of iterations per phase (coverage, distance and path complexity)
    degradation_tolerances: List[float] = (0.02, 0.05, 0.1) # Allowed degradation per phase (as fraction of best objective value)

@dataclass
class TSStrategy():
    """
    Configuration data class for the Tabu Search algorithm.
    """
    
    constructive_heuristic: ConstructiveHeuristicType = ConstructiveHeuristicType.RANDOM
    
    search_strategy: Literal['first', 'best'] = 'first'
    probabilistic_ts: bool = False
    probabilistic_param: float = 0.8
    
    tabu_radius: float = 0.0  # Radius around operated nodes to mark as tabu (0 = only the exact node)
    move_min_distance: int = 5  # Minimum distance for move operator search

    phased_optimization: Optional[PhasedOptimizationParams] = field(
        default_factory=PhasedOptimizationParams
    )

    def __post_init__(self):
        if self.probabilistic_ts and not (0 < self.probabilistic_param < 1):
            raise ValueError("Probabilistic parameter must be in the range (0, 1) when probabilistic TS is enabled.")
        if self.tabu_radius < 0:
            raise ValueError("Tabu radius must be non-negative.")
        if self.move_min_distance < 0:
            raise ValueError("Move minimum distance must be non-negative.")


class AgcspTS(Solver):
    
    def __init__(self, instance: AgcspInstance, tenure: int = 7, strategy: TSStrategy = TSStrategy(), 
                 termination_criteria: TerminationCriteria = TerminationCriteria(),
                 debug_options: DebugOptions = DebugOptions()):
        
        # Initialize parent class
        super().__init__(instance, termination_criteria, debug_options)
        
        # TS-specific properties
        self.strategy = strategy
        self.tabu_list = deque([None] * tenure, maxlen=tenure)
        
        # For debug history tracking
        if debug_options.log_history:
            self.history: List[tuple] = []

        self.sampling_limits = {
            'insert_nodes': 100,
            'insert_pos': 50,
            'remove_nodes': 100,
            'swap_pairs': 2000,
            'move_nodes': 100,
            '2opt_pairs': 2000
        }

    def solve(self, initial_solution: AgcspSolution = None) -> AgcspSolution:
        """Main method to solve the problem using Tabu Search."""
        self._reset_execution_state()
        
        if initial_solution:
            print("--- Iniciando TS com solução fornecida. ---")
            self.best_solution = initial_solution
        else:
            print("--- Gerando nova solução inicial... ---")
            self.best_solution = self._constructive_heuristic(self.strategy.constructive_heuristic)

        self._current_solution = self.best_solution

        print("-----------------------------------------------------------")
        print(f"Initial solution objective value: {self.evaluator.objfun(self.best_solution):.2f}")
        print("-----------------------------------------------------------")
        
        if self.strategy.phased_optimization is not None:
            self._solve_phased()
        else:
            self._solve_standard()
        
        self.execution_time = time.time() - self._start_time
        return self.best_solution
    
    def _solve_standard(self) -> AgcspSolution:
        while not self._check_termination():
            self._perform_debug_actions()

            # Perform neighborhood move
            self._current_solution = self._neighborhood_move(self._current_solution)
            
            # Update execution state (handles best solution tracking)
            self._update_execution_state()
    
    def _solve_phased(self) -> AgcspSolution:
        current_phase = 0
        phase_start_iter = 0
        
        # Store best component values for constraint enforcement
        best_components = list(self.evaluator.objfun_components(self.best_solution))
        
        while not self._check_termination():

            force_phase_change = False

            if self._iters - phase_start_iter >= self.strategy.phased_optimization.phase_iterations[current_phase]:
                force_phase_change = True
            
            self._perform_debug_actions()

            new_solution = None
        
            if self.strategy.search_strategy == 'first':
                new_solution = self._neighborhood_move_first_improving_phased(
                    self._current_solution, current_phase, best_components
                )
            elif self.strategy.search_strategy == 'best':
                new_solution = self._neighborhood_move_best_improving_phased(
                    self._current_solution, current_phase, best_components
                )
            else:
                raise ValueError(f"Unknown search strategy: {self.strategy.search_strategy}")

            if new_solution is None:
                if not force_phase_change and self.debug_options.verbose:
                    print(f"--- Fase {current_phase + 1} estagnada ({self._iters - phase_start_iter} iterações). Trocando de fase. ---")
                
                force_phase_change = True
                
            else:
                self._current_solution = new_solution
            
            if force_phase_change and not self._check_termination(): 
                current_phase += 1
                phase_start_iter = self._iters
                
                if current_phase >= len(self.strategy.phased_optimization.phase_iterations):
                    current_phase = 0 
                    
                best_components = list(self.evaluator.objfun_components(self.best_solution))
                
                if self.debug_options.verbose:
                    print(f"\n=== Trocando para Fase {current_phase + 1} (Objetivo: Otimizar C{current_phase + 1}) ===")

            self._update_execution_state()
        
    
    def _perform_debug_actions(self):
        """Perform debug actions, such as logging or printing debug information."""
        if self.debug_options.verbose:
            best_val = f'{self.evaluator.objfun(self.best_solution):.2f}' if self.best_solution else 'N/A'
            current_val = f'{self.evaluator.objfun(self._current_solution):.2f}' if self._current_solution else 'N/A'
            print(f"Iteration {self._iters}: Best ObjFun = {best_val}, Current ObjFun = {current_val}")

        if self.debug_options.log_history:
            if self.best_solution:
                best_obj = self.evaluator.objfun(self.best_solution)
                best_cov, best_dist, best_man = self.evaluator.objfun_components(self.best_solution)
            else:
                best_obj, best_cov, best_dist, best_man = 0.0, 0.0, 0.0, 0.0

            if self._current_solution:
                curr_obj = self.evaluator.objfun(self._current_solution)
                curr_cov, curr_dist, curr_man = self.evaluator.objfun_components(self._current_solution)
            else:
                curr_obj, curr_cov, curr_dist, curr_man = best_obj, best_cov, best_dist, best_man

            self.history.append({
                'Iteration': self._iters,
                'Best_Obj': best_obj,
                'Best_Cov_Pen': best_cov,
                'Best_Dist': best_dist,
                'Best_Man_Pen': best_man,
                'Current_Obj': curr_obj,
                'Current_Cov_Pen': curr_cov,
                'Current_Dist': curr_dist,
                'Current_Man_Pen': curr_man
            })

    def _get_heuristic_builder(self, instance, evaluator, strategy: ConstructiveHeuristicType) -> BaseConstructiveHeuristic:
        """Returns the correct instance of the heuristic based on the strategy."""
        
        mapping = {
            ConstructiveHeuristicType.BOUSTROPHEDON_SEGMENTED: BoustrophedonSegmentedHeuristic,
            ConstructiveHeuristicType.FSM_COVERAGE_PLANNER: FSMCoveragePlannerHeuristic,
            ConstructiveHeuristicType.RANDOM: RandomCoverageHeuristic,
        }

        HeuristicClass = mapping.get(strategy)

        if HeuristicClass is None:
            raise ValueError(f"Unknown Constructive Heuristic: {strategy}")

        return HeuristicClass(instance, evaluator)
    
    def _constructive_heuristic(self, strategy_construct_heuristic) -> AgcspSolution:
        """
        Constructs an initial feasible solution.
        """
        self._constructive_builder = self._get_heuristic_builder(self.instance, self.evaluator, strategy_construct_heuristic)
        return self._constructive_builder.generate_initial_solution()
    
    def _add_to_tabu_list(self, node: tuple | np.ndarray):
        """
        Adds a node (and nearby nodes if tabu_radius > 0) to the tabu list.
        
        Args:
            node: The node to add to tabu list (as tuple or numpy array)
        """
        # Convert to tuple for consistent comparison
        if isinstance(node, np.ndarray):
            node = tuple(node)
        
        if self.strategy.tabu_radius <= 0:
            # Only add the exact node
            self.tabu_list.append(node)
        else:
            # Add node and all nodes within tabu_radius
            node_array = np.array(node)
            nodes_to_add = [node]  # Start with the operated node
            
            # Find all nodes within tabu_radius
            for grid_node in self.instance.grid_nodes:
                grid_node_tuple = tuple(grid_node)
                if grid_node_tuple == node:
                    continue  # Already added
                
                distance = np.linalg.norm(grid_node - node_array)
                if distance <= self.strategy.tabu_radius:
                    nodes_to_add.append(grid_node_tuple)
            
            # Add all nodes at once (they will be expired together based on tenure)
            # We add them as a set to keep track of nodes that should be tabu together
            self.tabu_list.append(frozenset(nodes_to_add))
    
    def _is_node_tabu(self, node: tuple | np.ndarray) -> bool:
        """
        Checks if a node is in the tabu list.
        Handles both individual nodes and sets of nodes (when tabu_radius > 0).
        
        Args:
            node: The node to check (as tuple or numpy array)
            
        Returns:
            True if the node is tabu, False otherwise
        """
        # Convert to tuple for consistent comparison
        if isinstance(node, np.ndarray):
            node = tuple(node)
        
        for tabu_entry in self.tabu_list:
            if tabu_entry is None:
                continue
            
            # Handle both single nodes (tuples) and sets of nodes (frozensets)
            if isinstance(tabu_entry, frozenset):
                if node in tabu_entry:
                    return True
            elif tabu_entry == node:
                return True
        
        return False
        

    def _apply_move(self, solution: AgcspSolution, move: str, move_args: tuple) -> AgcspSolution:
        """Applies a move to the given solution and returns the new solution."""
        
        if self.debug_options.verbose:
            print(f"Applying move: {move} with args {move_args}")
        
        if move == 'insert':
            node, index = move_args
            node_tuple = tuple(node)
            node_array = np.array(node_tuple, dtype=solution.path.dtype if solution.path.size else float)
            if solution.path.size == 0:
                new_path = node_array.reshape(1, -1)
            else:
                new_path = np.insert(solution.path, index, node_array, axis=0)
            self._add_to_tabu_list(node_tuple)
            return AgcspSolution(new_path)
        elif move == 'remove':
            index, = move_args
            removed_node = tuple(solution.path[index])
            new_path = np.delete(solution.path, index, axis=0)
            self._add_to_tabu_list(removed_node)
            return AgcspSolution(new_path)
        elif move == 'move':
            index, new_node = move_args
            old_node = tuple(solution.path[index])
            new_path = np.array(solution.path, dtype=solution.path.dtype if solution.path.size else float)
            new_path[index] = new_node
            self._add_to_tabu_list(old_node)
            return AgcspSolution(new_path)
        elif move == 'swap':
            idx1, idx2 = move_args
            new_path = solution.path.copy() 
            self._add_to_tabu_list(tuple(new_path[idx1]))
            self._add_to_tabu_list(tuple(new_path[idx2]))
            new_path[idx1], new_path[idx2] = new_path[idx2].copy(), new_path[idx1].copy()
            return AgcspSolution(new_path)
        elif move == '2-opt':
            i, j = move_args
            self._add_to_tabu_list(tuple(solution.path[i]))
            self._add_to_tabu_list(tuple(solution.path[j]))
            
            new_path = np.concatenate([
                solution.path[:i+1],
                solution.path[j:i:-1],
                solution.path[j+1:]
            ])
            return AgcspSolution(new_path)
        else:
            raise ValueError(f"Unknown move type: {move}")

    def _neighborhood_move(self, solution: AgcspSolution) -> AgcspSolution:
        if self.strategy.search_strategy == 'first':
            return self._neighborhood_move_first_improving(solution)
        elif self.strategy.search_strategy == 'best':
            return self._neighborhood_move_best_improving(solution)
        else:
            raise ValueError(f"Unknown search strategy: {self.strategy.search_strategy}")

    def _neighborhood_move_best_improving(self, solution: AgcspSolution) -> AgcspSolution:
        best_delta = 0.0
        best_move_info = None
        path_len = len(solution.path)
        
        current_nodes = {tuple(np.array(point)) for point in solution.path}
        
        operators = ['remove', 'move', 'swap', '2-opt', 'insert']
        
        for operator in operators:
            
            if operator == 'insert':
                candidate_nodes = [tuple(np.array(node)) for node in self.instance.grid_nodes]
                candidate_nodes = [node for node in candidate_nodes if node not in current_nodes]
                random.shuffle(candidate_nodes)
                
                for node in candidate_nodes[:self.sampling_limits['insert_nodes']]:
                    if self._is_node_tabu(node):
                        continue
                    
                    path_indexes = list(range(path_len + 1))
                    random.shuffle(path_indexes)
                    
                    for index in path_indexes[:self.sampling_limits['insert_pos']]:
                        delta = self.evaluator.evaluate_insertion_delta(solution, node, index)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move_info = ('insert', (node, index))
                            
            elif operator == 'remove':
                remove_indices = list(range(path_len))
                random.shuffle(remove_indices)
                
                for index in remove_indices[:self.sampling_limits['remove_nodes']]:
                    node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(node_tuple):
                        continue
                    
                    delta = self.evaluator.evaluate_removal_delta(solution, index)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move_info = ('remove', (index,))

            elif operator == 'swap':
                if path_len < 2: continue
                
                all_pairs = [(i, j) for i in range(path_len) for j in range(i + 1, path_len)]
                random.shuffle(all_pairs)
                
                for idx1, idx2 in all_pairs[:self.sampling_limits['swap_pairs']]:
                    node1 = tuple(solution.path[idx1])
                    node2 = tuple(solution.path[idx2])

                    if self._is_node_tabu(node1) or self._is_node_tabu(node2):
                        continue
                        
                    delta = self.evaluator.evaluate_swap_delta(solution, idx1, idx2, return_components=False)
                    
                    if delta == float('inf'): continue
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move_info = ('swap', (idx1, idx2))
                            
            elif operator == 'move':
                move_indices = list(range(path_len))
                random.shuffle(move_indices)
                directions = ['up', 'down', 'left', 'right']
                min_distance = self.strategy.move_min_distance
                
                for index in move_indices[:self.sampling_limits['move_nodes']]:
                    old_node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(old_node_tuple):
                        continue
                    
                    for direction in directions:
                        new_node = self.evaluator._find_node_in_direction(
                            solution.path[index], min_distance, direction
                        )
                        
                        if new_node is None or np.array_equal(new_node, solution.path[index]):
                            continue

                        result = self.evaluator.evaluate_move_delta(
                            solution, index, min_distance, direction, return_components=True
                        )
                        
                        if result is None or result[0] == float('inf'): continue
                        
                        delta = sum(result)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move_info = ('move', (index, new_node))
            
            elif operator == '2-opt':
                if path_len < 4: continue
                
                all_pairs = [(i, j) for i in range(path_len - 2) for j in range(i + 2, path_len - 1)]
                random.shuffle(all_pairs)
                
                for i, j in all_pairs[:self.sampling_limits['2opt_pairs']]:
                    if self._is_node_tabu(tuple(solution.path[i])) or self._is_node_tabu(tuple(solution.path[j])):
                        continue
                        
                    delta = self.evaluator.evaluate_2opt_delta(solution, i, j, return_components=False)

                    if delta == float('inf'): continue
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move_info = ('2-opt', (i, j))
                            
        if best_move_info is not None:
            operator, move_args = best_move_info
            
            if self.debug_options.verbose:
                print(f"** BEST IMPROVING MOVE ENCONTRADO ** Tipo: {operator}, Delta: {best_delta:.4f}")
                
            return self._apply_move(solution, operator, move_args)
        else:
            return solution

    def _neighborhood_move_first_improving(self, solution: AgcspSolution) -> AgcspSolution:
        operators = ['remove', 'move', 'swap', '2-opt', 'insert']
        
        current_nodes = {tuple(np.array(point)) for point in solution.path}
        path_len = len(solution.path)
        
        for operator in operators:
            if operator == 'insert':
                candidate_nodes = [tuple(np.array(node)) for node in self.instance.grid_nodes]
                candidate_nodes = [node for node in candidate_nodes if node not in current_nodes]
                random.shuffle(candidate_nodes)
                
                for node in candidate_nodes[:self.sampling_limits['insert_nodes']]:
                    if self._is_node_tabu(node):
                        continue
                    
                    path_indexes = list(range(path_len + 1))
                    random.shuffle(path_indexes)
                    
                    for index in path_indexes[:self.sampling_limits['insert_pos']]:
                        delta = self.evaluator.evaluate_insertion_delta(solution, node, index)
                        if delta < 0:
                            return self._apply_move(solution, 'insert', (node, index))
            
            elif operator == 'remove':
                remove_indices = list(range(path_len))
                random.shuffle(remove_indices)

                for index in remove_indices[:self.sampling_limits['remove_nodes']]:
                    node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(node_tuple): continue
                    
                    result = self.evaluator.evaluate_removal_delta(solution, index, return_components=True)
                    
                    if result[0] == float('inf'): continue
                    
                    cov_delta, dist_delta, man_delta = result
                    total_delta = sum(result)
                    
                    if cov_delta <= 1e-6 and total_delta < 0:
                        return self._apply_move(solution, 'remove', (index,))

            elif operator == 'swap':
                if path_len < 2: continue
                
                all_pairs = [(i, j) for i in range(path_len) for j in range(i + 1, path_len)]
                random.shuffle(all_pairs)
                
                for idx1, idx2 in all_pairs[:self.sampling_limits['swap_pairs']]:
                    node1 = tuple(solution.path[idx1])
                    node2 = tuple(solution.path[idx2])

                    if self._is_node_tabu(node1) or self._is_node_tabu(node2):
                        continue
                        
                    delta = self.evaluator.evaluate_swap_delta(solution, idx1, idx2, return_components=False)
                    
                    if delta < 0 and delta != float('inf'):
                        return self._apply_move(solution, 'swap', (idx1, idx2))

            elif operator == 'move':
                move_indices = list(range(path_len))
                random.shuffle(move_indices)
                directions = ['up', 'down', 'left', 'right']
                min_distance = self.strategy.move_min_distance
                
                for index in move_indices[:self.sampling_limits['move_nodes']]:
                    old_node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(old_node_tuple):
                        continue
                    
                    random.shuffle(directions)
                    for direction in directions:
                        
                        new_node = self.evaluator._find_node_in_direction(
                            solution.path[index], min_distance, direction
                        )
                        
                        if new_node is None or np.array_equal(new_node, solution.path[index]):
                            continue
                        
                        result = self.evaluator.evaluate_move_delta(
                            solution, index, min_distance, direction, return_components=True
                        )
                        
                        if result is None or result[0] == float('inf'):
                            continue
                            
                        total_delta = sum(result)
                        
                        if total_delta < 0:
                            if self.debug_options.verbose:
                                print(f"  >>> MOVE ACEITO (Move Padrão): Index {index} em direção a {direction}. Delta: {total_delta:.4f}")
                            return self._apply_move(solution, 'move', (index, new_node))
            
            elif operator == '2-opt':
                if path_len < 4: continue
                
                all_pairs = [(i, j) for i in range(path_len - 2) for j in range(i + 2, path_len - 1)]
                random.shuffle(all_pairs)
                
                for i, j in all_pairs[:self.sampling_limits['2opt_pairs']]:
                    if self._is_node_tabu(tuple(solution.path[i])) or self._is_node_tabu(tuple(solution.path[j])):
                        continue
                        
                    delta = self.evaluator.evaluate_2opt_delta(solution, i, j, return_components=False)
                    
                    if delta < 0 and delta != float('inf'):
                        return self._apply_move(solution, '2-opt', (i, j))
        
        return solution
    
    def _neighborhood_move_first_improving_phased(self, solution: AgcspSolution, phase: int, 
                                                   best_components: List[float]) -> AgcspSolution:
        """
        Performs a first-improving neighborhood search with phase-specific objective focus.
        
        Args:
            solution: Current solution
            phase: Current optimization phase (0=coverage, 1=distance, 2=maneuvers)
            best_components: Best component values [coverage_penalty, distance_penalty, maneuver_penalty] 
                           from previous phases to use as constraints
        
        Returns:
            Improved solution or the same solution if no improvement found
        """
        operators = ['remove', 'move', 'swap', '2-opt', 'insert']
        
        current_nodes = {tuple(np.array(point)) for point in solution.path}
        tolerance = self.strategy.phased_optimization.degradation_tolerances[phase]
        path_len = len(solution.path)
        
        for operator in operators:
            if operator == 'insert':
                candidate_nodes = [tuple(np.array(node)) for node in self.instance.grid_nodes]
                candidate_nodes = [node for node in candidate_nodes if node not in current_nodes]
                random.shuffle(candidate_nodes)
                
                for node in candidate_nodes[:self.sampling_limits['insert_nodes']]:
                    if self._is_node_tabu(node):
                        continue
                    
                    path_indexes = list(range(path_len + 1))
                    random.shuffle(path_indexes)
                    
                    for index in path_indexes[:self.sampling_limits['insert_pos']]:
                        result = self.evaluator.evaluate_insertion_delta(
                            solution, node, index, return_components=True
                        )
                        
                        if result[0] == float('inf'):
                            continue
                        
                        cov_delta, dist_delta, man_delta = result
                        
                        if self._is_move_acceptable_for_phase(
                            phase, cov_delta, dist_delta, man_delta, 
                            best_components, tolerance
                        ):
                            return self._apply_move(solution, 'insert', (node, index))
            
            elif operator == 'remove':
                remove_indices = list(range(path_len))
                random.shuffle(remove_indices)
                
                for index in remove_indices[:self.sampling_limits['remove_nodes']]:
                    node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(node_tuple):
                        continue
                    
                    result = self.evaluator.evaluate_removal_delta(
                        solution, index, return_components=True
                    )
                    
                    if result[0] == float('inf'):
                        continue
                    
                    cov_delta, dist_delta, man_delta = result
                    
                    if self._is_move_acceptable_for_phase(
                        phase, cov_delta, dist_delta, man_delta,
                        best_components, tolerance
                    ):
                        return self._apply_move(solution, 'remove', (index,))
            
            elif operator == 'move':
                move_indices = list(range(path_len))
                random.shuffle(move_indices)
                directions = ['up', 'down', 'left', 'right']
                
                for index in move_indices[:self.sampling_limits['move_nodes']]:
                    node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(node_tuple):
                        continue
                    
                    random.shuffle(directions)
                    for direction in directions:
                        result = self.evaluator.evaluate_move_delta(
                            solution, index, self.strategy.move_min_distance, direction, return_components=True
                        )
                        
                        if result is None or result[0] == float('inf'):
                            continue
                        
                        cov_delta, dist_delta, man_delta = result
                        
                        if self._is_move_acceptable_for_phase(
                            phase, cov_delta, dist_delta, man_delta,
                            best_components, tolerance
                        ):
                            new_node = self.evaluator._find_node_in_direction(
                                np.array(solution.path[index]), self.strategy.move_min_distance, direction
                            )
                            return self._apply_move(solution, 'move', (index, new_node))

            elif operator == 'swap':
                if path_len < 2:
                    continue
                
                all_pairs = [(i, j) for i in range(path_len) for j in range(i + 1, path_len)]
                random.shuffle(all_pairs)

                for idx1, idx2 in all_pairs[:self.sampling_limits['swap_pairs']]:
                    node1 = tuple(solution.path[idx1])
                    node2 = tuple(solution.path[idx2])

                    if self._is_node_tabu(node1) or self._is_node_tabu(node2):
                        continue

                    result = self.evaluator.evaluate_swap_delta(
                        solution, idx1, idx2, return_components=True
                    )

                    if result[0] == float('inf'):
                        continue

                    cov_delta, dist_delta, man_delta = result

                    if self._is_move_acceptable_for_phase(
                        phase, cov_delta, dist_delta, man_delta,
                        best_components, tolerance
                    ):
                        if self.debug_options.verbose:
                            print(f"  >>> MOVE ACEITO (Swap): Indices ({idx1}, {idx2}). Delta Total: {cov_delta + dist_delta + man_delta:.4f}")
                        return self._apply_move(solution, 'swap', (idx1, idx2))
            
            elif operator == '2-opt':
                if path_len < 4: continue
                
                all_pairs = [(i, j) for i in range(path_len - 2) for j in range(i + 2, path_len - 1)]
                random.shuffle(all_pairs)
                
                for i, j in all_pairs[:self.sampling_limits['2opt_pairs']]:
                    if self._is_node_tabu(tuple(solution.path[i])) or self._is_node_tabu(tuple(solution.path[j])):
                        continue
                        
                    result = self.evaluator.evaluate_2opt_delta(
                        solution, i, j, return_components=True
                    )
                    
                    if result[0] == float('inf'): continue
                    
                    cov_delta, dist_delta, man_delta = result
                    
                    if self._is_move_acceptable_for_phase(
                        phase, cov_delta, dist_delta, man_delta,
                        best_components, tolerance
                    ):
                        if self.debug_options.verbose:
                            print(f"  >>> MOVE ACEITO (2-Opt): Indices ({i}, {j}). Delta Dist: {dist_delta:.4f}")
                        return self._apply_move(solution, '2-opt', (i, j))
        
        return None

    def _neighborhood_move_best_improving_phased(self, solution: AgcspSolution, phase: int, 
                                            best_components: List[float]) -> AgcspSolution | None:
    
        best_total_delta = float('inf')
        best_focus_delta = 0.0
        best_move_info = None
        
        current_nodes = {tuple(np.array(point)) for point in solution.path}
        tolerance = self.strategy.phased_optimization.degradation_tolerances[phase]
        cov_delta = 0
        dist_delta = 0
        man_delta = 0
        path_len = len(solution.path)
        
        focus_component_idx = phase

        operators = ['remove', 'move', 'swap', '2-opt', 'insert']
        
        for operator in operators:

            if operator == 'insert':
                candidate_nodes = [tuple(np.array(node)) for node in self.instance.grid_nodes]
                candidate_nodes = [node for node in candidate_nodes if node not in current_nodes]
                random.shuffle(candidate_nodes)
                
                for node in candidate_nodes[:self.sampling_limits['insert_nodes']]:
                    if self._is_node_tabu(node): continue
                    
                    path_indexes = list(range(path_len + 1))
                    random.shuffle(path_indexes)
                    
                    for index in path_indexes[:self.sampling_limits['insert_pos']]:
                        result = self.evaluator.evaluate_insertion_delta(solution, node, index, return_components=True)
                        if result[0] == float('inf'): continue
                        
                        cov_delta, dist_delta, man_delta = result
                        deltas = [cov_delta, dist_delta, man_delta]
                        
                        if self._is_move_acceptable_for_phase(phase, cov_delta, dist_delta, man_delta, best_components, tolerance):
                            focus_delta = deltas[focus_component_idx]
                            
                            if focus_delta < best_focus_delta:
                                best_focus_delta = focus_delta
                                best_total_delta = sum(deltas)
                                best_move_info = ('insert', (node, index))
                            elif focus_delta == best_focus_delta and sum(deltas) < best_total_delta:
                                best_total_delta = sum(deltas)
                                best_move_info = ('insert', (node, index))
                                
            elif operator == 'remove':
                remove_indices = list(range(path_len))
                random.shuffle(remove_indices)
                
                for index in remove_indices[:self.sampling_limits['remove_nodes']]:
                    node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(node_tuple): continue
                    
                    result = self.evaluator.evaluate_removal_delta(solution, index, return_components=True)
                    if result[0] == float('inf'): continue
                    
                    cov_delta, dist_delta, man_delta = result
                    deltas = [cov_delta, dist_delta, man_delta]
                    
                    if self._is_move_acceptable_for_phase(phase, cov_delta, dist_delta, man_delta, best_components, tolerance):
                        focus_delta = deltas[focus_component_idx]

                        if focus_delta < best_focus_delta:
                            best_focus_delta = focus_delta
                            best_total_delta = sum(deltas)
                            best_move_info = ('remove', (index,))
                        elif focus_delta == best_focus_delta and sum(deltas) < best_total_delta:
                            best_total_delta = sum(deltas)
                            best_move_info = ('remove', (index,))
                            
            elif operator == 'move':
                move_indices = list(range(path_len))
                random.shuffle(move_indices)
                directions = ['up', 'down', 'left', 'right']
                min_distance = self.strategy.move_min_distance
                
                for index in move_indices[:self.sampling_limits['move_nodes']]:
                    old_node_tuple = tuple(solution.path[index])
                    if self._is_node_tabu(old_node_tuple): continue
                    
                    for direction in directions:
                        result = self.evaluator.evaluate_move_delta(
                            solution, index, min_distance, direction, return_components=True
                        )
                        
                        if result is None or result[0] == float('inf'): continue
                        
                        new_node = self.evaluator._find_node_in_direction(solution.path[index], min_distance, direction)
                        if new_node is None or np.array_equal(new_node, solution.path[index]): continue
                        
                        cov_delta, dist_delta, man_delta = result
                        deltas = [cov_delta, dist_delta, man_delta]

                        if self._is_move_acceptable_for_phase(phase, cov_delta, dist_delta, man_delta, best_components, tolerance):
                            focus_delta = deltas[focus_component_idx]

                            if focus_delta < best_focus_delta:
                                best_focus_delta = focus_delta
                                best_total_delta = sum(deltas)
                                best_move_info = ('move', (index, new_node))
                            elif focus_delta == best_focus_delta and sum(deltas) < best_total_delta:
                                best_total_delta = sum(deltas)
                                best_move_info = ('move', (index, new_node))

            elif operator == 'swap':
                if path_len < 2: continue
                
                all_pairs = [(i, j) for i in range(path_len) for j in range(i + 1, path_len)]
                random.shuffle(all_pairs)
                
                for idx1, idx2 in all_pairs[:self.sampling_limits['swap_pairs']]:
                    node1 = tuple(solution.path[idx1])
                    node2 = tuple(solution.path[idx2])

                    if self._is_node_tabu(node1) or self._is_node_tabu(node2): continue

                    result = self.evaluator.evaluate_swap_delta(solution, idx1, idx2, return_components=True)
                    if result[0] == float('inf'): continue

                    cov_delta, dist_delta, man_delta = result
                    deltas = [cov_delta, dist_delta, man_delta]

                    if self._is_move_acceptable_for_phase(phase, cov_delta, dist_delta, man_delta, best_components, tolerance):
                        focus_delta = deltas[focus_component_idx]

                        if focus_delta < best_focus_delta:
                            best_focus_delta = focus_delta
                            best_total_delta = sum(deltas)
                            best_move_info = ('swap', (idx1, idx2))
                        elif focus_delta == best_focus_delta and sum(deltas) < best_total_delta:
                            best_total_delta = sum(deltas)
                            best_move_info = ('swap', (idx1, idx2))
            
            elif operator == '2-opt':
                if path_len < 4: continue
                
                all_pairs = [(i, j) for i in range(path_len - 2) for j in range(i + 2, path_len - 1)]
                random.shuffle(all_pairs)

                for i, j in all_pairs[:self.sampling_limits['2opt_pairs']]:
                    if self._is_node_tabu(tuple(solution.path[i])) or self._is_node_tabu(tuple(solution.path[j])):
                        continue

                    result = self.evaluator.evaluate_2opt_delta(
                        solution, i, j, return_components=True
                    )
                    
                    if result[0] == float('inf'): continue

                    cov_delta, dist_delta, man_delta = result
                    deltas = [cov_delta, dist_delta, man_delta]

                    if self._is_move_acceptable_for_phase(phase, cov_delta, dist_delta, man_delta, best_components, tolerance):
                        focus_delta = deltas[focus_component_idx]
                        
                        if focus_delta < best_focus_delta:
                            best_focus_delta = focus_delta
                            best_total_delta = sum(deltas)
                            best_move_info = ('2-opt', (i, j))
                        elif focus_delta == best_focus_delta and sum(deltas) < best_total_delta:
                            best_total_delta = sum(deltas)
                            best_move_info = ('2-opt', (i, j))

            if self._is_move_acceptable_for_phase(phase, cov_delta, dist_delta, man_delta, best_components, tolerance):
                deltas = [cov_delta, dist_delta, man_delta]
                focus_delta = deltas[focus_component_idx]
                current_total_delta = sum(deltas)

                if focus_delta < best_focus_delta:
                    best_focus_delta = focus_delta
                    best_total_delta = current_total_delta
                    best_move_info = (operator, move_args)
                    
                elif focus_delta == best_focus_delta and current_total_delta < best_total_delta:
                    best_total_delta = current_total_delta
                    best_move_info = (operator, move_args)

        if best_move_info is not None:
            operator, move_args = best_move_info
            
            if self.debug_options.verbose:
                print(f"** BEST IMPROVING ACEITO ** Tipo: {operator}, Delta Foco ({focus_component_idx}): {best_focus_delta:.4f}")
                
            return self._apply_move(solution, operator, move_args)
        else:
            return None
    
    def _is_move_acceptable_for_phase(self, phase: int, cov_delta: float, dist_delta: float, 
                                      man_delta: float, best_components: List[float], 
                                      tolerance: float) -> bool:
        """
        Determines if a move is acceptable based on the current optimization phase.
        
        Phase 0 (Coverage): Optimize coverage, allow distance and maneuvers to worsen within tolerance
        Phase 1 (Distance): Optimize distance, constrain coverage, allow maneuvers to worsen
        Phase 2 (Maneuvers): Optimize maneuvers, constrain both coverage and distance
        
        Args:
            phase: Current optimization phase (0, 1, or 2)
            cov_delta, dist_delta, man_delta: Component deltas for the move
            best_components: Best component values from previous iterations
            tolerance: Allowed degradation as a fraction of the best value
        
        Returns:
            True if the move is acceptable for the current phase
        """
        deltas = [cov_delta, dist_delta, man_delta]
        
        if phase == 0:
            # Phase 0: Focus on improving coverage
            # Accept if coverage improves
            return cov_delta < 0
            
        elif phase == 1:
            # Phase 1: Focus on improving distance
            # Constrain: coverage must not degrade beyond tolerance
            # Accept if: distance improves AND coverage constraint is satisfied
            coverage_constraint = cov_delta <= best_components[0] * tolerance
            return dist_delta < 0 and coverage_constraint
            
        elif phase == 2:
            # Phase 2: Focus on improving maneuvers
            # Constrain: both coverage and distance must not degrade beyond tolerance
            # Accept if: maneuvers improve AND both constraints satisfied
            coverage_constraint = cov_delta <= best_components[0] * tolerance
            distance_constraint = dist_delta <= best_components[1] * tolerance
            return man_delta < 0 and coverage_constraint and distance_constraint
            
        else:
            # Fallback: accept any improving move
            return sum(deltas) < 0

        
