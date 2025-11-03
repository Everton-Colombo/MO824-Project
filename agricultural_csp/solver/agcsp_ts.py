import math
import numpy as np
from ..instance import *
from ..evaluator import *
from ..solution import *
from .abc_solver import Solver, TerminationCriteria, DebugOptions
import time
from collections import deque
from typing import Literal, Optional, List
from dataclasses import dataclass
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

    phased_optimization: Optional[PhasedOptimizationParams] = PhasedOptimizationParams()

    def __post_init__(self):
        if self.probabilistic_ts and not (0 < self.probabilistic_param < 1):
            raise ValueError("Probabilistic parameter must be in the range (0, 1) when probabilistic TS is enabled.")


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

    def solve(self) -> AgcspSolution:
        """Main method to solve the problem using Tabu Search."""
        self._reset_execution_state()
        
        # Initialize with constructive heuristic
        self.best_solution = self._constructive_heuristic(self.strategy.constructive_heuristic)
        self._current_solution = self.best_solution
        
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
            # Check if we should move to the next phase
            if self._iters - phase_start_iter >= self.strategy.phased_optimization.phase_iterations[current_phase]:
                # Move to next phase
                current_phase += 1
                phase_start_iter = self._iters
                
                # Update best component values for the next phase's constraints
                best_components = list(self.evaluator.objfun_components(self.best_solution))
                
                if current_phase >= len(self.strategy.phased_optimization.phase_iterations):
                    current_phase = 0  # Restart phases if all completed
                    
                if self.debug_options.verbose:
                    print(f"\n=== Switching to Phase {current_phase + 1} ===")
                    print(f"Current best components: Cov={best_components[0]:.2f}, Dist={best_components[1]:.2f}, Man={best_components[2]:.2f}")
            
            self._perform_debug_actions()
            
            # Perform phased neighborhood move
            self._current_solution = self._neighborhood_move_first_improving_phased(
                self._current_solution, current_phase, best_components
            )
            
            # Update execution state
            self._update_execution_state()
        
    
    def _perform_debug_actions(self):
        """Perform debug actions, such as logging or printing debug information."""
        if self.debug_options.verbose:
            best_val = f'{self.evaluator.objfun(self.best_solution):.2f}' if self.best_solution else 'N/A'
            current_val = f'{self.evaluator.objfun(self._current_solution):.2f}' if self._current_solution else 'N/A'
            print(f"Iteration {self._iters}: Best ObjFun = {best_val}, Current ObjFun = {current_val}")

        if self.debug_options.log_history:
            self.history.append((
                self._iters, 
                self.evaluator.objfun(self.best_solution) if self.best_solution else 0,
                self.evaluator.objfun(self._current_solution) if self._current_solution else 0
            ))

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
        

    def _apply_move(self, solution: AgcspSolution, move: str, move_args: tuple) -> AgcspSolution:
        """Applies a move to the given solution and returns the new solution."""
        
        if move == 'insert':
            node, index = move_args
            node_tuple = tuple(node)
            node_array = np.array(node_tuple, dtype=solution.path.dtype if solution.path.size else float)
            if solution.path.size == 0:
                new_path = node_array.reshape(1, -1)
            else:
                new_path = np.insert(solution.path, index, node_array, axis=0)
            self.tabu_list.append(node_tuple)
            return AgcspSolution(new_path)
        elif move == 'remove':
            index, = move_args
            removed_node = tuple(solution.path[index])
            new_path = np.delete(solution.path, index, axis=0)
            self.tabu_list.append(removed_node)
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
        raise NotImplementedError("Best improving move not implemented yet.")

    def _neighborhood_move_first_improving(self, solution: AgcspSolution) -> AgcspSolution:
        # Randomly decide which operator to try first
        operators = ['insert', 'remove']
        random.shuffle(operators)
        
        tabu_nodes = {tuple(np.array(node)) for node in self.tabu_list if node is not None}
        current_nodes = {tuple(np.array(point)) for point in solution.path}
        
        for operator in operators:
            if operator == 'insert':
                # Evaluating insertions
                candidate_nodes = [tuple(np.array(node)) for node in self.instance.grid_nodes]
                candidate_nodes = [node for node in candidate_nodes if node not in current_nodes]
                random.shuffle(candidate_nodes)
                
                for node in candidate_nodes:
                    if node in tabu_nodes:
                        continue
                    
                    path_indexes = list(range(len(solution.path) + 1))
                    random.shuffle(path_indexes)
                    for index in path_indexes:
                        delta = self.evaluator.evaluate_insertion_delta(solution, node, index)
                        if delta < 0:
                            return self._apply_move(solution, 'insert', (node, index))
            
            elif operator == 'remove':
                # Evaluating removals
                for index in range(len(solution.path)):
                    node_tuple = tuple(solution.path[index])
                    if node_tuple in tabu_nodes:
                        continue
                    
                    delta = self.evaluator.evaluate_removal_delta(solution, index)
                    if delta < 0:
                        return self._apply_move(solution, 'remove', (index,))
        
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
        # Randomly decide which operator to try first
        operators = ['insert', 'remove']
        random.shuffle(operators)
        
        tabu_nodes = {tuple(np.array(node)) for node in self.tabu_list if node is not None}
        current_nodes = {tuple(np.array(point)) for point in solution.path}
        
        # Get degradation tolerance for this phase
        tolerance = self.strategy.phased_optimization.degradation_tolerances[phase]
        
        for operator in operators:
            if operator == 'insert':
                # Evaluating insertions
                candidate_nodes = [tuple(np.array(node)) for node in self.instance.grid_nodes]
                candidate_nodes = [node for node in candidate_nodes if node not in current_nodes]
                random.shuffle(candidate_nodes)
                
                for node in candidate_nodes:
                    if node in tabu_nodes:
                        continue
                    
                    path_indexes = list(range(len(solution.path) + 1))
                    random.shuffle(path_indexes)
                    for index in path_indexes:
                        # Get component deltas
                        result = self.evaluator.evaluate_insertion_delta(
                            solution, node, index, return_components=True
                        )
                        
                        if result[0] == float('inf'):  # Invalid move
                            continue
                        
                        cov_delta, dist_delta, man_delta = result
                        
                        # Check if move is acceptable for current phase
                        if self._is_move_acceptable_for_phase(
                            phase, cov_delta, dist_delta, man_delta, 
                            best_components, tolerance
                        ):
                            return self._apply_move(solution, 'insert', (node, index))
            
            elif operator == 'remove':
                # Evaluating removals
                remove_indices = list(range(len(solution.path)))
                random.shuffle(remove_indices)
                
                for index in remove_indices:
                    node_tuple = tuple(solution.path[index])
                    if node_tuple in tabu_nodes:
                        continue
                    
                    # Get component deltas
                    result = self.evaluator.evaluate_removal_delta(
                        solution, index, return_components=True
                    )
                    
                    if result[0] == float('inf'):  # Invalid move
                        continue
                    
                    cov_delta, dist_delta, man_delta = result
                    
                    # Check if move is acceptable for current phase
                    if self._is_move_acceptable_for_phase(
                        phase, cov_delta, dist_delta, man_delta,
                        best_components, tolerance
                    ):
                        return self._apply_move(solution, 'remove', (index,))
        
        return solution
    
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

        
