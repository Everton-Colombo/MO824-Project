import math
import numpy as np
from ..instance import *
from ..evaluator import *
from ..solution import *
from .abc_solver import Solver, TerminationCriteria, DebugOptions
import random
import time
from collections import deque
from typing import Literal
from dataclasses import dataclass

class RestartIntensificationComponent():
    def __init__(self, instance: AgcspInstance = None, restart_patience: int = 100, max_fixed_elements: int = 3):
        pass
        # self._instance = instance
        
        # self.recency_memory: List[int] = None
        # self.restart_patience = restart_patience
        # self.max_fixed_elements = max_fixed_elements

    # def set_instance(self, instance: AgcspInstance):
    #     self._instance = instance
    #     self.recency_memory = [0] * instance.n

    def update_recency_memory(self, best_solution: AgcspSolution):
        pass
        # elements_in_solution = set(best_solution.elements)
        # elements_not_in_solution = set(range(self._instance.n)) - elements_in_solution
        
        # for element in elements_in_solution:
        #     self.recency_memory[element] += 1
                
        # for element in elements_not_in_solution:
        #     self.recency_memory[element] = 0
        
    def get_attractive_elements(self) -> List[int]:
        pass
        # # return a list of the most recurring elements (up to max_fixed_elements) that dont have a zero value in recency_memory
        
        # sorted_elements = sorted(range(self._instance.n), key=lambda x: self.recency_memory[x], reverse=True)
        # return [element for element in sorted_elements if self.recency_memory[element] > 0][:self.max_fixed_elements]


@dataclass
class TSStrategy():
    """
    Configuration data class for the Tabu Search algorithm.
    """
    search_strategy: Literal['first', 'best'] = 'first'
    probabilistic_ts: bool = False
    probabilistic_param: float = 0.8
    ibr_component: RestartIntensificationComponent = None
    
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
        
        
        # # Initialize IBR component if present
        # if self.strategy.ibr_component is not None:
        #     self._fixed_elements: List[int] = []
        #     self.strategy.ibr_component.set_instance(instance)
        
        # For debug history tracking
        if debug_options.log_history:
            self.history: List[tuple] = []

    def solve(self) -> AgcspSolution:
        """Main method to solve the problem using Tabu Search."""
        self._reset_execution_state()
        
        # Initialize with constructive heuristic
        self.best_solution = self._constructive_heuristic()
        self._current_solution = self.best_solution
        
        while not self._check_termination():
            self._perform_debug_actions()
            
            # # Check for restart with intensification
            # if (self.strategy.ibr_component is not None and 
            #     (self._iters_wo_improvement + 1) % self.strategy.ibr_component.restart_patience == 0):
            #     self._fixed_elements = self.strategy.ibr_component.get_attractive_elements()
            #     self._current_solution = ScQbfSolution(self.best_solution.elements.copy())
                
            #     if self.debug_options.verbose:
            #         print(f"Restarting with intensification at iteration {self._iters}. Fixed elements: {self._fixed_elements}.")
            
            # Perform neighborhood move
            self._current_solution = self._neighborhood_move(self._current_solution)
            
            # Update execution state (handles best solution tracking)
            self._update_execution_state()
            
            # # Update IBR memory if enabled
            # if self.strategy.ibr_component is not None:
            #     self.strategy.ibr_component.update_recency_memory(self.best_solution)
        
        self.execution_time = time.time() - self._start_time
        return self.best_solution
    
    def _perform_debug_actions(self):
        """Perform debug actions, such as logging or printing debug information."""
        if self.debug_options.verbose:
            best_val = f'{self.evaluator.evaluate_objfun(self.best_solution):.2f}' if self.best_solution else 'N/A'
            current_val = f'{self.evaluator.evaluate_objfun(self._current_solution):.2f}' if self._current_solution else 'N/A'
            print(f"Iteration {self._iters}: Best ObjFun = {best_val}, Current ObjFun = {current_val}")

        if self.debug_options.log_history:
            self.history.append((
                self._iters, 
                self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 0,
                self.evaluator.evaluate_objfun(self._current_solution) if self._current_solution else 0
            ))
    
    def _constructive_heuristic(self) -> AgcspSolution:
        """
        Constructs an initial feasible solution..
        """
        pass
    
    def _apply_move(self, solution: AgcspSolution, move: str, move_args: tuple) -> AgcspSolution:
        if move == 'insert':
            node, index = move_args
            new_path = solution.path[:index] + [node] + solution.path[index:]
            return AgcspSolution(new_path)
        elif move == 'remove':
            index, = move_args
            new_path = solution.path[:index] + solution.path[index+1:]
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
        # Evaluating insertions
        cl = [node for node in self.instance.grid_nodes if node not in solution.path]
        for node in cl:
            if node in self.tabu_list:
                continue
            
            path_indexes = random.sample(range(len(solution.path) + 1), len(solution.path) + 1)
            for index in path_indexes:
                delta = self.evaluator.evaluate_insertion_delta(solution, node, index)
                if delta < 0:
                    return self._apply_move(solution, ('insert', node, index))
        
        # Evaluating removals
        for index in range(len(solution.path)):
            if solution.path[index] in self.tabu_list:
                continue

            delta = self.evaluator.evaluate_removal_delta(solution, index)
            if delta < 0:
                return self._apply_move(solution, ('remove', index))
        return solution