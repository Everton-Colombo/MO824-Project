# src/core/abstract_ts.py

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Any, Optional

from .solution import Solution
from .evaluator import Evaluator

class AbstractTabuSearch(ABC):
    """
    Abstract base class for the Tabu Search algorithm.
    It defines the general structure that can be specialized for different problems.
    """
    
    VERBOSE = True
    
    def __init__(self, obj_function: Evaluator, tenure: int, iterations: int, random_seed: int = 0):
        """
        Initializes the Tabu Search solver.
        
        Args:
            obj_function (Evaluator): The problem-specific objective function.
            tenure (int): The size of the tabu list.
            iterations (int): The number of iterations to run.
            random_seed (int): Seed for the random number generator.
        """
        self.obj_function = obj_function
        self.tenure = tenure
        self.iterations = iterations
        
        self.best_sol: Optional[Solution] = None
        self.current_sol: Optional[Solution] = None
        
        self.tabu_list: Optional[deque] = None
        random.seed(random_seed)
    
    @abstractmethod
    def constructive_heuristic(self) -> Solution:
        """Creates an initial feasible solution."""
        pass

    @abstractmethod
    def neighborhood_move(self):
        """Generates a neighborhood and performs a move."""
        pass
    
    @abstractmethod
    def make_tabu_list(self) -> deque:
        """Creates and initializes the tabu list."""
        pass

    def solve(self) -> Solution:
        """
        Main method of the Tabu Search algorithm.
        """
        self.current_sol = self.constructive_heuristic()
        self.tabu_list = self.make_tabu_list()
        self.best_sol = self.current_sol.copy()
        
        if self.VERBOSE:
            print("\nInitial solution found and set as best:")
            print(self.best_sol)
            print("-" * 50)
        
        for iteration in range(self.iterations):
            self.neighborhood_move()
            
            if self.current_sol.cost < self.best_sol.cost:
                self.best_sol = self.current_sol.copy()
                if self.VERBOSE:
                    print(f"(Iter. {iteration + 1}) New best solution: Cost={self.best_sol.cost:.2f}")
        
        return self.best_sol
    
    def is_tabu(self, element: Any) -> bool:
        """Checks if a given move attribute is in the tabu list."""
        return self.tabu_list is not None and element in self.tabu_list
    
    def aspiration_criteria(self, delta_cost: float) -> bool:
        """
        Checks if a tabu move can be accepted.
        Accepts if the move results in a better solution than the best-so-far.
        """
        if self.current_sol is None or self.best_sol is None:
            return False
        return (self.current_sol.cost + delta_cost) < self.best_sol.cost
    
    def add_to_tabu_list(self, element: Any):
        """Adds a move attribute to the tabu list."""
        if self.tabu_list is not None:
            self.tabu_list.append(element)
    
    def set_verbose(self, verbose: bool):
        """Sets the verbose mode for printing execution details."""
        self.VERBOSE = verbose