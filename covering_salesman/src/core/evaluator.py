from abc import ABC, abstractmethod
from .solution import Solution

class Evaluator(ABC):
    """
    Interface (Abstract Base Class) for all problem evaluators.
    It defines the contract that a concrete evaluator class MUST implement
    to be compatible with the Tabu Search framework.
    """
    @abstractmethod
    def evaluate(self, solution: Solution) -> float:
        """Calculates and returns the total cost of a given solution."""
        pass

    @abstractmethod
    def evaluate_removal_cost(self, solution: Solution, move_info: dict) -> float:
        """Calculates the cost delta for a removal move."""
        pass

    @abstractmethod
    def evaluate_addition_cost(self, solution: Solution, move_info: dict) -> float:
        """Calculates the cost delta for an addition move."""
        pass

    @abstractmethod
    def evaluate_exchange_cost(self, solution: Solution, move_info: dict) -> float:
        """Calculates the cost delta for exchange-type moves (e.g., 2-Opt, Swap)."""
        pass