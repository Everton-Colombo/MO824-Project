from abc import ABC, abstractmethod
import numpy as np
import heapq
import math
import random
from scipy.spatial.distance import cdist
from typing import List, Literal, Dict, Any, Tuple
from enum import Enum
from ...instance import AgcspInstance
from ...evaluator import AgcspEvaluator
from ...solution import AgcspSolution 

class ConstructiveHeuristicType(Enum):
    """Types of constructive heuristics available."""
    BOUSTROPHEDON_SEGMENTED = 'boustrophedon_segmented'
    FSM_COVERAGE_PLANNER = 'fsm_coverage_planner'
    GREEDY_COVERAGE = 'greedy_coverage'

class BaseConstructiveHeuristic(ABC):
    """Abstract base class for all constructive heuristics."""

    def __init__(self, instance: AgcspInstance, evaluator: AgcspEvaluator):
        self.instance = instance
        self.evaluator = evaluator
    
    @abstractmethod
    def generate_initial_solution(self) -> AgcspSolution:
        """
        Generates and returns a valid initial solution.
        """
        pass

class GreedyCoverageHeuristic(BaseConstructiveHeuristic):
    """
    Builds an initial solution using a Greedy approach focused on maximizing coverage step by step.
    """

    def generate_initial_solution(self) -> AgcspSolution:
        """
        Performs greedy search, adding the node that improves coverage the most at each step, respecting collision and angle limits.
        """
        instance = self.instance
        evaluator = self.evaluator
        
        if len(instance.grid_nodes) == 0:
            return AgcspSolution(path=[])
            
        solution = AgcspSolution(path=[tuple(self.instance.grid_nodes[0])])
        uncovered_nodes = set(tuple(node) for node in self.instance.grid_nodes)
        
        max_allowed_penalty = 1.0 - float(np.cos(np.deg2rad(self.instance.max_turn_angle)))
        
        while not self.evaluator.is_feasible(solution):
            coverage_mask = self.evaluator._coverage_mask(solution)
            covered_indices = np.argwhere(coverage_mask)
            covered_nodes_coords = covered_indices + self.instance.min_coords
            covered_nodes_set = set(tuple(node) for node in covered_nodes_coords)
            
            uncovered_nodes = uncovered_nodes - covered_nodes_set
            
            current_coverage = self.evaluator.coverage_proportion(solution)
            print(f"Current coverage: {current_coverage:.4f}, path length: {len(solution.path)}, uncovered: {len(uncovered_nodes)}")
            
            path_set = set(tuple(node) if isinstance(node, (list, np.ndarray)) else node for node in solution.path)
            candidate_nodes = list(uncovered_nodes - path_set)
            
            if not candidate_nodes:
                print("No more candidate nodes to add, but solution is still infeasible.")
                break
            
            random.shuffle(candidate_nodes)
            
            node_added = False
            for node in candidate_nodes:
                new_path = list(solution.path) + [node]
                new_solution = AgcspSolution(new_path)
                
                if self.evaluator.hits_obstacle(new_solution):
                    continue
                
                if len(solution.path) >= 2:
                    p_prev = np.array(solution.path[-2])
                    p_curr = np.array(solution.path[-1])
                    p_next = np.array(node)
                    
                    angle_penalty = self.evaluator._calculate_angle_penalty_at_node(p_prev, p_curr, p_next)
                    
                    if angle_penalty > max_allowed_penalty:
                        continue
                
                new_coverage = self.evaluator.coverage_proportion(new_solution)
                if new_coverage > current_coverage:
                    solution = new_solution
                    node_added = True
                    break
            
            if not node_added:
                print(f"No valid uncovered nodes found. Trying covered nodes near uncovered areas...")
                
                half_sprayer = self.instance.sprayer_length / 2
                uncovered_arr = np.array(list(uncovered_nodes))
                covered_arr = np.array(list(covered_nodes_set - path_set))
                
                if len(covered_arr) > 0 and len(uncovered_arr) > 0:
                    distances = cdist(covered_arr, uncovered_arr)
                    
                    near_uncovered_mask = np.any(distances <= half_sprayer, axis=1)
                    nearby_covered_nodes = covered_arr[near_uncovered_mask]
                    
                    if len(nearby_covered_nodes) > 0:
                        np.random.shuffle(nearby_covered_nodes)
                        
                        for node in nearby_covered_nodes:
                            node_tuple = tuple(node)
                            new_path = list(solution.path) + [node_tuple]
                            new_solution = AgcspSolution(new_path)
                            
                            if self.evaluator.hits_obstacle(new_solution):
                                continue
                            
                            if len(solution.path) >= 2:
                                p_prev = np.array(solution.path[-2])
                                p_curr = np.array(solution.path[-1])
                                p_next = np.array(node_tuple)
                                
                                angle_penalty = self.evaluator._calculate_angle_penalty_at_node(p_prev, p_curr, p_next)
                                
                                if angle_penalty > max_allowed_penalty:
                                    continue
                            
                            new_coverage = self.evaluator.coverage_proportion(new_solution)
                            if new_coverage > current_coverage:
                                solution = new_solution
                                node_added = True
                                print(f"Added covered node near uncovered area: {node_tuple}")
                                break
            
            if not node_added:
                print(f"Could not find any valid node to add from {len(candidate_nodes)} candidates.")
                break

        return solution

