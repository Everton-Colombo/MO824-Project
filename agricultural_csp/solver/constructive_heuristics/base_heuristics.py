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

class BoustrophedonSegmentedHeuristic(BaseConstructiveHeuristic):
    
    def __init__(self, instance: AgcspInstance, evaluator: AgcspEvaluator):
        super().__init__(instance, evaluator)
        
    def _find_contiguous_segments(self, nodes: np.ndarray) -> list[np.ndarray]:
        """
        Takes a list of nodes in a column and divides it into contiguous segments.
        """
        if len(nodes) == 0:
            return []

        segments = []
        current_segment = [nodes[0]]

        for i in range(1, len(nodes)):
            if nodes[i][0] - nodes[i-1][0] > 1.5:
                segments.append(np.array(current_segment))
                current_segment = []
            current_segment.append(nodes[i])
        
        if current_segment:
            segments.append(np.array(current_segment))
            
        return segments

    def _generate_boustrophedon_path(self) -> AgcspSolution:
        """
        Generates an initial route in Boustrophedon pattern.
        """
        instance = self.instance
        
        visitable_indices = np.argwhere(instance._visitable_mask)
        if visitable_indices.size == 0:
            return AgcspSolution(path=[])
        
        visitable_nodes = visitable_indices + instance.min_coords

        path_points = []
        last_point = None
        sprayerLength = instance.sprayer_length
        step = sprayerLength - 1
        
        min_row, min_col = np.min(visitable_nodes, axis=0)
        max_row, max_col = np.max(visitable_nodes, axis=0)

        min_row += math.floor(sprayerLength / 2.0)
        max_row -= math.floor(sprayerLength / 2.0)
        
        row = min_row

        while row <= max_row:
            nodes_in_column = visitable_nodes[(visitable_nodes[:, 1] >= row) & (visitable_nodes[:, 1] <= row)]

            if len(nodes_in_column) == 0:
                row += step
                continue

            nodes_in_column = nodes_in_column[np.argsort(nodes_in_column[:, 0])]
            segments = self._find_contiguous_segments(nodes_in_column)

            if not segments:
                row += step
                continue
                
            if last_point is None:
                target_segment = segments[0]
                entry_point = tuple(target_segment[0])
                print(f"  Iniciando em {entry_point}")
            else:
                best_entry_point = None
                min_dist = float('inf')
                
                for segment in segments:
                    endpoints = [segment[0], segment[-1]]
                    for point in endpoints:
                        dist = np.linalg.norm(np.array(last_point) - point)
                        if dist < min_dist:
                            if self.evaluator.is_path_segment_valid(last_point, tuple(point)):
                                min_dist = dist
                                best_entry_point = tuple(point)
                                target_segment = segment
                
                entry_point = best_entry_point

            if entry_point is not None:
                path_points.append(entry_point)
                last_point = entry_point 

                entry_point_arr = np.array(entry_point)

                if np.array_equal(entry_point_arr, target_segment[0]):
                    segment_to_add = target_segment
                else:
                    segment_to_add = target_segment[::-1]

                internal_step = self.instance.sprayer_length 

                for i in range(internal_step, len(segment_to_add), internal_step):
                  current_node = tuple(segment_to_add[i])
                  
                  path_points.append(current_node)
                  last_point = current_node 
                    
                final_node_of_segment = tuple(segment_to_add[-1])
                
                if len(segment_to_add) > 1 and not np.array_equal(np.array(last_point), final_node_of_segment):
                  path_points.append(final_node_of_segment)
                  last_point = final_node_of_segment
                    
            else:
                last_point = None

            row += step
        
        if not path_points:
            return AgcspSolution(path=[])
            
        unique_path = [path_points[0]]
        for i in range(1, len(path_points)):
            p_current = np.array(path_points[i])
            p_previous = np.array(path_points[i-1])
            
            if not np.array_equal(p_current, p_previous):
                unique_path.append(path_points[i])
                
        solution = AgcspSolution(unique_path)
        return solution

    
    def generate_initial_solution(self) -> AgcspSolution:
      """
      Orchestrates the generation of the solution
      """
      
      solution = self._generate_boustrophedon_path()
  
      return solution

