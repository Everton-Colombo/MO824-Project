from typing import List, Tuple
import numpy as np
import functools

Node = Tuple[float, float]

class AgcspSolution:
    def __init__(self, path: List[Node]):
        self.path = np.array(path)
        
        self._cache = {'__hash': hash(self.path.tobytes())}
        
    @property
    def cache(self):
        if self._cache['__hash'] == hash(self.path.tobytes()):
            return self._cache
        else:
            # Cache is stale, reset it
            self._cache = {'__hash': hash(self.path.tobytes())}
            return self._cache

    def __repr__(self):
        return f"AgcspSolution(path={self.path}, cache_keys={list(self.cache.keys())})"

def cache_on_solution(func):
    """Decorator to cache function results on AgcspSolution's internal cache."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if one of the arguments is an AgcspSolution
        solution: AgcspSolution = None
        for arg in args:
            if isinstance(arg, AgcspSolution):
                solution = arg
                break
        if solution is None:
            for _, arg in kwargs.items():
                if isinstance(arg, AgcspSolution):
                    solution = arg
                    break
        if solution is None:
            return func(*args, **kwargs)
        
        # Check if the result is already cached
        if func.__name__ in solution.cache:
            return solution.cache[func.__name__]
        else:
            result = func(*args, **kwargs)
            solution.cache[func.__name__] = result
            return result
        
    return wrapper