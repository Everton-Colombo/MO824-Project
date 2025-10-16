from typing import List, Tuple
import numpy as np

Node = Tuple[float, float]

class AgcspSolution:
    def __init__(self, path: List[Node]):
        self.path = np.array(path)

    def __repr__(self):
        return f"AgcspSolution(path={self.path}, travelled_distance={self.travelled_distance})"