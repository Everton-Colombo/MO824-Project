import networkx as nx

class CspSolution:
    
    def __init__(self, path: list[tuple]):
        self.path = path
        
        self.travelled_distance: float
        # self.curve_penalty: float
        # self.area_covered: float
        # self.total_cost: float
        