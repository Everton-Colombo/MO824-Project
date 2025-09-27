import networkx as nx

class CspInstance:
    
    def __init__(self, graph: nx.Graph, cover_radius: float):
        self.graph = graph
        self.cover_radius = cover_radius