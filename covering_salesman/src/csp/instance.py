import networkx as nx

class CspInstance:
    def __init__(self, filepath: str):
        self.name: str = ""
        self.num_nodes: int = 0
        self.neighbor_count: int = 0
        self.distances: list[list[int]] = []
        self.coverage: dict[int, set[int]] = {}
        self._load_from_file(filepath)
        
        print(f"Instância '{self.name}' carregada: {self.num_nodes} nós.")

    def _load_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        self.name = lines[0]
        self.num_nodes = int(lines[1])
        self.neighbor_count = int(lines[2])

        current_line = 4 
        for i in range(self.num_nodes):
            row_values = list(map(int, lines[current_line].split()))
            self.distances.append(row_values)
            current_line += 1
        
        current_line += 1

        for i in range(self.num_nodes):
            parts = list(map(int, lines[current_line].split()))
            covering_node = parts[0]
            covered_nodes = set(parts[1:])
            covered_nodes.add(covering_node)
            self.coverage[covering_node] = covered_nodes
            current_line += 1

    def get_all_nodes(self) -> list[int]:
        """Retorna uma lista com os IDs de todos os nós (0 a n-1)."""
        return list(range(self.num_nodes))

    def get_distance(self, node1: int, node2: int) -> int:
        """Retorna a distância pré-calculada entre dois nós."""
        return self.distances[node1][node2]

    def _build_graph(self, node_positions: list[tuple[float, float]]) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.num_nodes))
        return g    