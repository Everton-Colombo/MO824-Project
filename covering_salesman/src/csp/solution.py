from src.core.solution import Solution

class CspSolution(Solution):
    def __init__(self, path: list[tuple[float, float]] = None):
        super().__init__()
        self.path: list[tuple[float, float]] = path if path is not None else []

    def __len__(self) -> int:
        return len(self.path)
        
    def __repr__(self) -> str:
        path_str = " -> ".join(map(str, self.path))
        return (f"Custo: {self.cost:.2f}, Viável: {self.is_feasible}, "
                f"Nós na rota: {len(self.path)}\nRota: {path_str}")
        
    def copy(self):
        new_solution = CspSolution(self.path.copy())
        new_solution.cost = self.cost
        new_solution.is_feasible = self.is_feasible
        return new_solution