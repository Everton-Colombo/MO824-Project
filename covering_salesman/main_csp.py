from pathlib import Path
from src.csp.instance import CspInstance
from src.csp.evaluator import CspEvaluator
from src.csp.solver import TabuSearchCSP

def run_solver():
    """
    Loads an instance, configures the Tabu Search solver, runs it,
    and prints the best solution found.
    """
    # --- Configuration ---
    instance_filename = "eil51-7.csp2" 
    tenure = 10
    iterations = 1000
    
    # --- Path Setup ---
    root_dir = Path(__file__).resolve().parent
    instance_path = root_dir / "instances" / instance_filename

    # --- Solver Execution ---
    try:
        # 1. Load Instance
        instance = CspInstance(instance_path)

        # 2. Create Evaluator
        evaluator = CspEvaluator(instance)

        # 3. Configure and Create Tabu Search Solver
        solver = TabuSearchCSP(evaluator, tenure, iterations)

        # 4. Solve the problem
        print("\nStarting Tabu Search...")
        best_solution = solver.solve()

        # 5. Print the result
        print("\n--- Best Solution Found ---")
        print(best_solution)

    except FileNotFoundError:
        print(f"ERROR: Instance file not found at '{instance_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_solver()