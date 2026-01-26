# research_main.py
import sys
import json
import os
import pandas as pd
from datetime import datetime
import numpy as np

# Add project root to path to allow imports from other folders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from geometry_engine import generate_geometry_data
    from solvers.highs_solver import HighsMIPSolver
except ImportError as e:
    print(f"Error: Failed to import necessary modules. {e}", file=sys.stderr)
    print("Please ensure geometry_engine.py and solvers/highs_solver.py exist.", file=sys.stderr)
    sys.exit(1)

def run_experiment(n: int, k: int, q: int, weights: list, results_csv_path: str):
    """
    Controls a single experiment: generates geometry, solves, and saves results.
    """
    print("="*60)
    print(f"Starting Experiment: n={n}, k={k}, q={q}, weights={weights}")
    print("="*60)

    result_log = {
        'timestamp': datetime.now().isoformat(), 'n': n, 'k': k, 'q': q, 
        'weights': str(weights), 'solution_found': False, 'model_status': 'ERROR',
        'solve_time_s': 0, 'mip_nodes': 0, 'mip_gap': None,
        'non_zero_points': 0, 'multiplicity_sum': 0
    }

    try:
        # 1. Generate geometry data
        geometry = generate_geometry_data(n, k, q, weights)
        
        # 2. Initialize and run the solver
        solver = HighsMIPSolver(data=geometry)
        solution_info = solver.solve()

        # 3. Prepare results for logging
        result_log.update({
            'solution_found': solution_info['solution_found'],
            'model_status': solution_info['model_status'],
            'solve_time_s': round(solution_info['solve_time'], 4),
            'mip_nodes': solution_info['node_count'],
            'mip_gap': solution_info['gap'],
            'non_zero_points': np.count_nonzero(solution_info['x_counts']) if solution_info['x_counts'] is not None else 0,
            'multiplicity_sum': int(np.sum(solution_info['x_counts'])) if solution_info['x_counts'] is not None else 0,
        })
        
        # 4. Print a clear summary of the solver output
        print("\n" + "="*25 + " FINAL RESULT " + "="*25)
        print(f"  Model Status:     {result_log['model_status']}")
        print(f"  Solution Found:   {result_log['solution_found']}")
        print(f"  Solve Time (sec): {result_log['solve_time_s']}")
        print(f"  MIP Nodes:        {result_log['mip_nodes']}")
        print(f"  MIP Gap:          {result_log['mip_gap']}")
        if solution_info.get('x_counts') is not None:
            print(f"  Multiplicity Sum: {result_log['multiplicity_sum']} (Expected: {n})")
        print("="*64)

    except Exception as e:
        print(f"\n!!! An error occurred during the experiment: {e}", file=sys.stderr)
        result_log['model_status'] = f'ERROR: {str(e).replace(",", ";")}'

    # 5. Append results to CSV
    print(f"\nAppending results to {results_csv_path}...")
    try:
        df_new = pd.DataFrame([result_log])
        if os.path.exists(results_csv_path):
            df_new.to_csv(results_csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(results_csv_path, mode='w', header=True, index=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"!!! Failed to write to CSV file: {e}", file=sys.stderr)

def main():
    if len(sys.argv) != 5:
        print('Usage: python research_main.py <n> <k> <q> "<weights>"')
        print('Example: python research_main.py 21 3 4 "[16, 17]"')
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
        q = int(sys.argv[3])
        # Use json.loads for robust parsing of the list string
        weights = json.loads(sys.argv[4])
        if not isinstance(weights, list):
            raise ValueError("Weights argument must be a valid JSON list.")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error: Invalid arguments. {e}", file=sys.stderr)
        sys.exit(1)

    results_csv = "research_results.csv"
    run_experiment(n, k, q, weights, results_csv)

if __name__ == "__main__":
    main()
