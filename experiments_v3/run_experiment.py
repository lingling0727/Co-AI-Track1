# run_experiment.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 현재 프로젝트의 모듈을 가져올 수 있도록 경로 설정함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from geometry_engine import generate_geometry_data
    from solvers.highs_solver import HighsMIPSolver
except ImportError as e:
    print(f"Error: Failed to import necessary modules. {e}", file=sys.stderr)
    print("Please ensure geometry_engine.py and solvers/highs_solver.py exist.", file=sys.stderr)
    sys.exit(1)

def run_experiment(n: int, k: int, q: int, weights: list[int], results_csv_path: str):
    """
    실험 한 번 돌리는 전체 과정임: 기하 생성, 솔버 실행, 결과 저장.
    """
    print("="*60)
    print(f"Starting Experiment: n={n}, k={k}, q={q}, weights={weights}")
    print("="*60)

    if not weights:
        print(f"Error: Weights list cannot be empty.", file=sys.stderr)
        return

    result_log = {
        'timestamp': datetime.now().isoformat(), 'n': n, 'k': k, 'q': q,
        'weights': str(weights),
        'solution_found': False, 'model_status': 'ERROR: UNKNOWN',
        'solve_time_s': 0, 'mip_nodes': 0, 'mip_gap': None
    }

    try:
        # 1. 입력받은 가중치로 기하 데이터 생성함
        geometry = generate_geometry_data(n, k, q, weights)
        
        # 2. 솔버 만들고 실행함
        solver = HighsMIPSolver(data=geometry)
        solution_info = solver.solve()

        # 3. CSV에 기록할 결과 준비함
        result_log.update({
            'solution_found': solution_info['solution_found'],
            'model_status': solution_info['model_status'],
            'solve_time_s': round(solution_info['solve_time'], 4),
            'mip_nodes': solution_info['node_count'],
            'mip_gap': solution_info['gap'],
        })
        
        print("\n" + "="*25 + " FINAL RESULT " + "="*25)
        print(f"  Model Status:     {result_log['model_status']}")
        print(f"  Solution Found:   {result_log['solution_found']}")
        print(f"  Solve Time (sec): {result_log['solve_time_s']}")
        print(f"  MIP Nodes:        {result_log['mip_nodes']}")
        print(f"  MIP Gap:          {result_log['mip_gap']}")
        print("="*64)

    except Exception as e:
        print(f"\n!!! An error occurred during the experiment: {e}", file=sys.stderr)
        result_log['model_status'] = f'ERROR: {str(e).replace(",", ";")}'

    # 4. CSV 파일에 결과 한 줄 추가함
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
        print('Usage: python3 run_experiment.py <n> <k> <q> "<weights>"')
        print('Example: python3 run_experiment.py 34 3 8 "28,32"')
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
        q = int(sys.argv[3])
        # "w1,w2,..." 형태의 가중치 문자열을 숫자 리스트로 변환함
        weights_str = sys.argv[4]
        weights = [int(w.strip()) for w in weights_str.split(',')]
    except ValueError as e:
        print(f"Error: Invalid arguments. n, k, q must be integers, and weights a comma-separated list of numbers. {e}", file=sys.stderr)
        sys.exit(1)

    results_csv = "improved_vs_baseline.csv"
    run_experiment(n, k, q, weights, results_csv)

if __name__ == "__main__":
    main()
