import pandas as pd
import time
import os
import sys

# 현재 디렉토리를 sys.path에 추가하여 모듈 임포트 지원
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from highs_solver import HighsMIPSolver

def run_benchmark(prop_list):
    results = []
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for prop in prop_list:
        print(f"\n{'='*60}")
        print(f">>> Benchmarking {prop}...")
        print(f"{'='*60}")
        
        dataset_path = os.path.join(base_dir, "..", "datasets", prop)
        
        if not os.path.exists(dataset_path):
            print(f"Skip {prop}: Dataset not found at {dataset_path}")
            continue

        # --- 2. HighsMIPSolver (HiGHS ILP) ---
        print(f"\nRunning HighsMIPSolver (MIP)...")
        hms = HighsMIPSolver(dataset_path)
        start = time.time()
        # HiGHS는 내부적으로 노드 수를 관리하므로 solver.solve() 결과를 받아옴
        hms_result = hms.solve() 
        hms_time = time.time() - start
        
        results.append({
            "Proposition": prop,
            "Solver": "HiGHS(MIP)",
            "Time(s)": round(hms_time, 4),
            "Nodes": "Internal", # HiGHS 내부 엔진이 관리
            "Pruned": "N/A",
            "Solutions": 1 if hms_result is not None else 0
        })

    # CSV 저장
    output_path = os.path.join(base_dir, "comparison_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nBenchmark complete. Results saved to '{output_path}'.")

if __name__ == "__main__":
    # 테스트할 프로포지션 목록
    props = ["prop1", "prop2", "prop3", "prop4"]
    run_benchmark(props)