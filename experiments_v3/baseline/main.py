import sys
import subprocess
import re
import pandas as pd
import os

def parse_output(output):
    """터미널 출력에서 상태, 시간, 노드, 반복 횟수를 추출함"""
    # Baseline과 Improved의 다양한 출력 형식을 모두 잡는 정규식
    status = re.search(r"(?:Model\s+)?Status:\s*([^\n\r]+)", output, re.I)
    time_val = re.search(r"Solve\s*Time(?:\s*\(sec\))?:\s*([\d.]+)", output, re.I)
    nodes = re.search(r"(?:MIP\s+)?Nodes:\s*(\d+)", output, re.I)
    
    return {
        'Status': status.group(1).strip() if status else "N/A",
        'Time': float(time_val.group(1)) if time_val else 0.0,
        'Nodes': int(nodes.group(1)) if nodes else 0
    }

def main():
    if len(sys.argv) != 5:
        print('Usage: python benchmark_once.py <n> <k> <q> "<weights>"')
        sys.exit(1)

    n, k, q, weights = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    print(f"[*] Starting Comparison for n={n}, k={k}, q={q}, weights={weights}")
    
    # 1. Baseline (main.py) - 기존 재현 모델
    print(f"    > Running Baseline (main.py)...")
    cmd_base = [sys.executable, "main.py", n, k, q, weights]
    proc_base = subprocess.run(cmd_base, capture_output=True, text=True)
    if proc_base.returncode != 0:
        print(f"      [Error] Baseline failed: {proc_base.stderr.strip()}")
        res_base = {'Status': 'RuntimeError', 'Time': 0, 'Nodes': 0}
    else:
        res_base = parse_output(proc_base.stdout)

    # 2. Improved (run_experiment.py) - 개선된 Kurz 모델
    print(f"    > Running Improved (run_experiment.py)...")
    cmd_imp = [sys.executable, "run_experiment.py", n, k, q, weights]
    proc_imp = subprocess.run(cmd_imp, capture_output=True, text=True)
    if proc_imp.returncode != 0:
        print(f"      [Error] Improved failed: {proc_imp.stderr.strip()}")
        res_imp = {'Status': 'RuntimeError', 'Time': 0, 'Nodes': 0}
    else:
        res_imp = parse_output(proc_imp.stdout)

    # 3. 상세 비교 리포트 생성
    comparison = {
        'Metric': ['ILP Formulation', 'Status', 'Solve Time (s)', 'MIP Nodes'],
        'Baseline (Reproduction)': [
            f'Simple ILP (sum(x) = w)', 
            res_base['Status'], 
            f"{res_base['Time']:.4f}", 
            res_base['Nodes']
        ],
        'Improved (Our Creation)': [
            f'Kurz ILP (sum(x) = n-w)', 
            res_imp['Status'], 
            f"{res_imp['Time']:.4f}", 
            res_imp['Nodes']
        ]
    }
    df = pd.DataFrame(comparison)
    
    print("\n" + "="*75)
    print(f"   [RESEARCH REPORT] n={n}, k={k}, q={q}, weights={weights}")
    print("="*75)
    print(df.to_string(index=False))
    print("-"*75)
    
    # 개선 지표 분석
    if res_base['Time'] > 0 and res_imp['Time'] > 0:
        print(f"  > Speedup: {res_base['Time']/res_imp['Time']:.2f}x faster")
    if res_base['Nodes'] > 0:
        reduction = (res_base['Nodes'] - res_imp['Nodes']) / res_base['Nodes'] * 100
        print(f"  > Node Reduction: {reduction:.2f}% (Efficiency gain)")
    print("="*75)

if __name__ == "__main__":
    main()