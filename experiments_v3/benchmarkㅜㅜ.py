import subprocess
import re
import pandas as pd
import time
import os

# 1. 실험할 파라미터 리스트 (n, k, q, weights)
params_list = [
    (7, 3, 2, "4"), (15, 4, 2, "8"), (4, 2, 3, "3"), (13, 3, 3, "9"),
    (6, 3, 4, "4"), (6, 2, 5, "5"), (8, 2, 7, "7"), (10, 2, 9, "9"),
    (34, 3, 8, "28,32"), (65, 3, 4, "48,56")
]

def parse_output(output):
    """터미널 출력에서 상태, 시간, 노드 수를 추출함"""
    # Baseline(main.py)용 정규식
    status = re.search(r"Status:\s*([^\n\r]+)", output)
    time_val = re.search(r"Solve\s*Time(?:\s*\(sec\))?:\s*([\d.]+)", output)
    nodes = re.search(r"(?:MIP\s*)?Nodes:\s*(\d+)", output)
    
    return {
        'Status': status.group(1).strip() if status else "N/A",
        'Time': float(time_val.group(1)) if time_val else 0.0,
        'Nodes': int(nodes.group(1)) if nodes else 0
    }

def run_all():
    results = []
    print(f"{'ID':<3} | {'Params':<20} | {'Baseline (Time/Nodes)':<25} | {'Improved (Time/Nodes)':<25}")
    print("-" * 85)

    for i, (n, k, q, w) in enumerate(params_list, 1):
        p_str = f"{n} {k} {q} \"{w}\""
        
        # --- Baseline 실행 (main.py) ---
        cmd_base = ["python", "main.py", str(n), str(k), str(q), w]
        try:
            out_base = subprocess.check_output(cmd_base, stderr=subprocess.STDOUT, text=True)
            res_base = parse_output(out_base)
        except Exception: res_base = {'Status': 'Error', 'Time': 0, 'Nodes': 0}

        # --- Improved 실행 (run_experiment.py) ---
        cmd_imp = ["python", "run_experiment.py", str(n), str(k), str(q), w]
        try:
            out_imp = subprocess.check_output(cmd_imp, stderr=subprocess.STDOUT, text=True)
            res_imp = parse_output(out_imp)
        except Exception: res_imp = {'Status': 'Error', 'Time': 0, 'Nodes': 0}

        # --- 결과 정리 ---
        speedup = res_base['Time'] / res_imp['Time'] if res_imp['Time'] > 0 else 0
        node_red = res_base['Nodes'] - res_imp['Nodes']
        
        row = {
            'ID': i, 'n': n, 'k': k, 'q': q, 'Weights': w,
            'Base_Status': res_base['Status'], 'Base_Time': res_base['Time'], 'Base_Nodes': res_base['Nodes'],
            'Exp_Status': res_imp['Status'], 'Exp_Time': res_imp['Time'], 'Exp_Nodes': res_imp['Nodes'],
            'Speedup': round(speedup, 2), 'Nodes_Saved': node_red
        }
        results.append(row)
        
        print(f"{i:<3} | {n},{k},{q} | {res_base['Time']:>7.3f}s / {res_base['Nodes']:>5} | {res_imp['Time']:>7.3f}s / {res_imp['Nodes']:>5}")

    # CSV 저장
    df = pd.DataFrame(results)
    df.to_csv("comparison_summary.csv", index=False)
    print(f"\n[완료] 결과가 'comparison_summary.csv'에 저장되었어.")

if __name__ == "__main__":
    run_all()