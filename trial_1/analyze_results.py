import csv
import os

def generate_markdown_report(csv_file):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    print("### 실험 결과 비교 분석 (Latest Runs)\n")
    print("| Case (n, k, q) | Weights | ILP Status | ILP Time (s) | Heuristic Status | Heuristic Time (s) | Heuristic Cost |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 동일 케이스에 대해 여러 실험이 있을 경우, 가장 마지막(최신) 결과만 유지
        latest_runs = {}
        for row in reader:
            # Key: (n, k, q, Weights)
            key = (row['n'], row['k'], row['q'], row['Weights'])
            latest_runs[key] = row
        
        for key, row in latest_runs.items():
            n, k, q, weights = key
            ilp_status = row['ILP_Status']
            ilp_time = row['ILP_Time']
            heu_status = row['Heuristic_Status']
            heu_time = row['Heuristic_Time']
            heu_cost = row['Heuristic_Cost']
            
            print(f"| n={n}, k={k}, q={q} | {weights} | {ilp_status} | {ilp_time} | {heu_status} | {heu_time} | {heu_cost} |")

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv")
    generate_markdown_report(csv_path)