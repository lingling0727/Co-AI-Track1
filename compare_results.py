import csv
import os
import sys

def load_results(filepath, is_baseline=False):
    """
    CSV 파일을 읽어서 (n, k, q, weights)를 키로 하는 딕셔너리로 반환합니다.
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 헤더 이름이 baseline과 trial2에서 다를 수 있으므로 매핑 처리
            try:
                if is_baseline:
                    # Baseline Header: n, k, q, Weights, Phase1_Time, Nodes_Visited
                    n = int(row['n'])
                    k = int(row['k'])
                    q = int(row['q'])
                    weights = row['Weights']
                    pre_time = float(row['Phase0_Time'])
                    search_time = float(row['Phase1_Time'])
                    nodes = int(row.get('Nodes_Visited', 0))
                    pruned = int(row.get('Pruned_Nodes', 0))
                    status = "Feasible" if int(row['Total_Solutions']) > 0 else "Infeasible"
                else:
                    # Trial2 Header: Length(n), Dimension(k), Field(q), Target_Weights, Search_Time(s), Nodes_Visited
                    n = int(row['Length(n)'])
                    k = int(row['Dimension(k)'])
                    q = int(row['Field(q)'])
                    weights = row['Target_Weights']
                    pre_time = float(row.get('Precomp_Time(s)', 0.0))
                    search_time = float(row['Search_Time(s)'])
                    nodes = int(row.get('Nodes_Visited', 0))
                    pruned = int(row.get('Pruned_Nodes', 0))
                    status = row['Existence_Status']
                
                # 키 생성 (Weights 문자열 포맷이 다를 수 있으니 공백 제거 후 비교)
                weights_key = weights.replace(" ", "")
                key = (n, k, q, weights_key)
                
                # 최신 실험 결과로 덮어쓰기 (같은 파라미터 실험이 여러 번일 경우)
                data[key] = {
                    'pre_time': pre_time,
                    'search_time': search_time,
                    'nodes': nodes,
                    'pruned': pruned,
                    'status': status
                }
            except KeyError as e:
                print(f"[Error] Missing column in {filepath}: {e}")
                continue
            except ValueError:
                continue
    return data

def main():
    print("="*50)
    print("[*] Comparing Baseline vs Trial2 Results")
    print("="*50)

    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Baseline 결과 파일 경로 찾기
    # 우선순위 A: 현재 폴더의 하위 폴더 (./baseline/experiment_results.csv)
    baseline_csv = os.path.join(current_dir, 'baseline', 'experiment_results.csv')
    # 우선순위 B: 상위 폴더의 형제 폴더 (../baseline/experiment_results.csv)
    if not os.path.exists(baseline_csv):
        baseline_csv = os.path.join(current_dir, '..', 'baseline', 'experiment_results.csv')

    # 2. Trial2 결과 파일 경로 (trial2 폴더 내부)
    trial2_csv = os.path.join(current_dir, 'trial2', 'experiment_results.csv')

    baseline_data = load_results(baseline_csv, is_baseline=True)
    trial2_data = load_results(trial2_csv, is_baseline=False)

    if not baseline_data:
        print("No baseline data found. Please run baseline experiments first.")
    if not trial2_data:
        print("No trial2 data found. Please run trial2 experiments first.")

    # 3. 비교 및 결과 저장
    output_csv = os.path.join(current_dir, 'comparison_summary.csv')
    
    # 공통된 키와 각자만 있는 키 합집합
    all_keys = sorted(list(set(baseline_data.keys()) | set(trial2_data.keys())))
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "k", "q", "Weights",
            "Base_Pre(s)", "Trial2_Pre(s)", "Base_Search(s)", "Trial2_Search(s)",
            "Baseline_Time(s)", "Trial2_Time(s)", "Time_Diff(s)", "Speedup(x)",
            "Baseline_Nodes", "Trial2_Nodes", "Nodes_Diff",
            "Baseline_Pruned", "Trial2_Pruned", "Pruned_Diff",
            "Baseline_Status", "Trial2_Status"
        ])

        for key in all_keys:
            n, k, q, w = key
            base = baseline_data.get(key, {})
            trial = trial2_data.get(key, {})

            b_pre = base.get('pre_time', 0.0)
            b_search = base.get('search_time', 0.0)
            t_pre = trial.get('pre_time', 0.0)
            t_search = trial.get('search_time', 0.0)
            
            b_total = b_pre + b_search if base else 'N/A'
            t_total = t_pre + t_search if trial else 'N/A'
            
            b_nodes = base.get('nodes', 'N/A')
            t_nodes = trial.get('nodes', 'N/A')
            b_pruned = base.get('pruned', 'N/A')
            t_pruned = trial.get('pruned', 'N/A')
            b_status = base.get('status', 'N/A')
            t_status = trial.get('status', 'N/A')

            # 계산 가능한 경우 차이 계산
            time_diff = f"{b_total - t_total:.4f}" if isinstance(b_total, float) and isinstance(t_total, float) else "N/A"
            speedup = f"{b_total / t_total:.2f}" if isinstance(b_total, float) and isinstance(t_total, float) and t_total > 0 else "N/A"
            nodes_diff = b_nodes - t_nodes if isinstance(b_nodes, int) and isinstance(t_nodes, int) else "N/A"
            pruned_diff = b_pruned - t_pruned if isinstance(b_pruned, int) and isinstance(t_pruned, int) else "N/A"

            writer.writerow([
                n, k, q, w, 
                f"{b_pre:.4f}" if base else "N/A", f"{t_pre:.4f}" if trial else "N/A",
                f"{b_search:.4f}" if base else "N/A", f"{t_search:.4f}" if trial else "N/A",
                f"{b_total:.4f}" if base else "N/A", f"{t_total:.4f}" if trial else "N/A",
                time_diff, speedup, b_nodes, t_nodes, nodes_diff, b_pruned, t_pruned, pruned_diff, b_status, t_status
            ])

    print(f"\n[*] Comparison saved to: {output_csv}")

if __name__ == "__main__":
    main()