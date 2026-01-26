import time
import sys
import os
import csv
import datetime

# [수정] 모듈 경로 설정: trial2 폴더에서 실행 시 상위 폴더(루트)의 모듈을 찾도록 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # 같은 디렉토리에 있는 모듈들을 import 합니다.
    from geometry import generate_projective_points, generate_hyperplanes
    from ilp_model import CodeExtender
    from checker import verify_solution, filter_isomorphic_solutions
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Debug: Python is searching in these paths: {sys.path}")
    print(f"Details: {e}")
    sys.exit(1)

def save_geometry_data(k, q, points):
    """
    생성된 기하학적 데이터를 텍스트 파일로 저장합니다.
    """
    # [수정] 파일 저장 경로를 현재 스크립트 위치 기준으로 명확히 지정
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, f"geometry_k{k}_q{q}.txt")
    
    try:
        with open(filename, "w") as f:
            f.write(f"# Projective Geometry PG({k-1}, {q})\n")
            f.write(f"# Dimension k: {k}, Field size q: {q}\n")
            f.write(f"# Total Points: {len(points)}\n")
            f.write(f"# Format: Point_ID: (coord1, coord2, ...)\n")
            f.write("-" * 40 + "\n")
            for idx, p in enumerate(points):
                f.write(f"{idx}: {p}\n")
        print(f"  > [Saved] Geometry data saved to '{filename}'")
    except Exception as e:
        print(f"  > [Error] Failed to save geometry data: {e}")

def save_experiment_results(n, k, q, weights, num_points, ilp_time, phase2_time, total_sols, unique_sols, nodes_visited, pruned_nodes):
    """
    실험 결과를 CSV 파일로 저장합니다.
    """
    # [수정] 파일 저장 경로를 현재 스크립트 위치 기준으로 명확히 지정
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv")

    # 정의된 헤더
    headers = [
        "Timestamp", "Length(n)", "Dimension(k)", "Field(q)", "Target_Weights", "Num_Points",
        "Existence_Status", "Search_Time(s)", "Verify_Time(s)", 
        "Total_Solutions", "Unique_Solutions", "Nodes_Visited", "Pruned_Nodes"
    ]

    # 파일 상태 및 헤더 일치 여부 확인
    file_exists = os.path.isfile(filename)
    should_write_header = False
    
    if file_exists:
        try:
            with open(filename, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    existing_headers = next(reader)
                    if existing_headers != headers:
                        should_write_header = True
                        file_exists = False # 헤더가 다르면 덮어쓰기 모드로 전환
                except StopIteration:
                    should_write_header = True # 파일이 비어있음
        except Exception:
            should_write_header = True
            file_exists = False
    else:
        should_write_header = True

    mode = 'a' if file_exists else 'w'
    
    try:
        with open(filename, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if should_write_header:
                writer.writerow(headers)
            
            # 해 존재 여부 (Exhaustive Search이므로 0이면 존재하지 않음 증명)
            status = "Feasible" if total_sols > 0 else "Infeasible"

            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                n, k, q, str(list(weights)), num_points,
                status, f"{ilp_time:.4f}", f"{phase2_time:.4f}", total_sols, unique_sols,
                nodes_visited, pruned_nodes
            ])
        print(f"  > [Logged] Experiment results saved to '{filename}'")
    except Exception as e:
        print(f"  > [Error] Failed to save experiment results: {e}")

def run_classification(n, k, q, weights_str, base_code_counts=None, points_km1=None):
    """
    주어진 파라미터로 선형 부호 분류를 수행하는 메인 함수
    """
    try:
        target_weights = set(map(int, weights_str.split(',')))
    except ValueError:
        print(f"Error: Invalid format for weights '{weights_str}'. Should be comma-separated numbers.")
        return

    print("="*50)
    mode = "EXTENSION" if base_code_counts else "CONSTRUCTION (Scratch)"
    print(f"[*] Starting {mode} for [n={n}, k={k}]_q={q} code")
    print(f"[*] Target Non-Zero Weights: {target_weights}")
    print("="*50)

    # --- 단계 1: 기하학적 구조 생성 ---
    print("\n[1] Generating Projective Geometry...")
    start_geom = time.time()
    try:
        points = generate_projective_points(k, q)
        # PG(k-1, q)에서 점과 초평면은 동형이므로 동일한 생성 함수 사용
        hyperplanes = points 
        num_points = len(points)
        expected_points = (q**k - 1) // (q - 1)
        if num_points != expected_points:
            print(f"  > Warning: Generated {num_points} points, but expected {expected_points}.")
    except Exception as e:
        print(f"  > [Error] Failed to generate geometry: {e}")
        return
    end_geom = time.time()
    print(f"  > Generated {len(points)} points/hyperplanes for PG({k-1}, {q}) in {end_geom - start_geom:.4f}s.")
    
    # 기하학 데이터 파일 저장
    save_geometry_data(k, q, points)

    # --- 단계 2: ILP 모델 생성 및 해 열거 (Phase 0 & 1) ---
    print("\n[2] Building ILP model and starting enumeration (Phase 0 & 1)...")
    start_ilp = time.time()
    try:
        extender = CodeExtender(n=n, k=k, q=q, target_weights=target_weights)
        # 확장 모드일 경우 base_code 정보 전달
        solutions, nodes_visited, pruned_nodes = extender.build_and_solve(points, hyperplanes, base_code_counts, points_km1)
        
    except ImportError:
         print("  > [Error] 'gurobipy' is not installed. Please run 'pip install gurobipy'.")
         return
    except Exception as e:
        print(f"  > [Error] An error occurred during ILP solving: {e}")
        return
    end_ilp = time.time()
    ilp_time = end_ilp - start_ilp
    print(f"  > Phase 1 finished in {ilp_time:.4f}s. Found {len(solutions)} candidate solution(s).")

    if not solutions:
        print("\n[*] No solutions found. The code with the given parameters likely does not exist.")
        save_experiment_results(n, k, q, target_weights, num_points, ilp_time, 0.0, 0, 0, nodes_visited, pruned_nodes)
        return

    # --- 단계 3: 해 검증 및 동형성 필터링 (Phase 2) ---
    print("\n[3] Verifying and filtering solutions (Phase 2)...")
    start_check = time.time()
    
    verified_solutions = [sol for sol in solutions if verify_solution(sol, n, target_weights, points, hyperplanes, q)]
    if len(verified_solutions) != len(solutions):
        print(f"  > Warning: {len(solutions) - len(verified_solutions)} solutions failed verification.")

    unique_solutions = filter_isomorphic_solutions(verified_solutions, n, points, hyperplanes, q)
    end_check = time.time()
    phase2_time = end_check - start_check
    print(f"  > Phase 2 finished in {phase2_time:.4f}s.")
    print(f"  > Found {len(unique_solutions)} non-equivalent candidate(s) after basic filtering.")

    # --- 최종 결과 출력 ---
    print("\n[*] Final Results:")
    if not unique_solutions:
        print("  No valid codes found after filtering.")
    else:
        for i, sol in enumerate(unique_solutions[:10]): # 최대 10개까지만 출력
            print(f"  - Solution #{i+1}: {sol}")
        if len(unique_solutions) > 10:
            print(f"  ... and {len(unique_solutions) - 10} more solutions.")
            
    print("\nClassification finished.")
    
    # 결과 파일 저장
    save_experiment_results(n, k, q, target_weights, num_points, ilp_time, phase2_time, len(solutions), len(unique_solutions), nodes_visited, pruned_nodes)
    
    return unique_solutions, points # 다음 확장을 위해 반환


if __name__ == "__main__":
    # --- 논문 재현 테스트 케이스 ---
    # 사용자가 인자를 주지 않았을 때 논문의 케이스를 실행
    if len(sys.argv) == 1:
        print("No arguments provided. Running Paper Replication Test Cases...\n")
        
        # Case 1: Proposition 2 Replication
        # "No projective [35, 4, {28, 32}]_8 code exists."
        # 이를 확인하기 위해 먼저 [n, 3]_8 코드를 찾고 확장 시도 (시간 관계상 바로 4차원 시도)
        # 주의: 8은 소수가 아니므로 geometry.py가 수정되어야 정확히 동작함 (현재는 소수체 가정)
        # 여기서는 논문의 Proposition 1 (q=4) 테스트
        
        # Test: [6, 3, 4]_2 (Hamming-like) -> Extension to [7, 4, 4]_2 (Simplex)
        # 작은 스케일로 확장 로직 검증
        print(">>> Test Case: Extending [3, 2]_2 to [7, 3]_2 (Hamming Code Construction)")
        
        # 1. k=2 (Line)
        # n=3, k=2, q=2, weights={2} (Simplex code)
        sols_k2, points_k2 = run_classification(n=3, k=2, q=2, weights_str="2")
        
        if sols_k2:
            base_sol = sols_k2[0] # 첫 번째 해를 기반으로 확장
            print("\n>>> Extending the found [3, 2]_2 code to k=3...")
            # 2. k=3 (Plane)
            # Target: n=7, k=3, q=2, weights={3, 4} (Hamming)
            run_classification(n=7, k=3, q=2, weights_str="3,4", base_code_counts=base_sol, points_km1=points_k2)
            
    elif len(sys.argv) == 5:
        n_param, k_param, q_param, weights_param = sys.argv[1:]
        run_classification(int(n_param), int(k_param), int(q_param), weights_param)
    else:
        print("Usage: python src/main.py <n> <k> <q> <weights>")
        print("Example: python src/main.py 7 3 2 \"3,4\"")
        print("\nRunning with default example: [n=7, k=3, q=2], weights={3,4}")
        run_classification(n=7, k=3, q=2, weights_str="3,4")