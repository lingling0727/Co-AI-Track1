import time
import sys
import os
import csv
import datetime

# 필수 모듈 임포트 체크
try:
    # 같은 디렉토리에 있는 모듈들을 import 합니다.
    # (geometry.py, ilp_model.py, checker.py 파일이 같은 폴더에 있어야 합니다)
    from geometry import generate_projective_points
    from ilp_model import CodeExtender
    from checker import verify_solution, filter_isomorphic_solutions
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Details: {e}")
    print("Ensure 'geometry.py', 'ilp_model.py', 'checker.py' are in the same folder.")
    sys.exit(1)

def save_geometry_data(k, q, points):
    """생성된 기하학적 데이터를 텍스트 파일로 저장합니다."""
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

def save_experiment_results(n, k, q, weights, num_points, phase0_time, phase1_time, phase2_time, total_sols, unique_sols, nodes_visited=0, pruned_nodes=0):
    """실험 결과를 CSV 파일로 저장합니다."""
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv")
    file_exists = os.path.isfile(filename)
    
    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 파일이 처음 생성될 때만 헤더 작성
            if not file_exists:
                writer.writerow([
                    "Timestamp", "n", "k", "q", "Weights", "Points",
                    "Phase0_Time", "Phase1_Time", "Phase2_Time", "Existence_Status",
                    "Total_Solutions", "Unique_Solutions", "Nodes_Visited", "Pruned_Nodes"
                ])
            
            status = "Feasible" if total_sols > 0 else "Infeasible"
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                n, k, q, str(list(weights)), num_points,
                f"{phase0_time:.4f}", f"{phase1_time:.4f}", f"{phase2_time:.4f}", status, total_sols, unique_sols, nodes_visited, pruned_nodes
            ])
        print(f"  > [Logged] Experiment results saved to '{filename}'")
    except Exception as e:
        print(f"  > [Error] Failed to save experiment results: {e}")

def run_classification(n, k, q, d_or_weights, base_code_counts=None, points_km1=None):
    """
    주어진 파라미터로 선형 부호 분류를 수행하는 메인 함수
    d_or_weights: 최소 거리(d) 정수 또는 가중치 리스트 문자열("0,1,2")
    """
    
    # --- [핵심 개선] 입력값 파싱 로직 ---
    target_weights = set()
    try:
        # 입력이 정수 형태의 문자열("3")이면 d값으로 해석
        if isinstance(d_or_weights, int) or (isinstance(d_or_weights, str) and ',' not in d_or_weights):
            d = int(d_or_weights)
            max_weight = n - d
            if max_weight < 0:
                print(f"[Error] n({n}) must be greater than or equal to d({d}).")
                return
            # 0부터 n-d까지 모든 가중치를 허용해야 해를 찾을 수 있음
            target_weights = set(range(max_weight + 1))
            print(f"[*] Input interpreted as d={d}. Auto-generated Allowed Weights (<= {n}-{d}): {target_weights}")
        else:
            # 콤마가 있으면 가중치 리스트로 해석 ("0,1,2,3")
            target_weights = set(map(int, d_or_weights.split(',')))
            print(f"[*] Input interpreted as explicit weights: {target_weights}")
            
    except ValueError:
        print(f"Error: Invalid format for d or weights '{d_or_weights}'. Should be an integer d or comma-separated weights.")
        return

    print("="*60)
    mode = "EXTENSION" if base_code_counts else "CONSTRUCTION (Scratch)"
    print(f"[*] Starting {mode} for [n={n}, k={k}, d={d if 'd' in locals() else 'Custom'}]_q={q} code")
    print(f"[*] Target Hyperplane Weights: {target_weights}")
    print("="*60)

    # --- 단계 1: 기하학적 구조 생성 ---
    print("\n[1] Generating Projective Geometry...")
    
    # [Safety Check] 예상되는 점의 개수 계산
    try:
        expected_points = (q**k - 1) // (q - 1)
        print(f"  > Estimated points in PG({k-1}, {q}): {expected_points:,}")
        if expected_points > 50_000: # 메모리 보호를 위해 제한
            print(f"  > [Error] Geometry is too large (> 50,000 points). Execution aborted.")
            return
    except OverflowError:
        print("  > [Error] Parameters are too large.")
        return

    start_geom = time.time()
    try:
        # geometry.py의 함수 호출 (반드시 galois 라이브러리 사용하는 버전이어야 함)
        points = generate_projective_points(k, q)
        hyperplanes = points # PG(k-1, q)에서 점과 초평면은 동형
        num_points = len(points)
        
        if num_points != expected_points:
            print(f"  > Warning: Generated {num_points} points, but expected {expected_points}.")
            
    except Exception as e:
        print(f"  > [Error] Failed to generate geometry: {e}")
        return
    end_geom = time.time()
    print(f"  > Generated {len(points)} points/hyperplanes for PG({k-1}, {q}) in {end_geom - start_geom:.4f}s.")
    
    save_geometry_data(k, q, points)

    # --- 단계 2: ILP 모델 생성 및 해 열거 (Phase 1) ---
    print("\n[2] Building ILP model and starting enumeration (Phase 1)...")
    start_ilp = time.time()
    try:
        extender = CodeExtender(n=n, k=k, q=q, target_weights=target_weights)
        # ILP 풀이 시작
        solutions, nodes_visited, pruned_nodes = extender.build_and_solve(points, hyperplanes, base_code_counts, points_km1)
        
    except ImportError:
         print("  > [Error] 'highspy' library is not installed. Please run 'pip install highspy'.")
         return
    except Exception as e:
        print(f"  > [Error] An error occurred during ILP solving: {e}")
        return
    end_ilp = time.time()
    ilp_time = end_ilp - start_ilp
    print(f"  > Phase 1 finished in {ilp_time:.4f}s. Found {len(solutions)} candidate solution(s).")

    if not solutions:
        print("\n[*] No solutions found. Try checking if 'd' is too large for this n,k,q.")
        save_experiment_results(n, k, q, target_weights, num_points, 0.0, ilp_time, 0.0, 0, 0, nodes_visited, pruned_nodes)
        return

    # --- 단계 3: 해 검증 및 동형성 필터링 (Phase 2) ---
    print("\n[3] Verifying and filtering solutions (Phase 2)...")
    start_check = time.time()
    
    # 1. 검증 (Verification)
    verified_solutions = [sol for sol in solutions if verify_solution(sol, n, target_weights, points, hyperplanes, q)]
    if len(verified_solutions) != len(solutions):
        print(f"  > Warning: {len(solutions) - len(verified_solutions)} solutions failed verification.")

    # 2. 동형 제거 (Isomorphism Check)
    unique_solutions = filter_isomorphic_solutions(verified_solutions, n, points, hyperplanes, q)
    end_check = time.time()
    phase2_time = end_check - start_check
    
    print(f"  > Phase 2 finished in {phase2_time:.4f}s.")
    print(f"  > Found {len(unique_solutions)} non-equivalent canonical code(s).")

    # --- 최종 결과 출력 ---
    print("\n[*] Final Results:")
    if not unique_solutions:
        print("  No valid codes found after filtering.")
    else:
        for i, sol in enumerate(unique_solutions[:10]):
            print(f"  - Code #{i+1}: {sol}")
        if len(unique_solutions) > 10:
            print(f"  ... and {len(unique_solutions) - 10} more codes.")
            
    print("\nClassification finished.")
    
    # 결과 파일 저장
    save_experiment_results(n, k, q, target_weights, num_points, 0.0, ilp_time, phase2_time, len(solutions), len(unique_solutions), nodes_visited, pruned_nodes)
    
    return unique_solutions, points

if __name__ == "__main__":
    # --- 명령행 인자 처리 ---
    if len(sys.argv) < 5:
        print("Usage: python main.py <n> <k> <q> <d_or_weights>")
        print("Example 1 (Recommend): python main.py 6 3 3 3       (Input d=3)")
        print("Example 2 (Advanced):  python main.py 6 3 3 \"0,1,2,3\" (Input Weights)")
        sys.exit(1)
    
    # 파라미터 파싱
    try:
        n_in = int(sys.argv[1])
        k_in = int(sys.argv[2])
        q_in = int(sys.argv[3])
        d_or_w_in = sys.argv[4] # 문자열로 받아서 함수 내부에서 처리
        
        run_classification(n_in, k_in, q_in, d_or_w_in)
        
    except ValueError:
        print("[Error] n, k, q must be integers.")
        sys.exit(1)
