import sys
import time
import numpy as np
import itertools
import csv
import os

# 라이브러리 체크
try:
    import highspy
except ImportError:
    print("Error: 'highspy' library is not installed. Please run 'pip install highspy'")
    sys.exit(1)

try:
    import galois
except ImportError:
    print("Error: 'galois' library is not installed. Please run 'pip install galois'")
    sys.exit(1)

def generate_projective_geometry(k, q):
    """PG(k-1, q)의 점들과 인시던스 행렬 생성"""
    GF = galois.GF(q)
    points = []
    # k가 작으므로 itertools 사용 (효율적)
    for vec in itertools.product(range(q), repeat=k):
        if all(v == 0 for v in vec): continue
        
        vec = list(vec)
        # 정규화 (첫 0이 아닌 성분을 1로)
        first_nz_idx = next((i for i, x in enumerate(vec) if x != 0), -1)
        if vec[first_nz_idx] == 1:
            points.append(vec)
            
    points_matrix = GF(points).T 
    dot_products = points_matrix.T @ points_matrix
    incidence = (dot_products == 0).astype(int)
    return incidence

def get_status_string(status_enum):
    """HiGHS 상태 Enum을 읽기 쉬운 문자열로 변환"""
    if status_enum == highspy.HighsModelStatus.kOptimal:
        return "Optimal"
    elif status_enum == highspy.HighsModelStatus.kInfeasible:
        return "Infeasible"
    elif status_enum == highspy.HighsModelStatus.kUnbounded:
        return "Unbounded"
    elif status_enum == highspy.HighsModelStatus.kTimeLimit:
        return "TimeLimit"
    else:
        return str(status_enum)

def save_to_csv(data, filename):
    """결과를 CSV 파일에 저장 (없으면 생성, 있으면 추가)"""
    file_exists = os.path.isfile(filename)
    
    # 컬럼 순서 정의
    fieldnames = ['n', 'k', 'q', 'weights', 'Method', 'Time(s)', 'Nodes', 'LP_Calls', 'Status']
    
    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"\n[SUCCESS] Results saved to '{filename}'")
    except Exception as e:
        print(f"\n[ERROR] Failed to save CSV: {e}")

def solve_mip(n, k, q, allowed_weights):
    print(f"\nGenerating geometry for PG({k-1}, {q})...")
    start_gen = time.time()
    incidence = generate_projective_geometry(k, q)
    gen_time = time.time() - start_gen
    num_points = incidence.shape[0]
    print(f"Geometry generated in {gen_time:.4f}s. Points: {num_points}")
    
    print(f"Building HiGHS MIP model for n={n}, k={k}, q={q}, weights={allowed_weights}...")
    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    h.setOptionValue("presolve", "on")
    h.setOptionValue("time_limit", 300.0) # 5분 제한
    
    # --- 변수 및 제약 조건 설정 (기존과 동일) ---
    h.addVars(num_points, [0.0]*num_points, [float(n)]*num_points)
    for i in range(num_points):
        h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    
    num_weights = len(allowed_weights)
    num_y_vars = num_points * num_weights
    y_start_idx = num_points
    
    h.addVars(num_y_vars, [0.0]*num_y_vars, [1.0]*num_y_vars)
    for i in range(num_y_vars):
        h.changeColIntegrality(y_start_idx + i, highspy.HighsVarType.kInteger)
        
    col_indices = list(range(num_points))
    h.addRow(float(n), float(n), len(col_indices), np.array(col_indices, dtype=np.int32), np.array([1.0]*num_points, dtype=np.float64))
    
    for i in range(num_points):
        p_indices = np.where(incidence[i] == 1)[0]
        row_idx = list(p_indices)
        row_coeffs = [1.0] * len(p_indices)
        base_y = y_start_idx + i * num_weights
        for w_idx, w in enumerate(allowed_weights):
            row_idx.append(base_y + w_idx)
            row_coeffs.append(-float(w))
        h.addRow(0.0, 0.0, len(row_idx), np.array(row_idx, dtype=np.int32), np.array(row_coeffs, dtype=np.float64))
        
        y_indices = [base_y + w_idx for w_idx in range(num_weights)]
        h.addRow(1.0, 1.0, len(y_indices), np.array(y_indices, dtype=np.int32), np.array([1.0]*num_weights, dtype=np.float64))
        
    print("Solving...")
    start_solve = time.time()
    h.run()
    solve_time = time.time() - start_solve
    
    # --- 결과 추출 및 저장 ---
    info = h.getInfo() # 솔버 통계 정보 가져오기
    model_status = h.getModelStatus()
    status_str = get_status_string(model_status)
    
    print(f"\n=== Solve Finished ===")
    print(f"Status: {status_str}")
    print(f"Time:   {solve_time:.4f}s")
    print(f"Nodes:  {info.mip_node_count}")
    
    # CSV 저장용 데이터 구성
    result_data = {
        'n': n,
        'k': k,
        'q': q,
        'weights': str(allowed_weights).replace(',', ';'), # CSV 쉼표 충돌 방지
        'Method': 'MIP (HiGHS)',
        'Time(s)': round(solve_time, 4),
        'Nodes': info.mip_node_count,
        'LP_Calls': info.simplex_iteration_count,
        'Status': status_str
    }
    
    # 파일명 자동 생성 (예: mip_result_n7_k3_q4.csv)
    csv_filename = f"mip_result_n{n}_k{k}_q{q}.csv"
    save_to_csv(result_data, csv_filename)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python main.py n k q \"w1,w2,...\"")
        print("Example: python main.py 7 3 4 \"4\"")
        sys.exit(1)
    
    # --- 입력 파라미터 안전 확인 ---
    n_in = int(sys.argv[1])
    k_in = int(sys.argv[2])
    q_in = int(sys.argv[3])
    w_str = sys.argv[4]
    w_in = [int(x) for x in w_str.split(',')]
    
    print("="*40)
    print(" [ Parameter Check ]")
    print(f"  n = {n_in}")
    print(f"  k = {k_in}")
    print(f"  q = {q_in}  <-- (Check: Is this correct?)")
    print(f"  weights = {w_in}")
    print("="*40)
    
    solve_mip(n_in, k_in, q_in, w_in)
