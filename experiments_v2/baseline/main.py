import sys
import time
import numpy as np
import itertools

# HiGHS 라이브러리 확인
try:
    import highspy
except ImportError:
    print("Error: 'highspy' library is not installed. Please run 'pip install highspy'")
    sys.exit(1)

# Galois 라이브러리 확인 (기하 생성을 위해 필요)
try:
    import galois
except ImportError:
    print("Error: 'galois' library is not installed. Please run 'pip install galois'")
    sys.exit(1)

def generate_projective_geometry(k, q):
    """
    PG(k-1, q)의 점들과 인시던스 행렬을 생성합니다.
    """
    GF = galois.GF(q)
    
    # 1. 모든 가능한 벡터 생성 (0 벡터 제외)
    # k가 작으므로 itertools로 충분함
    points = []
    for vec in itertools.product(range(q), repeat=k):
        if all(v == 0 for v in vec):
            continue
        
        # 정규화: 첫 번째 0이 아닌 성분이 1이 되도록 함
        vec = list(vec)
        first_nz_idx = next((i for i, x in enumerate(vec) if x != 0), -1)
        
        if vec[first_nz_idx] == 1:
            points.append(vec)
            
    points_matrix = GF(points).T # (k, N)
    
    # 2. 인시던스 행렬 생성 (점과 초평면의 내적이 0)
    # PG(k-1, q)에서 점과 초평면의 개수는 같음 (Self-dual)
    dot_products = points_matrix.T @ points_matrix
    incidence = (dot_products == 0).astype(int)
    
    return incidence

def solve_mip(n, k, q, allowed_weights):
    print(f"Generating geometry for PG({k-1}, {q})...")
    start_gen = time.time()
    incidence = generate_projective_geometry(k, q)
    gen_time = time.time() - start_gen
    num_points = incidence.shape[0]
    print(f"Geometry generated in {gen_time:.4f}s. Points: {num_points}")
    
    print(f"Building HiGHS MIP model for n={n}, allowed_weights={allowed_weights}...")
    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    h.setOptionValue("presolve", "on")
    h.setOptionValue("time_limit", 300.0)
    
    # --- 변수 설정 ---
    # x_j: 각 점의 중복도 (Integer >= 0)
    h.addVars(num_points, [0.0]*num_points, [float(n)]*num_points)
    for i in range(num_points):
        h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    
    # y_{i, w}: 초평면 i가 가중치 w를 선택했는지 여부 (Binary)
    num_weights = len(allowed_weights)
    num_y_vars = num_points * num_weights
    y_start_idx = num_points
    
    h.addVars(num_y_vars, [0.0]*num_y_vars, [1.0]*num_y_vars)
    for i in range(num_y_vars):
        h.changeColIntegrality(y_start_idx + i, highspy.HighsVarType.kInteger)
        
    # --- 제약 조건 ---
    # 1. 총 점의 개수 합 = n
    col_indices = list(range(num_points))
    h.addRow(float(n), float(n), len(col_indices), np.array(col_indices, dtype=np.int32), np.array([1.0]*num_points, dtype=np.float64))
    
    print("Adding constraints...")
    for i in range(num_points):
        # 2. 각 초평면의 가중치 제약: sum(x in H) - sum(w * y_w) = 0
        p_indices = np.where(incidence[i] == 1)[0]
        
        row_idx = list(p_indices)
        row_coeffs = [1.0] * len(p_indices)
        
        base_y = y_start_idx + i * num_weights
        for w_idx, w in enumerate(allowed_weights):
            row_idx.append(base_y + w_idx)
            row_coeffs.append(-float(w))
            
        h.addRow(0.0, 0.0, len(row_idx), np.array(row_idx, dtype=np.int32), np.array(row_coeffs, dtype=np.float64))
        
        # 3. 유일한 가중치 선택: sum(y_w) = 1
        y_indices = [base_y + w_idx for w_idx in range(num_weights)]
        h.addRow(1.0, 1.0, len(y_indices), np.array(y_indices, dtype=np.int32), np.array([1.0]*num_weights, dtype=np.float64))
        
    print("Solving...")
    start_solve = time.time()
    status = h.run()
    solve_time = time.time() - start_solve
    
    print(f"Status: {status}")
    print(f"Solve Time: {solve_time:.4f}s")
    
    if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
        print("Optimal solution found (Unique_Solutions=1 check passed).")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python main.py n k q \"w1,w2,...\"")
        sys.exit(1)
    
    solve_mip(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), [int(x) for x in sys.argv[4].split(',')])