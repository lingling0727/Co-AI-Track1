import sys
import time
import numpy as np
import itertools
import csv
import os

# 필수 라이브러리 체크
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

# ==========================================
# 1. 기하 구조 생성 (Projective Geometry)
# ==========================================
def generate_projective_geometry(k, q):
    """
    PG(k-1, q)의 점들과 인시던스 행렬을 생성합니다.
    """
    GF = galois.GF(q)
    points = []
    
    # 모든 가능한 벡터 생성 (0 벡터 제외)
    for vec in itertools.product(range(q), repeat=k):
        if all(v == 0 for v in vec):
            continue
        
        vec = list(vec)
        # 정규화: 첫 번째 0이 아닌 성분이 1이 되도록 함 (Projective Point 정의)
        first_nz_idx = next((i for i, x in enumerate(vec) if x != 0), -1)
        if vec[first_nz_idx] == 1:
            points.append(vec)
            
    points_matrix = GF(points).T
    
    # 인시던스 행렬 생성 (내적 = 0)
    dot_products = points_matrix.T @ points_matrix
    incidence = (dot_products == 0).astype(int)
    
    return np.array(points), incidence

# ==========================================
# 2. LP (Linear Programming) 가지치기
# ==========================================
def solve_lp_relaxation(current_solution_indices, incidence, n, d, k, q):
    """
    현재 선택된 점들을 포함하면서, 남은 점들을 추가했을 때
    최소 거리 d를 만족하는 해가 존재할 수 있는지 LP로 확인 (Relaxation)
    """
    num_points = incidence.shape[0]
    
    h = highspy.Highs()
    h.setOptionValue("output_flag", False) # 로그 끔 (속도 향상)
    
    # 변수: x_j (각 점의 선택 여부, 0 <= x_j <= 1, 실수)
    # 이미 선택된 점은 1로 고정, 나머지는 0~1
    lower_bounds = [0.0] * num_points
    upper_bounds = [1.0] * num_points
    
    for idx in current_solution_indices:
        lower_bounds[idx] = 1.0 # 이미 선택됨
    
    h.addVars(num_points, lower_bounds, upper_bounds)
    
    # 제약조건 1: 총 점의 개수는 n개여야 함
    # sum(x_j) = n
    col_indices = np.array(range(num_points), dtype=np.int32)
    coeffs = np.array([1.0] * num_points, dtype=np.float64)
    h.addRow(float(n), float(n), len(col_indices), col_indices, coeffs)
    
    # 제약조건 2: 모든 초평면(Hyperplane)에 대해 점의 개수는 n-d 이하여야 함
    # sum(x_j in H_i) <= n - d
    max_points_in_hyperplane = n - d
    
    for i in range(num_points): # 모든 초평면에 대해
        p_indices = np.where(incidence[i] == 1)[0]
        row_idx = np.array(p_indices, dtype=np.int32)
        row_coeffs = np.array([1.0] * len(p_indices), dtype=np.float64)
        
        # Upper bound만 설정 (-infinity ~ max_points)
        h.addRow(-highspy.kHighsInf, float(max_points_in_hyperplane), len(row_idx), row_idx, row_coeffs)
        
    # 목적 함수는 딱히 필요 없으므로 0으로 설정 (Feasibility Check 위주)
    # 하지만 LP의 특성상 max sum(x) 같은 걸 둬도 됨. 여기선 Feasibility만 봄.
    
    h.run()
    status = h.getModelStatus()
    
    # 해가 존재하면(Optimal) True, 불가능하면(Infeasible) False
    if status == highspy.HighsModelStatus.kOptimal:
        return True
    else:
        return False

# ==========================================
# 3. Hybrid Backtracking Solver (대칭성 제거 적용)
# ==========================================
class HybridSolver:
    def __init__(self, n, k, d, q):
        self.n = n
        self.k = k
        self.d = d
        self.q = q
        self.points, self.incidence = generate_projective_geometry(k, q)
        self.num_points = len(self.points)
        self.found_solution = None
        self.nodes_visited = 0
        self.lp_calls = 0
        
        # Griesmer Bound 확인
        g_bound = 0
        for i in range(k):
            g_bound += np.ceil(d / (q**i))
        self.griesmer = int(g_bound)
        
        print(f"[Init] Griesmer Bound for [{n}, {k}, {d}]_{q}: n >= {self.griesmer}")
        if n < self.griesmer:
            print("[Warning] n is smaller than Griesmer Bound. Solution unlikely.")

    def find_standard_basis_indices(self):
        """
        점 리스트에서 표준 기저 벡터(Standard Basis)에 해당하는 인덱스를 찾음.
        예: [1,0,0], [0,1,0], [0,0,1] ...
        """
        basis_indices = []
        identity = np.eye(self.k, dtype=int) # 단위 행렬 생성
        
        for row in identity:
            row_list = list(row)
            # points 리스트에서 해당 벡터와 일치하는 인덱스 찾기
            for idx, p in enumerate(self.points):
                if list(p) == row_list:
                    basis_indices.append(idx)
                    break
                    
        if len(basis_indices) != self.k:
            print("[Error] Could not find all standard basis vectors in points.")
            return []
            
        return basis_indices

    def is_valid(self, current_solution):
        """
        현재 솔루션이 코드의 조건(최소 거리 d)을 만족하는지 검사
        즉, 어떤 초평면에도 (n-d)개보다 많은 점이 있으면 안 됨.
        현재 단계에서는 '아직 n개가 안 찼어도' 이미 초과했으면 False.
        """
        # 현재까지 선택된 점들의 인덱스
        indices = current_solution
        if not indices:
            return True
            
        # 각 초평면에 포함된 점의 개수 계산
        # incidence 행렬의 하위 집합만 가져와서 합산
        sub_incidence = self.incidence[:, indices] # (num_points, len(indices))
        counts = np.sum(sub_incidence, axis=1)
        
        max_allowed = self.n - self.d
        if np.max(counts) > max_allowed:
            return False
        return True

    def solve(self):
        start_time = time.time()
        
        # --- 핵심: 대칭성 제거 (Basis Fixing) ---
        # 1. 표준 기저 벡터들의 인덱스를 먼저 찾습니다.
        basis_indices = self.find_standard_basis_indices()
        
        print(f"Applying Basis Fixing: Fixing first {self.k} points to Standard Basis.")
        print(f"Fixed Indices: {basis_indices}")
        
        # 2. 솔루션 리스트에 미리 넣어두고 탐색 시작
        # 이렇게 하면 깊이(Depth)가 0이 아니라 k부터 시작됩니다.
        initial_solution = basis_indices[:]
        
        # 재귀 탐색 시작
        self.backtrack(initial_solution)
        
        elapsed_time = time.time() - start_time
        return self.found_solution, elapsed_time

    def backtrack(self, current_solution):
        # 1. 종료 조건: 해를 찾았거나 이미 찾은 경우
        if self.found_solution is not None:
            return

        self.nodes_visited += 1
        
        # 2. 유효성 검사 (기하학적 제약)
        if not self.is_valid(current_solution):
            return

        # 3. 성공 조건: n개의 점을 모두 선택함
        if len(current_solution) == self.n:
            self.found_solution = current_solution
            return

        # 4. LP 가지치기 (일정 깊이 이상에서만 수행 - 성능 최적화)
        # 너무 얕은 깊이에서는 LP가 느릴 수 있음. 여기선 항상 수행하거나 조정 가능.
        if len(current_solution) >= self.k: # 기저 고정 이후부터 체크
            self.lp_calls += 1
            if not solve_lp_relaxation(current_solution, self.incidence, self.n, self.d, self.k, self.q):
                return # LP가 Infeasible하면 이 가지는 가망 없음 -> Pruning

        # 5. 다음 점 선택 (Branching)
        # 대칭성 제거 2단계: 순서 강제 (Order Enforcing)
        # 이전에 넣은 점의 인덱스보다 더 큰 인덱스만 고려함.
        last_index = current_solution[-1] if current_solution else -1
        
        # 남은 필요한 점의 개수
        remaining_needed = self.n - len(current_solution)
        
        # 앞으로 탐색할 수 있는 점들의 범위
        # (남은 점 개수만큼은 뒤에 남아 있어야 함)
        start_idx = last_index + 1
        end_idx = self.num_points - remaining_needed + 1
        
        for i in range(start_idx, end_idx):
            # 다음 단계로 진행
            current_solution.append(i)
            self.backtrack(current_solution)
            if self.found_solution: return
            current_solution.pop() # Backtrack

# ==========================================
# 4. 메인 실행 및 저장
# ==========================================
def run_experiment(n, k, d, q):
    print(f"\n=== Hybrid Pruning Experiment (Optimized) ===")
    print(f"Parameters: n={n}, k={k}, d={d}, q={q}")
    
    solver = HybridSolver(n, k, d, q)
    solution, duration = solver.solve()
    
    status = "Optimal (Found)" if solution else "Infeasible / Not Found"
    
    print(f"\n[Done] Status: {status}")
    print(f"       Time: {duration:.4f}s")
    print(f"       Nodes Visited: {solver.nodes_visited}")
    print(f"       LP Calls: {solver.lp_calls}")
    
    if solution:
        print(f"       Solution Indices: {solution}")
        # 실제 점 좌표 출력 (앞 5개만)
        coords = [solver.points[i] for i in solution]
        print(f"       Coordinates (first 5): {coords[:5]} ...")

    # 결과 저장
    filename = f"hybrid_opt_result_n{n}_k{k}_d{d}_q{q}.csv"
    save_result(n, k, d, q, status, duration, solver.nodes_visited, solver.lp_calls, filename)

def save_result(n, k, d, q, status, time_sec, nodes, lp_calls, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = ['n', 'k', 'd', 'q', 'Method', 'Time(s)', 'Nodes', 'LP_Calls', 'Status']
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'n': n, 'k': k, 'd': d, 'q': q,
            'Method': 'Hybrid (BasisFixing+LP)',
            'Time(s)': round(time_sec, 6),
            'Nodes': nodes,
            'LP_Calls': lp_calls,
            'Status': status
        })
    print(f"[SUCCESS] Results saved to '{filename}'")

if __name__ == "__main__":
    # 사용자 입력 처리
    if len(sys.argv) < 5:
        print("Usage: python hybrid_optimized.py n k d q")
        print("Example: python hybrid_optimized.py 7 3 2 4")
        sys.exit(1)
        
    n_in = int(sys.argv[1])
    k_in = int(sys.argv[2])
    d_in = int(sys.argv[3])
    q_in = int(sys.argv[4])
    
    run_experiment(n_in, k_in, d_in, q_in)
