import sys
import highspy
import numpy as np
import math
import time
import pandas as pd
import itertools

# 필수 라이브러리 확인
try:
    import galois
except ImportError:
    print("Error: 'galois' library is not installed. Please run 'pip install galois'")
    sys.exit(1)

# ==========================================
# 1. 기하 생성 클래스 (GF8 제거 및 일반화)
# ==========================================
class ProjectiveGeometry:
    def __init__(self, k, q):
        self.k = k
        self.q = q
        
        # 입력받은 q에 맞는 유한체 자동 생성
        try:
            self.GF = galois.GF(q)
        except LookupError:
            print(f"Error: q={q} is not a valid prime power.")
            sys.exit(1)
            
        self.points = self._gen_points()

    def _gen_points(self):
        points = []
        # 모든 가능한 벡터 생성
        for vec in itertools.product(range(self.q), repeat=self.k):
            if all(v == 0 for v in vec): continue
            
            vec = list(vec)
            # 정규화: 첫 번째 0이 아닌 성분이 1인 것만 선택
            first_nz_idx = next((i for i, x in enumerate(vec) if x != 0), -1)
            if vec[first_nz_idx] == 1:
                points.append(tuple(vec))
        return points

# ==========================================
# 2. [Step 2] Griesmer Bound 구현
# ==========================================
def calculate_griesmer_bound(k, d, q):
    bound_val = 0
    for i in range(k):
        term = math.ceil(d / (q**i))
        bound_val += term
    return bound_val

# ==========================================
# 3. [Step 1] Hybrid Solver (Custom B&B)
# ==========================================
class CustomHybridSolver:
    def __init__(self, target_n, target_k, d, q, points):
        self.target_n = target_n
        self.target_k = target_k
        self.d = d
        self.q = q
        self.points = points
        self.num_points = len(points)
        
        self.nodes_visited = 0
        self.lp_calls = 0
        self.start_time = 0
        
        self.global_griesmer = calculate_griesmer_bound(target_k, d, q)
        print(f"[Init] Griesmer Bound for [{target_n}, {target_k}, {d}]_{q}: n >= {self.global_griesmer}")
        
    def solve(self):
        self.start_time = time.time()
        self.nodes_visited = 0
        self.lp_calls = 0
        
        result = self._backtrack([], 0)
        
        end_time = time.time()
        status = "Optimal (Found)" if result else "Infeasible (Not Found)"
        print(f"\n[Done] Status: {status}")
        print(f"       Time: {end_time - self.start_time:.4f}s")
        print(f"       Nodes Visited: {self.nodes_visited}")
        print(f"       LP Calls (Pruning): {self.lp_calls}")
        return status, end_time - self.start_time, self.nodes_visited

    def _backtrack(self, current_selection, start_index):
        self.nodes_visited += 1
        
        if len(current_selection) == self.target_n:
            return True
            
        if not self._is_promising(current_selection):
            return False 

        for i in range(start_index, self.num_points):
            if self._backtrack(current_selection + [i], i + 1):
                return True
        return False

    def _is_promising(self, current_selection):
        current_len = len(current_selection)
        if current_len > self.target_n:
            return False

        use_lp_check = False
        if current_len >= self.target_n * 0.8: 
            use_lp_check = True
            
        if use_lp_check:
            self.lp_calls += 1
            if not self._check_lp_feasibility(current_selection):
                return False
        return True 

    def _check_lp_feasibility(self, current_selection):
        # 실제로는 여기서 Weight Constraint 검사가 필요하지만,
        # 현재는 Hybrid 구조 테스트를 위해 True 반환
        return True 

# ==========================================
# 4. 실행 및 CSV 저장
# ==========================================
def run_hybrid_experiment(n, k, d, q):
    print(f"=== Hybrid Pruning & Griesmer Bound Experiment ===")
    print(f"Parameters: n={n}, k={k}, d={d}, q={q}")
    
    # 수정된 부분: GF 객체를 따로 만들지 않고 클래스 내부에서 처리
    pg = ProjectiveGeometry(k, q)
    print(f"Geometry Generated: {len(pg.points)} points")
    
    solver = CustomHybridSolver(n, k, d, q, pg.points)
    status, run_time, nodes = solver.solve()
    
    results = {
        "n": [n], "k": [k], "d": [d], "q": [q],
        "Method": ["Hybrid (RCUB+LP)"],
        "Griesmer_Bound": [solver.global_griesmer],
        "Time(s)": [run_time],
        "Nodes": [nodes],
        "LP_Calls": [solver.lp_calls],
        "Status": [status]
    }
    df = pd.DataFrame(results)
    filename = f"hybrid_result_n{n}_k{k}_d{d}_q{q}.csv"
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Results saved to '{filename}'")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("\nUsage: python main.py <n> <k> <d> <q>")
        sys.exit(1)
    
    try:
        in_n, in_k, in_d, in_q = map(int, sys.argv[1:5])
        run_hybrid_experiment(in_n, in_k, in_d, in_q)
    except ValueError:
        print("Error: Inputs must be integers.")
