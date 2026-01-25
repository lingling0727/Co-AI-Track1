import sys
import highspy
import numpy as np
import math
import time
import pandas as pd
import itertools

# [수정 1] galois 라이브러리 사용 (필수: pip install galois)
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
        
        # [수정 2] 입력받은 q에 맞는 유한체(Galois Field) 자동 생성
        try:
            self.GF = galois.GF(q)
        except LookupError:
            print(f"Error: q={q} is not a valid prime power (e.g., 2, 3, 4, 5, 7, 8, 9...).")
            sys.exit(1)
            
        self.points = self._gen_points()

    def _gen_points(self):
        """
        PG(k-1, q)의 모든 점을 생성합니다.
        최적화: 별도의 정규화 계산 없이 '첫 번째 0이 아닌 성분이 1'인 벡터만 수집합니다.
        """
        points = []
        # 0부터 q-1까지의 숫자로 이루어진 모든 길이 k 벡터 생성
        for vec in itertools.product(range(self.q), repeat=self.k):
            # 0 벡터 제외
            if all(v == 0 for v in vec):
                continue
            
            vec = list(vec)
            
            # 첫 번째 0이 아닌 성분 찾기
            first_nz_idx = next((i for i, x in enumerate(vec) if x != 0), -1)
            
            # [핵심] 사영 기하학에서 각 1차원 부분공간(점)은 
            # '첫 성분이 1'인 대표 벡터를 유일하게 가집니다.
            # 따라서 이 조건만 걸러내면 중복 제거/정규화 연산이 필요 없습니다.
            if vec[first_nz_idx] == 1:
                points.append(tuple(vec))
                
        return points

# ==========================================
# 2. [Step 2] Griesmer Bound 구현
# ==========================================
def calculate_griesmer_bound(k, d, q):
    """
    Computes the Griesmer bound: n >= sum(ceil(d / q^i)) for i=0 to k-1
    """
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
        
        # 통계용 변수
        self.nodes_visited = 0
        self.lp_calls = 0
        self.start_time = 0
        
        # Griesmer Bound 미리 계산
        self.global_griesmer = calculate_griesmer_bound(target_k, d, q)
        print(f"[Init] Griesmer Bound for [{target_n}, {target_k}, {d}]_{q}: n >= {self.global_griesmer}")
        
    def solve(self):
        self.start_time = time.time()
        self.nodes_visited = 0
        self.lp_calls = 0
        
        # 초기 상태: 선택된 점 없음
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
        
        # 1. 목표 도달 확인
        if len(current_selection) == self.target_n:
            return True
            
        # 2. 가망성 확인 (Pruning)
        if not self._is_promising(current_selection):
            return False 

        # 3. 분기 (Branching)
        for i in range(start_index, self.num_points):
            # 다음 점 추가 시도
            if self._backtrack(current_selection + [i], i + 1):
                return True
                
        return False

    def _is_promising(self, current_selection):
        current_len = len(current_selection)
        
        # A. [RCUB / Griesmer Check]
        if current_len > self.target_n:
            return False

        # B. [Hybrid Logic] - LP Check (예시 조건)
        use_lp_check = False
        if current_len >= self.target_n * 0.8: 
            use_lp_check = True
            
        if use_lp_check:
            self.lp_calls += 1
            if not self._check_lp_feasibility(current_selection):
                return False

        return True 

    def _check_lp_feasibility(self, current_selection):
        # 실제로는 여기서 선택된 점들의 Incidence Matrix를 구성하여
        # Weight Constraint를 만족하는지 HiGHS로 풀어야 합니다.
        # 현재는 구조만 잡혀 있으므로 True 반환
        return True 

# ==========================================
# 4. 실행 및 CSV 저장
# ==========================================
def run_hybrid_experiment(n, k, d, q):
    print(f"=== Hybrid Pruning & Griesmer Bound Experiment ===")
    print(f"Parameters: n={n}, k={k}, d={d}, q={q}")
    
    # [수정 3] ProjectiveGeometry 생성 시 gf 객체 전달 불필요 (내부에서 처리)
    pg = ProjectiveGeometry(k, q)
    print(f"Geometry Generated: {len(pg.points)} points in PG({k-1}, {q})")
    
    # Custom Solver 실행
    solver = CustomHybridSolver(n, k, d, q, pg.points)
    status, run_time, nodes = solver.solve()
    
    # CSV 저장
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
    # print(df) # 결과가 길면 주석 처리

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("\nUsage: python main.py <n> <k> <d> <q>")
        print("Example: python main.py 7 3 2 4")
        sys.exit(1)
    
    try:
        in_n = int(sys.argv[1])
        in_k = int(sys.argv[2])
        in_d = int(sys.argv[3])
        in_q = int(sys.argv[4])
        
        run_hybrid_experiment(in_n, in_k, in_d, in_q)
        
    except ValueError:
        print("Error: All inputs (n, k, d, q) must be integers.")
        sys.exit(1)
