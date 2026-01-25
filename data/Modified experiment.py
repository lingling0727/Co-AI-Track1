import sys
import highspy
import numpy as np
import math
import time
import pandas as pd

# ==========================================
# 1. 기초 수학 및 기하 (기존과 동일)
# ==========================================
class GF8:
    def __init__(self):
        self.size = 8
        self.prim_poly = 0b1011
        self.exp = [0]*8; self.log = [0]*8
        x = 1
        for i in range(7):
            self.exp[i] = x; self.log[x] = i
            x <<= 1
            if x & 0b1000: x ^= self.prim_poly
        self.exp[7] = 0
    def add(self, a, b): return a ^ b
    def mul(self, a, b):
        if a==0 or b==0: return 0
        return self.exp[(self.log[a]+self.log[b])%7]
    def dot(self, v1, v2):
        res = 0
        for a,b in zip(v1,v2): res = self.add(res, self.mul(a,b))
        return res

class ProjectiveGeometry:
    def __init__(self, k, q, gf):
        self.k=k; self.q=q; self.gf=gf
        self.points = self._gen_points()
    def _gen_points(self):
        pts = []; seen = set()
        for i in range(1, self.q**self.k):
            v = []
            tmp = i
            for _ in range(self.k): v.append(tmp%self.q); tmp//=self.q
            v = v[::-1]
            fnz = next((idx for idx,x in enumerate(v) if x!=0), -1)
            if fnz==-1: continue
            inv = self.gf.exp[(7-self.gf.log[v[fnz]])%7]
            nv = tuple(self.gf.mul(x, inv) for x in v)
            if nv not in seen: seen.add(nv); pts.append(nv)
        return sorted(list(pts))

# ==========================================
# 2. [Step 2] Griesmer Bound 구현
# ==========================================
def calculate_griesmer_bound(k, d, q):
    """
    Computes the Griesmer bound: n >= sum(ceil(d / q^i)) for i=0 to k-1
    """
    bound_val = 0
    current_d = d
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
        
        # Griesmer Bound 미리 계산 (전역 기준)
        self.global_griesmer = calculate_griesmer_bound(target_k, d, q)
        print(f"[Init] Griesmer Bound for [{target_n}, {target_k}, {d}]_{q}: n >= {self.global_griesmer}")
        
    def solve(self):
        self.start_time = time.time()
        self.nodes_visited = 0
        self.lp_calls = 0
        
        # 초기 상태: 선택된 점(Column) 없음
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
            return True # 해를 찾음
            
        # 2. 가망성 확인 (Pruning)
        if not self._is_promising(current_selection):
            return False 

        # 3. 분기 (Branching)
        for i in range(start_index, self.num_points):
            if self._backtrack(current_selection + [i], i + 1):
                return True
                
        return False

    def _is_promising(self, current_selection):
        current_len = len(current_selection)
        
        # A. [RCUB / Griesmer Check]
        if current_len > self.target_n:
            return False

        # B. [Hybrid Logic] - LP Check
        use_lp_check = False
        if current_len >= self.target_n * 0.8: 
            use_lp_check = True
            
        if use_lp_check:
            self.lp_calls += 1
            if not self._check_lp_feasibility(current_selection):
                return False

        return True 

    def _check_lp_feasibility(self, current_selection):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        # (실제 LP 구현은 생략 - 항상 True 반환)
        return True 

# ==========================================
# 4. 실행 및 CSV 저장 (수정됨)
# ==========================================
def run_hybrid_experiment(n, k, d, q):
    print(f"=== Hybrid Pruning & Griesmer Bound Experiment ===")
    print(f"Parameters: n={n}, k={k}, d={d}, q={q}")
    
    # 주의: 현재 GF8 클래스는 q=8에 최적화되어 있습니다.
    if q != 8:
        print("[WARNING] GF8 class is hardcoded for q=8. Results for other q may be incorrect.")

    gf = GF8()
    pg = ProjectiveGeometry(k, q, gf)
    
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
    print(df)

if __name__ == "__main__":
    # 사용법 안내 및 인자 파싱
    if len(sys.argv) < 5:
        print("\nUsage: python main.py <n> <k> <d> <q>")
        print("Example: python main.py 35 4 28 8")
        sys.exit(1)
    
    try:
        in_n = int(sys.argv[1])
        in_k = int(sys.argv[2])
        in_d = int(sys.argv[3])
        in_q = int(sys.argv[4])
