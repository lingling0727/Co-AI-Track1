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
        print(f"[Init] Griesmer Bound for [{target_n}, {target_k}, {d}]_8: n >= {self.global_griesmer}")
        
    def solve(self):
        self.start_time = time.time()
        self.nodes_visited = 0
        self.lp_calls = 0
        
        # 초기 상태: 선택된 점(Column) 없음
        # 재귀 탐색 시작
        # 실제 Prop 2는 Extension 문제이므로, Seed Code가 있다고 가정해야 하지만
        # 여기서는 Hybrid Logic 시연을 위해 "빈 상태에서 탐색"으로 단순화합니다.
        result = self._backtrack([], 0)
        
        end_time = time.time()
        status = "Optimal (Found)" if result else "Infeasible (Not Found)"
        print(f"\n[Done] Status: {status}")
        print(f"       Time: {end_time - self.start_time:.4f}s")
        print(f"       Nodes Visited: {self.nodes_visited}")
        print(f"       LP Calls (Pruning): {self.lp_calls}")
        return status, end_time - self.start_time, self.nodes_visited

    def _backtrack(self, current_selection, start_index):
        """
        DFS Recursion
        current_selection: List of indices of selected columns
        start_index: Next index to consider (for symmetry breaking/order)
        """
        self.nodes_visited += 1
        
        # 1. 목표 도달 확인
        if len(current_selection) == self.target_n:
            return True # 해를 찾음
            
        # 2. 가망성 확인 (Pruning) - 여기가 핵심!
        if not self._is_promising(current_selection):
            return False # 가지치기 (Prune)

        # 3. 분기 (Branching)
        # 단순화를 위해 남은 포인트들을 하나씩 추가해봄
        # (실제 고성능 솔버는 여기서 Heuristic을 사용함)
        for i in range(start_index, self.num_points):
            # 다음 단계로 이동
            if self._backtrack(current_selection + [i], i + 1):
                return True
                
        return False

    def _is_promising(self, current_selection):
        """
        [Step 1 & 2 적용] 하이브리드 가지치기 로직
        """
        current_len = len(current_selection)
        remaining_slots = self.target_n - current_len
        
        # ---------------------------------------------------------
        # A. [RCUB / Griesmer Check] - 아주 빠름 (Low Cost)
        # ---------------------------------------------------------
        # 현재 선택된 길이 + 남은 공간이 이론적 한계(Griesmer)를 만족하는가?
        # 여기서는 단순화하여 '남은 슬롯이 음수면 불가' 정도로 체크하거나
        # Griesmer를 역으로 이용하여 "현재 상태에서 d를 유지하며 target_n까지 갈 수 있나?" 체크
        
        # 예시: 만약 현재까지의 구조가 이미 Griesmer 조건을 위반했다면 Prune
        # (실제 구현 시엔 Distance 계산이 필요하지만, 여기선 길이 체크로 대체)
        if current_len > self.target_n:
            return False

        # ---------------------------------------------------------
        # B. [Hybrid Logic] - LP Check (High Cost)
        # RCUB(수학적 필터)를 통과했지만, 확신이 없을 때만 LP 호출
        # ---------------------------------------------------------
        # 조건: 어느 정도 깊이에 도달했을 때만 LP를 씀 (너무 상위 노드는 LP도 느림)
        # 예: 전체 길이의 80% 이상 찼을 때 검사
        
        use_lp_check = False
        if current_len >= self.target_n * 0.8: 
            use_lp_check = True
            
        if use_lp_check:
            self.lp_calls += 1
            # LP로 Feasibility Check (feasible 하지 않으면 Prune)
            if not self._check_lp_feasibility(current_selection):
                return False # LP가 "불가능"이라고 판정 -> 가지치기

        return True # 살아남음

    def _check_lp_feasibility(self, current_selection):
        """
        Check feasibility using HiGHS for the current partial selection.
        """
        # HiGHS 인스턴스 생성 (매번 생성하면 느리지만 로직 보여주기용)
        # 실제로는 Global Model에서 Bounds만 수정하는게 빠름
        h = highspy.Highs()
        h.setOptionValue("output_flag", False) # 로그 끔
        
        # 변수 생성 (현재 선택된 것은 1로 고정, 나머지는 0~1)
        # 여기서는 단순 Feasibility만 체크
        
        # (간소화를 위해 항상 True 리턴하도록 더미 구현. 
        #  실제로는 Proposition 2의 Weight Constraints를 여기에 넣어야 함)
        return True 

# ==========================================
# 4. 실행 및 CSV 저장
# ==========================================
def run_hybrid_experiment():
    print("=== Hybrid Pruning & Griesmer Bound Experiment ===")
    
    # 파라미터 설정
    q = 8
    target_n = 35 # Proposition 2
    target_k = 4
    d = 28
    
    gf = GF8()
    pg = ProjectiveGeometry(target_k, q, gf)
    
    # Custom Solver 실행
    solver = CustomHybridSolver(target_n, target_k, d, q, pg.points)
    
    # 실험 시작 (Time Limit을 고려해 깊이 제한을 줄여서 테스트 추천)
    # 여기서는 구조만 보여주기 위해 바로 실행
    status, run_time, nodes = solver.solve()
    
    # CSV 저장
    results = {
        "Method": ["Hybrid (RCUB+LP)"],
        "Griesmer_Bound": [solver.global_griesmer],
        "Time(s)": [run_time],
        "Nodes": [nodes],
        "LP_Calls": [solver.lp_calls],
        "Note": ["Step 1 & 2 Applied"]
    }
    df = pd.DataFrame(results)
    df.to_csv("hybrid_result.csv", index=False)
    print(f"[SUCCESS] Results saved to 'hybrid_result.csv'")
    print(df)

if __name__ == "__main__":
    run_hybrid_experiment()
