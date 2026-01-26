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

class OrderlyGenerator:
    def __init__(self, n, k, d, q):
        self.n = n
        self.k = k
        self.d = d
        self.q = q
        
        # 1. 기하 구조 생성 및 '사전식 정렬' (Lexicographical Sort)
        # 점들을 미리 정렬해두어야 '순서'라는 개념이 성립합니다.
        self.points, self.incidence = self.generate_sorted_geometry(k, q)
        self.num_points = len(self.points)
        
        # 탐색 상태 변수
        self.found_solutions = []
        self.nodes_visited = 0
        self.canonical_pruned = 0 # 동형성 검사로 잘려나간 가지 수
        
        # 허용되는 최대 직선 가중치 (n - d)
        self.max_weight = n - d
        
        print(f"[Init] Orderly Generation for [{n}, {k}, {d}]_{q}")
        print(f"       Geometry: PG({k-1}, {q}) with {self.num_points} points (Sorted).")
        print(f"       Constraint: Max points per hyperplane <= {self.max_weight}")

    def generate_sorted_geometry(self, k, q):
        """
        PG(k-1, q)의 점들을 생성하고 '사전식 순서'로 정렬하여 반환합니다.
        Orderly Generation의 핵심은 입력 데이터의 정렬입니다.
        """
        GF = galois.GF(q)
        points = []
        
        # itertools.product는 이미 사전식 순서대로 생성함
        for vec in itertools.product(range(q), repeat=k):
            if all(v == 0 for v in vec): continue
            
            vec = list(vec)
            # Projective Point 정규화 (첫 0이 아닌 성분을 1로)
            first_nz = next((i for i, x in enumerate(vec) if x != 0), -1)
            if vec[first_nz] == 1:
                points.append(tuple(vec)) # 튜플로 변환 (비교 가능하게)
        
        # 명시적 정렬 (Python 튜플 비교 = Lexicographical)
        points.sort()
        
        # 인시던스 행렬 생성
        pts_matrix = GF(list(points)).T
        dot_products = pts_matrix.T @ pts_matrix
        incidence = (dot_products == 0).astype(int)
        
        return list(points), incidence

    def find_standard_basis(self):
        """
        기저 고정(Basis Fixing)을 위해 표준 기저 벡터들의 인덱스를 찾습니다.
        예: (0,0,1), (0,1,0), (1,0,0) ... (사전식 역순 주의)
        """
        basis_vecs = []
        identity = np.eye(self.k, dtype=int)
        
        indices = []
        for row in identity:
            # 튜플로 변환하여 찾기
            target = tuple(row)
            try:
                idx = self.points.index(target)
                indices.append(idx)
            except ValueError:
                pass
                
        # 인덱스 순서대로 정렬 (탐색 순서 유지를 위해)
        indices.sort()
        return indices

    def is_canonical_partial(self, current_solution):
        """
        [고급 Orderly Generation 조건]
        현재까지 만들어진 부분 집합이 '표준 형태(Canonical)'인지 검사합니다.
        여기서는 가장 강력하고 계산 비용이 적은 'Basis Fixing'을 기본으로 사용합니다.
        (즉, 첫 k개의 점이 기저 벡터가 아니면 Canonical이 아님 -> 가지치기)
        """
        # Basis Fixing 전략:
        # 길이가 k 미만일 때는, 반드시 표준 기저 벡터 중에서만 골라야 함.
        # 이 함수는 backtrack 내부의 로직으로 대체되어 여기선 True 리턴.
        return True

    def check_constraints(self, current_solution, new_point_idx):
        """
        [디오판토스 제약 + 기하 제약]
        새로운 점을 추가했을 때, 모든 초평면의 점 개수가 n-d 이하인지 검사.
        Orderly Generation에서는 이 검사를 '생성 즉시' 수행합니다.
        """
        # 현재 솔루션 + 새로운 점
        temp_indices = current_solution + [new_point_idx]
        
        # 인시던스 행렬 슬라이싱 (관련된 컬럼만)
        sub_matrix = self.incidence[:, temp_indices]
        
        # 행별 합계 (각 초평면 위의 점 개수)
        weights = np.sum(sub_matrix, axis=1)
        
        if np.max(weights) > self.max_weight:
            return False # 제약 위반 (Pruning)
        return True

    def solve(self):
        start_time = time.time()
        
        # 1. Row Symmetry 제거: 초기 k개의 점을 표준 기저로 강제 할당
        # 이것만으로도 대칭성의 약 99%가 제거됩니다.
        basis_indices = self.find_standard_basis()
        
        if len(basis_indices) < self.k:
            print("[Error] Standard basis not found in geometry.")
            return None
            
        print(f"Applying Canonical Basis Fixing: {basis_indices}")
        print("Starting Orderly Search...")
        
        # k개의 점을 이미 선택한 상태에서 탐색 시작
        self.backtrack(basis_indices)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return self.found_solutions, duration

    def backtrack(self, current_solution):
        # 해를 하나 찾으면 멈출 것인가? (여기선 하나 찾고 종료하도록 설정)
        if self.found_solutions:
            return

        self.nodes_visited += 1
        
        # 1. 목표 도달 (n개의 점 선택 완료)
        if len(current_solution) == self.n:
            self.found_solutions.append(current_solution)
            print(f"\n[Success] Found Canonical Solution!")
            print(f"Indices: {current_solution}")
            return

        # 2. 다음 후보군 생성 (Lexicographical Extension)
        # Orderly Generation의 핵심:
        # "가장 마지막에 넣은 점의 인덱스보다 큰 인덱스만 고려한다."
        # 이는 집합 {A, B}와 {B, A} 중 {A, B} (A < B) 만 생성하게 함.
        last_idx = current_solution[-1]
        
        # 남은 필요한 점의 개수
        remaining = self.n - len(current_solution)
        
        # 탐색 범위 설정
        start_idx = last_idx + 1
        end_idx = self.num_points - remaining + 1
        
        for next_idx in range(start_idx, end_idx):
            # 3. Code Property 제약 검사 (Incremental)
            # 논문의 Phase 1(가중치 열거)을 여기서 실시간으로 수행
            if self.check_constraints(current_solution, next_idx):
                
                # 4. 재귀 호출 (Deepen)
                current_solution.append(next_idx)
                self.backtrack(current_solution)
                if self.found_solutions: return
                current_solution.pop() # Backtrack

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python orderly_gen.py n k d q")
        print("Example: python orderly_gen.py 7 3 2 4")
        sys.exit(1)
        
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    d = int(sys.argv[3])
    q = int(sys.argv[4])
    
    solver = OrderlyGenerator(n, k, d, q)
    solutions, time_sec = solver.solve()
    
    print(f"\n[Done] Time: {time_sec:.4f}s")
    print(f"       Nodes Visited: {solver.nodes_visited}")
    if solutions:
        print("       Status: Optimal (Found)")
    else:
        print("       Status: Infeasible (Not Found)")
