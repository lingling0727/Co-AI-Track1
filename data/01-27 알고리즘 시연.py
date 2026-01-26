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

class OrderlyGeneratorAll:
    def __init__(self, n, k, d, q):
        self.n = n
        self.k = k
        self.d = d
        self.q = q
        
        # 1. 기하 구조 생성 및 '사전식 정렬' (Lexicographical Sort)
        self.points, self.incidence = self.generate_sorted_geometry(k, q)
        self.num_points = len(self.points)
        
        # 탐색 상태 변수
        self.found_solutions = [] # 모든 해를 저장할 리스트
        self.nodes_visited = 0
        
        # 허용되는 최대 직선 가중치 (n - d)
        self.max_weight = n - d
        
        print(f"[Init] Orderly Generation (Find ALL) for [{n}, {k}, {d}]_{q}")
        print(f"       Geometry: PG({k-1}, {q}) with {self.num_points} points (Sorted).")
        print(f"       Constraint: Max points per hyperplane <= {self.max_weight}")

    def generate_sorted_geometry(self, k, q):
        """PG(k-1, q)의 점들을 생성하고 '사전식 순서'로 정렬하여 반환"""
        GF = galois.GF(q)
        points = []
        
        for vec in itertools.product(range(q), repeat=k):
            if all(v == 0 for v in vec): continue
            
            vec = list(vec)
            # Projective Point 정규화
            first_nz = next((i for i, x in enumerate(vec) if x != 0), -1)
            if vec[first_nz] == 1:
                points.append(tuple(vec))
        
        points.sort() # 사전식 정렬
        
        pts_matrix = GF(list(points)).T
        dot_products = pts_matrix.T @ pts_matrix
        incidence = (dot_products == 0).astype(int)
        
        return list(points), incidence

    def find_standard_basis(self):
        """기저 고정(Basis Fixing)을 위한 표준 기저 인덱스 찾기"""
        basis_vecs = []
        identity = np.eye(self.k, dtype=int)
        
        indices = []
        for row in identity:
            target = tuple(row)
            try:
                idx = self.points.index(target)
                indices.append(idx)
            except ValueError:
                pass
        indices.sort()
        return indices

    def check_constraints(self, current_solution, new_point_idx):
        """새로운 점을 추가했을 때, 모든 초평면의 점 개수가 n-d 이하인지 검사"""
        temp_indices = current_solution + [new_point_idx]
        sub_matrix = self.incidence[:, temp_indices]
        weights = np.sum(sub_matrix, axis=1)
        
        if np.max(weights) > self.max_weight:
            return False
        return True

    def solve(self):
        start_time = time.time()
        
        # Row Symmetry 제거: 초기 k개의 점을 표준 기저로 고정
        basis_indices = self.find_standard_basis()
        
        if len(basis_indices) < self.k:
            print("[Error] Standard basis not found in geometry.")
            return [], 0
            
        print(f"Applying Canonical Basis Fixing: {basis_indices}")
        print("Starting Full Exhaustive Search...")
        
        # 탐색 시작
        self.backtrack(basis_indices)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return self.found_solutions, duration

    def backtrack(self, current_solution):
        # [변경점] "해를 찾으면 멈춤(return)" 코드를 삭제했습니다.
        # 이제 끝까지 탐색합니다.

        self.nodes_visited += 1
        
        # 1. 목표 도달 (n개의 점 선택 완료)
        if len(current_solution) == self.n:
            # 찾은 해를 복사해서 저장 (Deep Copy)
            self.found_solutions.append(list(current_solution))
            
            # (옵션) 해를 찾을 때마다 로그를 찍고 싶으면 아래 주석 해제
            # print(f"Found Solution #{len(self.found_solutions)}: {current_solution}")
            return

        # 2. 다음 후보군 생성 (Lexicographical Extension)
        last_idx = current_solution[-1]
        remaining = self.n - len(current_solution)
        
        start_idx = last_idx + 1
        end_idx = self.num_points - remaining + 1
        
        for next_idx in range(start_idx, end_idx):
            # 3. 제약 조건 검사 (가지치기)
            if self.check_constraints(current_solution, next_idx):
                current_solution.append(next_idx)
                self.backtrack(current_solution)
                current_solution.pop() # Backtrack

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python orderly_gen_all.py n k d q")
        sys.exit(1)
        
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    d = int(sys.argv[3])
    q = int(sys.argv[4])
    
    solver = OrderlyGeneratorAll(n, k, d, q)
    solutions, time_sec = solver.solve()
    
    print("="*50)
    print(f"[Done] Search Completed.")
    print(f"       Time: {time_sec:.4f}s")
    print(f"       Total Nodes Visited: {solver.nodes_visited}")
    print(f"       Total Canonical Solutions Found: {len(solutions)}")
    print("="*50)
    
    if solutions:
        print("First 5 Solutions:")
        for i, sol in enumerate(solutions[:5]):
            print(f"  {i+1}: {sol}")
        if len(solutions) > 5:
            print(f"  ... and {len(solutions)-5} more.")
            
        # CSV 파일로 모든 해 저장 (선택 사항)
        csv_filename = f"solutions_n{n}_k{k}_d{d}_q{q}.csv"
        try:
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Index'] + [f'P{i}' for i in range(n)])
                for idx, sol in enumerate(solutions):
                    writer.writerow([idx+1] + sol)
            print(f"\n[Info] All solutions saved to '{csv_filename}'")
        except Exception as e:
            print(f"[Error] Could not save CSV: {e}")
            
    else:
        print("       Status: Infeasible (No Solution Found)")
