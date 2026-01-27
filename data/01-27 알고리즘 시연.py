import sys
import time
import numpy as np
import itertools
import csv
import copy

# 필수 라이브러리 체크
try:
    import galois
except ImportError:
    print("Error: 'galois' library is not installed. Please run 'pip install galois'")
    sys.exit(1)

class IsomorphismChecker:
    def __init__(self, k, q, points_map):
        """
        k: 차원
        q: 필드 크기
        points_map: 정렬된 전체 사영 공간 점들의 리스트 (index -> coordinate vector)
        """
        self.k = k
        self.q = q
        self.GF = galois.GF(q)
        self.points_map = self.GF(points_map)  # 전체 점 좌표 (N x k)
        
        # GL(k, q) 행렬 생성 (k가 작을 때만 유효, k=3, q=3이면 약 1.1만개로 가능)
        print(f"[IsoCheck] Generating GL({k}, {q}) matrices...")
        self.gl_matrices = self._generate_gl_matrices()
        print(f"[IsoCheck] Generated {len(self.gl_matrices)} matrices.")

    def _generate_gl_matrices(self):
        """GL(k, q)의 모든 가역 행렬을 생성"""
        matrices = []
        # 가능한 모든 k*k 행렬 생성
        for flat_mat in itertools.product(range(self.q), repeat=self.k*self.k):
            mat_np = np.array(flat_mat, dtype=int).reshape(self.k, self.k)
            mat_gf = self.GF(mat_np)
            if np.linalg.det(mat_gf) != 0:
                matrices.append(mat_gf)
        return matrices

    def get_weight_distribution(self, point_indices):
        """
        해당 점 집합으로 생성되는 선형 코드의 Weight Distribution 계산
        (1차 필터링용 불변량)
        """
        # Generator Matrix G 생성 (k x n)
        G = self.points_map[point_indices].T
        
        # 모든 메시지 벡터 생성 (q^k 개)
        messages = list(itertools.product(range(self.q), repeat=self.k))
        messages = self.GF(messages)
        
        # Codewords 생성: C = m * G
        codewords = messages @ G
        
        # [수정됨] Galois Array를 일반 Numpy 배열로 변환
        # galois 라이브러리 특성상 직접 count_nonzero를 하면 bool 변환 에러가 발생할 수 있음
        codewords_np = np.array(codewords)
        
        # Hamming Weight 계산 (0이 아닌 성분의 개수)
        weights = np.count_nonzero(codewords_np, axis=1)
        
        # 분포 카운팅 (예: {0:1, 3:10, 4:16 ...})
        dist = {}
        for w in weights:
            dist[w] = dist.get(w, 0) + 1
            
        # 딕셔너리를 정렬된 튜플로 변환 (hashable)
        return tuple(sorted(dist.items()))

    def normalize_points(self, points_matrix):
        """
        점들의 집합을 표준형으로 정규화 (사영 공간 좌표 통일)
        각 열(벡터)에 대해 첫 번째 0이 아닌 성분을 1로 만듦.
        """
        normalized = []
        for col in points_matrix.T:
            # 갈로아 필드 연산
            for val in col:
                if val != 0:
                    scaler = self.GF(1) / val
                    col = col * scaler
                    break
            normalized.append(tuple(col.tolist()))
        return set(normalized) # 순서 무시를 위해 set으로 반환

    def are_isomorphic(self, sol1_indices, sol2_indices):
        """
        두 해(점 인덱스 리스트)가 동형인지 GL(k,q) 전수 조사로 확인
        """
        pts1 = self.points_map[sol1_indices] # shape (n, k)
        pts2_set = self.normalize_points(self.points_map[sol2_indices].T)
        
        # 전수 조사: A * pts1 == pts2가 되는 A가 존재하는지
        for A in self.gl_matrices:
            # 변환: transformed = (pts1 * A.T).T  -> (k x n)
            # pts1은 (n, k)이므로 pts1 @ A.T 하면 (n, k)가 됨.
            # 사영 변환은 점 벡터 v에 대해 A*v^T
            
            transformed_pts = (A @ pts1.T) # (k, n)
            
            # 정규화 후 집합 비교
            if self.normalize_points(transformed_pts) == pts2_set:
                return True
        return False

class OrderlyGeneratorAll:
    def __init__(self, n, k, d, q):
        self.n = n
        self.k = k
        self.d = d
        self.q = q
        
        self.points, self.incidence = self.generate_sorted_geometry(k, q)
        self.num_points = len(self.points)
        self.found_solutions = [] 
        self.nodes_visited = 0
        self.max_weight = n - d
        
        print(f"[Init] Searching for [{n}, {k}, {d}]_{q} (Projective)")

    def generate_sorted_geometry(self, k, q):
        GF = galois.GF(q)
        points = []
        for vec in itertools.product(range(q), repeat=k):
            if all(v == 0 for v in vec): continue
            vec = list(vec)
            # Normalize
            first_nz = next((i for i, x in enumerate(vec) if x != 0), -1)
            if vec[first_nz] == 1:
                points.append(tuple(vec))
        points.sort()
        pts_matrix = GF(list(points)).T
        dot_products = pts_matrix.T @ pts_matrix
        incidence = (dot_products == 0).astype(int)
        return list(points), incidence

    def find_standard_basis(self):
        basis_vecs = []
        identity = np.eye(self.k, dtype=int)
        indices = []
        for row in identity:
            try:
                indices.append(self.points.index(tuple(row)))
            except ValueError: pass
        return indices

    def check_constraints(self, current_solution, new_point_idx):
        temp_indices = current_solution + [new_point_idx]
        sub_matrix = self.incidence[:, temp_indices]
        if np.max(np.sum(sub_matrix, axis=1)) > self.max_weight:
            return False
        return True

    def solve(self):
        start_time = time.time()
        basis_indices = self.find_standard_basis()
        self.backtrack(basis_indices)
        duration = time.time() - start_time
        return self.found_solutions, duration

    def backtrack(self, current_solution):
        if len(current_solution) == self.n:
            self.found_solutions.append(list(current_solution))
            return

        last_idx = current_solution[-1]
        # 사영 코드 조건: 중복 불가 (start = last + 1)
        # 만약 Multiset을 원하면 start = last 로 변경
        for next_idx in range(last_idx + 1, self.num_points):
            if self.check_constraints(current_solution, next_idx):
                current_solution.append(next_idx)
                self.backtrack(current_solution)
                current_solution.pop()

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    # 파라미터 하드코딩 (테스트용) 또는 argv
    n, k, d, q = 6, 3, 3, 3
    if len(sys.argv) >= 5:
        n, k, d, q = map(int, sys.argv[1:5])

    print(f"Target: n={n}, k={k}, d={d}, q={q}")
    
    # 1. 해 탐색
    solver = OrderlyGeneratorAll(n, k, d, q)
    solutions, t_sec = solver.solve()
    
    print(f"Phase 1 Done. Found {len(solutions)} candidate solutions.")
    
    if not solutions:
        sys.exit(0)

    # 2. 동형성 검사 (Isomorphism Check)
    print("="*50)
    print("Starting Strict Isomorphism Check...")
    
    iso_checker = IsomorphismChecker(k, q, solver.points)
    
    # 그룹핑: Weight Distribution이 같은 것끼리 묶음 (속도 최적화)
    grouped_solutions = {}
    for sol in solutions:
        wd = iso_checker.get_weight_distribution(sol)
        if wd not in grouped_solutions:
            grouped_solutions[wd] = []
        grouped_solutions[wd].append(sol)
        
    print(f" grouped into {len(grouped_solutions)} distinct weight distributions.")
    
    final_unique_solutions = []
    
    for wd, candidates in grouped_solutions.items():
        # 각 그룹 내에서 GL(k,q) 검사 수행
        unique_in_group = []
        for cand in candidates:
            is_new = True
            for existing in unique_in_group:
                if iso_checker.are_isomorphic(cand, existing):
                    is_new = False
                    break
            if is_new:
                unique_in_group.append(cand)
        
        final_unique_solutions.extend(unique_in_group)
        print(f" -> Group {wd}: reduced {len(candidates)} to {len(unique_in_group)}")

    print("="*50)
    print(f"Final Result: {len(final_unique_solutions)} Non-isomorphic Solutions.")
    for idx, sol in enumerate(final_unique_solutions):
        print(f"Unique Sol #{idx+1}: {sol}")
