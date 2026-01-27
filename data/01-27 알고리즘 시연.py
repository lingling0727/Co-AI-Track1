import sys
import time
import numpy as np
import itertools
import csv
import os
import argparse
from datetime import datetime

# 필수 라이브러리 체크
try:
    import galois
except ImportError:
    print("Error: 'galois' library is not installed. Please run 'pip install galois'")
    sys.exit(1)

class IsomorphismChecker:
    def __init__(self, k, q, points_map):
        self.k = k
        self.q = q
        self.GF = galois.GF(q)
        self.points_map = self.GF(points_map)
        
        # GL(k, q) 행렬 생성
        print(f"[IsoCheck] Generating GL({k}, {q}) matrices...")
        self.gl_matrices = self._generate_gl_matrices()
        print(f"[IsoCheck] Generated {len(self.gl_matrices)} matrices.")

    def _generate_gl_matrices(self):
        """GL(k, q)의 모든 가역 행렬을 생성"""
        matrices = []
        for flat_mat in itertools.product(range(self.q), repeat=self.k*self.k):
            mat_np = np.array(flat_mat, dtype=int).reshape(self.k, self.k)
            mat_gf = self.GF(mat_np)
            if np.linalg.det(mat_gf) != 0:
                matrices.append(mat_gf)
        return matrices

    def get_weight_distribution(self, point_indices):
        """1차 필터링: Weight Distribution 계산"""
        G = self.points_map[point_indices].T
        messages = list(itertools.product(range(self.q), repeat=self.k))
        messages = self.GF(messages)
        codewords = messages @ G
        
        # [Fix] Galois Array를 일반 Numpy 배열로 변환 (Bool 오류 방지)
        codewords_np = np.array(codewords)
        weights = np.count_nonzero(codewords_np, axis=1)
        
        dist = {}
        for w in weights:
            dist[w] = dist.get(w, 0) + 1
        return tuple(sorted(dist.items()))

    def normalize_points(self, points_matrix):
        """사영 공간 좌표 정규화"""
        normalized = []
        for col in points_matrix.T:
            for val in col:
                if val != 0:
                    scaler = self.GF(1) / val
                    col = col * scaler
                    break
            normalized.append(tuple(col.tolist()))
        return set(normalized)

    def are_isomorphic(self, sol1_indices, sol2_indices):
        """GL(k,q) 전수 조사를 통한 동형성 판별"""
        pts1 = self.points_map[sol1_indices]
        pts2_set = self.normalize_points(self.points_map[sol2_indices].T)
        
        for A in self.gl_matrices:
            transformed_pts = (A @ pts1.T)
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
        
        print(f"[Phase 1] Searching for [{n}, {k}, {d}]_{q} (Projective Code)")

    def generate_sorted_geometry(self, k, q):
        GF = galois.GF(q)
        points = []
        for vec in itertools.product(range(q), repeat=k):
            if all(v == 0 for v in vec): continue
            vec = list(vec)
            first_nz = next((i for i, x in enumerate(vec) if x != 0), -1)
            if vec[first_nz] == 1:
                points.append(tuple(vec))
        points.sort()
        pts_matrix = GF(list(points)).T
        dot_products = pts_matrix.T @ pts_matrix
        incidence = (dot_products == 0).astype(int)
        return list(points), incidence

    def find_standard_basis(self):
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
        for next_idx in range(last_idx + 1, self.num_points):
            if self.check_constraints(current_solution, next_idx):
                current_solution.append(next_idx)
                self.backtrack(current_solution)
                current_solution.pop()

def save_solutions_to_csv(filename, solutions):
    if not solutions:
        return
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: Index + Points...
            header = ['Index'] + [f'P{i}' for i in range(len(solutions[0]))]
            writer.writerow(header)
            for idx, sol in enumerate(solutions):
                writer.writerow([idx+1] + sol)
        print(f"[Info] Solutions saved to '{filename}'")
    except Exception as e:
        print(f"[Error] Could not save solutions CSV: {e}")

def save_performance_log(filename, stats):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
        print(f"[Info] Performance stats appended to '{filename}'")
    except Exception as e:
        print(f"[Error] Could not save performance CSV: {e}")

if __name__ == "__main__":
    # 1. 터미널 인자 파싱 (argparse)
    parser = argparse.ArgumentParser(description="Linear Code Search & Isomorphism Check")
    parser.add_argument('-n', type=int, required=True, help='Code Length')
    parser.add_argument('-k', type=int, required=True, help='Dimension')
    parser.add_argument('-d', type=int, required=True, help='Minimum Distance')
    parser.add_argument('-q', type=int, required=True, help='Field Size')
    
    args = parser.parse_args()
    
    n, k, d, q = args.n, args.k, args.d, args.q
    
    print("="*60)
    print(f"Target Parameters: n={n}, k={k}, d={d}, q={q}")
    print("="*60)

    # 2. Phase 1: 해 탐색
    solver = OrderlyGeneratorAll(n, k, d, q)
    phase1_solutions, phase1_time = solver.solve()
    print(f"[Phase 1] Found {len(phase1_solutions)} candidate solutions in {phase1_time:.4f}s.")
    
    final_unique_solutions = []
    iso_time = 0
    
    if phase1_solutions:
        # 3. Phase 2: 동형성 검사
        print("-" * 60)
        iso_start = time.time()
        iso_checker = IsomorphismChecker(k, q, solver.points)
        
        # Weight Distribution으로 그룹핑
        grouped = {}
        for sol in phase1_solutions:
            wd = iso_checker.get_weight_distribution(sol)
            if wd not in grouped: grouped[wd] = []
            grouped[wd].append(sol)
            
        print(f"[IsoCheck] Grouped candidates into {len(grouped)} weight distributions.")
        
        # 그룹별 GL(k,q) 전수 검사
        for wd, candidates in grouped.items():
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
            
        iso_time = time.time() - iso_start
        print(f"[IsoCheck] Reduced to {len(final_unique_solutions)} unique solutions in {iso_time:.4f}s.")
    else:
        print("[Info] No candidates found in Phase 1. Skipping Phase 2.")

    # 4. 결과 저장 (CSV)
    # (1) 솔루션 저장
    sol_csv_name = f"solutions_n{n}_k{k}_d{d}_q{q}.csv"
    save_solutions_to_csv(sol_csv_name, final_unique_solutions)
    
    # (2) 성능 로그 저장
    perf_stats = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n': n, 'k': k, 'd': d, 'q': q,
        'Phase1_Candidates': len(phase1_solutions),
        'Unique_Solutions': len(final_unique_solutions),
        'Phase1_Time_sec': round(phase1_time, 4),
        'IsoCheck_Time_sec': round(iso_time, 4),
        'Total_Time_sec': round(phase1_time + iso_time, 4),
        'Nodes_Visited': solver.nodes_visited
    }
    save_performance_log("performance_stats.csv", perf_stats)
    
    print("="*60)
    print("Done.")
