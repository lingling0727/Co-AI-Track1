import os
import time
import random
import math
import ast
from collections import Counter
import csv
import datetime

# generate_dataset 모듈에서 데이터 로드 함수 임포트
try:
    from generate_dataset import load_dataset
except ImportError:
    # generate_dataset.py가 같은 경로에 없을 경우를 대비한 더미 함수
    def load_dataset(n, k, q):
        print(f"[Warning] generate_dataset module not found. Using dummy data for {n},{k},{q}.")
        return [(1,0,0), (0,1,0), (0,0,1)], [[1,0,0],[0,1,0],[0,0,1]]

# OR-Tools 임포트 (ILP 솔버)
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("[Warning] 'ortools' library not found. ILP solver will be skipped.")

class RecursiveCodeClassifier:
    """
    Sascha Kurz의 2024년 방법론(Lattice Point Enumeration + ILP Pruning)을 적용한 분류기
    """
    def __init__(self, n, k, q, weights):
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = set(map(int, weights.split(',')))
        self.min_weight = min(self.target_weights)
        self.max_weight = max(self.target_weights)
        
        # 데이터 로드 (Incidence Matrix 포함)
        self.points, self.incidence = load_dataset(n, k, q)
        
        self.seed = random.randint(0, 2**31 - 1)
        random.seed(self.seed)
        
        self.num_points = len(self.points)
        self.num_hyperplanes = len(self.incidence)
        
        # [Optimization] Incidence Matrix를 희소 리스트(Sparse List)로 변환
        # point_hyperplanes_map[p_idx] = [h_idx1, h_idx2, ...] (점 p가 포함된 초평면 인덱스들)
        self.point_hyperplanes_map = [
            [h for h, val in enumerate(row) if val == 1] for row in self.incidence
        ]

        # 통계 및 로그용
        self.pruned_count = 0
        self.lp_calls = 0
        self.solutions_found = []
        self.start_time = 0

    def solve(self):
        self.start_time = time.time()
        self.solutions_found = []
        self.pruned_count = 0
        self.lp_calls = 0
        
        print(f"[*] Starting Recursive Search for n={self.n}, k={self.k}, q={self.q}")
        print(f"    Target Weights: {self.target_weights}")
        
        # 초기 상태: 선택된 점 없음, 현재 길이 0, 시작 인덱스 0
        # current_counts: {point_idx: count}
        # current_k_values: 각 초평면별 현재 포함된 점의 개수 (k_h) - 증분 업데이트용
        initial_k_values = [0] * self.num_hyperplanes
        self._backtrack(start_idx=0, current_n=0, current_counts={}, current_k_values=initial_k_values)
        
        elapsed = time.time() - self.start_time
        status = "Found" if self.solutions_found else "Infeasible"
        return {
            "status": status,
            "time": elapsed,
            "solutions_count": len(self.solutions_found),
            "pruned_branches": self.pruned_count,
            "lp_calls": self.lp_calls
        }

    def _backtrack(self, start_idx, current_n, current_counts, current_k_values):
        """
        재귀적 탐색 (DFS)
        start_idx: Symmetry Breaking을 위해 인덱스 순서 강제
        """
        # 1. Base Case: 길이가 n에 도달
        if current_n == self.n:
            if self._verify_solution_fast(current_k_values):
                print(f"    [!] Solution Found! Counts: {current_counts}")
                self.solutions_found.append(current_counts.copy())
            return

        # 2. Pruning Check (Heuristic & LP)
        # 너무 자주 호출하면 느려지므로, 일정 깊이마다 혹은 남은 n이 적을 때 호출
        remaining_n = self.n - current_n
        
        # 간단한 Weight Bound Check (Partial Weight Enumerator)
        # 최적화: 증분 업데이트된 current_k_values 사용
        if not self._check_partial_weights_fast(current_k_values, remaining_n):
            self.pruned_count += 1
            return

        # LP Relaxation Pruning (비용이 비싸므로 조건부 실행)
        # 최적화: 너무 자주 호출하지 않도록 빈도 조절 (예: 3단계마다, 그리고 끝부분에서는 생략)
        if ORTOOLS_AVAILABLE and remaining_n > 5 and (current_n % 3 == 0):
            if not self._check_feasibility_lp(start_idx, current_n, current_counts, current_k_values):
                self.pruned_count += 1
                return

        # 3. Recursive Step
        # start_idx부터 끝까지 점들을 순회하며 추가
        # x_i >= 0. 우리는 중복 조합을 탐색하므로, 현재 점을 여러 번 선택 가능
        # 하지만 구현 편의상, 현재 점을 1개 이상 추가하는 경우와 아예 건너뛰는 경우로 분기하지 않고,
        # for loop로 "다음 추가할 점"을 선택하는 방식 사용 (조합 탐색 표준)
        
        for i in range(start_idx, self.num_points):
            # 점 i를 1개 추가
            new_counts = current_counts.copy()
            new_counts[i] = new_counts.get(i, 0) + 1

            # [Optimization] k_values 증분 업데이트 (In-place update & Revert)
            # 점 i가 포함된 초평면들의 k값만 1씩 증가
            for h_idx in self.point_hyperplanes_map[i]:
                current_k_values[h_idx] += 1
            
            self._backtrack(i, current_n + 1, new_counts, current_k_values)
            
            # Backtracking: 상태 복구
            for h_idx in self.point_hyperplanes_map[i]:
                current_k_values[h_idx] -= 1
            
            # 한 가지 해만 찾고 끝낼거면 여기서 break 가능
            if self.solutions_found: 
                return

    def _check_partial_weights_fast(self, current_k_values, remaining_n):
        """
        최적화된 Partial Weight Check
        current_k_values: 이미 계산된 각 초평면별 점의 개수 리스트
        """
        # Loop Unrolling이나 Numpy를 쓰면 더 빠르겠지만, Python 리스트 순회로 구현
        for k_h_curr in current_k_values:
            # 가능한 무게 범위
            min_possible_w = self.n - (k_h_curr + remaining_n)
            max_possible_w = self.n - k_h_curr
            
            # 범위 교차 검사: [min_p, max_p] 와 Target Weights W 가 교집합이 있어야 함
            # 즉, W의 어떤 원소 w가 min_p <= w <= max_p 를 만족해야 함
            can_satisfy = False
            for w in self.target_weights:
                if min_possible_w <= w <= max_possible_w:
                    can_satisfy = True
                    break
            
            if not can_satisfy:
                return False # Prune
        return True

    def _check_feasibility_lp(self, start_idx, current_n, current_counts, current_k_values):
        """
        LP Relaxation을 사용하여 현재 상태에서 해가 존재할 수 있는지 검사 (Pruning)
        """
        self.lp_calls += 1
        solver = pywraplp.Solver.CreateSolver('GLOP') # GLOP is Google's LP solver (faster than SCIP)
        if not solver: return True

        remaining_n = self.n - current_n
        
        # 변수: 남은 점들을 얼마나 더 선택할 것인가? (Continuous variables for Relaxation)
        # x_j >= 0 (실수)
        vars_map = {}
        # start_idx 부터 끝까지만 변수로 고려 (Symmetry Breaking 구조 따름)
        for j in range(start_idx, self.num_points):
            vars_map[j] = solver.NumVar(0, remaining_n, f'x_{j}')
            
        # 제약 1: 남은 개수 합 = remaining_n
        solver.Add(solver.Sum(vars_map.values()) == remaining_n)
        
        # 제약 2: 각 초평면에 대해 무게 조건 만족 가능성
        # w_h = n - (k_h_curr + sum(incidence[j][h] * x_j))
        # min_W <= w_h <= max_W (Relaxed constraint)
        # -> n - max_W <= k_h_total <= n - min_W
        
        min_k = self.n - self.max_weight
        max_k = self.n - self.min_weight
        
        for h_idx in range(self.num_hyperplanes):
            # 최적화: 미리 계산된 k값 사용
            k_h_curr = current_k_values[h_idx]
            
            # Future contribution expression
            future_k_expr = solver.Sum(vars_map[j] * self.incidence[j][h_idx] for j in range(start_idx, self.num_points))
            
            # Range constraint
            solver.Add(k_h_curr + future_k_expr >= min_k)
            solver.Add(k_h_curr + future_k_expr <= max_k)

        status = solver.Solve()
        return status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE

    def _verify_solution(self, counts):
        """
        완성된 해가 모든 제약조건을 만족하는지 최종 검증
        """
        for h_idx in range(self.num_hyperplanes):
            k_h = sum(cnt for p_idx, cnt in counts.items() if self.incidence[p_idx][h_idx])
            weight = self.n - k_h
            if weight not in self.target_weights:
                return False
        return True

def main():
    # 실험 파라미터 파일 경로
    param_file = "experiment_parameters.txt"
    
    if not os.path.exists(param_file):
        print(f"'{param_file}' not found. Please create it or run generate_dataset.py first.")
        return

    print("=== Code Classification Experiment: ILP vs Heuristic ===")
    results_csv = "experiment_results.csv"

    with open(results_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Timestamp", "n", "k", "q", "Weights",
            "Points", "Status", "Time", "Solutions_Count",
            "Pruned_Branches", "LP_Calls"
        ]
        # extrasaction='ignore' prevents errors when row dicts have extra keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        # 항상 헤더를 다시 작성
        writer.writeheader()

        with open(param_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # 쉼표가 무게 집합 {} 안에도 있으므로, 앞의 3개 쉼표까지만 분리
                parts = line.split(",", 3)
                n = int(parts[0].strip())
                k = int(parts[1].strip())
                q = int(parts[2].strip())
                weights = parts[3].strip().replace("{", "").replace("}", "")
                
                print(f"\n[Experiment] n={n}, k={k}, q={q}, Weights={{{weights}}}")
                
                try:
                    classifier = RecursiveCodeClassifier(n, k, q, weights)
                except Exception as e:
                    print(f"  [Error] {e}")
                    continue
                
                # Run Recursive Search
                print("  > Running Recursive Search with ILP Pruning...")
                result = classifier.solve()
                print(f"  [Result] Status: {result['status']}, Time: {result['time']:.4f}s, Found: {result['solutions_count']}")
                
                # 결과 CSV 저장
                writer.writerow({
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "n": n, "k": k, "q": q, "Weights": f"{{{weights}}}",
                    "Points": classifier.num_points,
                    "Status": result['status'],
                    "Time": round(result['time'], 4),
                    "Solutions_Count": result['solutions_count'],
                    "Pruned_Branches": result['pruned_branches'],
                    "LP_Calls": result['lp_calls']
                })
                csvfile.flush() # 데이터 유실 방지를 위해 즉시 기록

if __name__ == "__main__":
    main()