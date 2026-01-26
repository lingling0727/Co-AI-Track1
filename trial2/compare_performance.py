"""
compare_performance.py
논문의 방법론을 따르는 Baseline(Backtracking + LP Pruning)과 
우리가 개선한 Proposed Method(Method 1+2)의 성능을 비교합니다.
"""
import time
import csv
import sys

# 필요한 모듈 임포트
try:
    from geometry import generate_projective_points, get_incidence_matrix
    from ilp_model import CodeExtender  # Proposed Method
except ImportError:
    print("Error: 'geometry.py' or 'ilp_model.py' not found.")
    sys.exit(1)

# OR-Tools 확인 (Baseline용)
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: 'ortools' not found. Baseline method will run without LP pruning (Pure Backtracking).")

# Gurobi 확인
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# --- 1. Baseline Solver (Paper's Approach: Backtracking + LP Pruning) ---
class BaselineSolver:
    def __init__(self, n, k, q, target_weights, points, incidence_matrix, use_lp=True, use_bounds=True):
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = sorted(list(target_weights))
        self.points = points
        self.incidence = incidence_matrix # Hyperplanes x Points
        self.num_points = len(points)
        self.num_hyperplanes = len(incidence_matrix)
        
        self.use_lp = use_lp
        self.use_bounds = use_bounds
        
        self.min_weight = min(self.target_weights)
        self.max_weight = max(self.target_weights)
        
        # Precompute point -> hyperplanes map
        self.point_hyperplanes_map = [[] for _ in range(self.num_points)]
        for h in range(self.num_hyperplanes):
            for p in range(self.num_points):
                if self.incidence[h][p] == 1:
                    self.point_hyperplanes_map[p].append(h)
        
        self.solutions = []
        self.nodes_visited = 0
        self.lp_calls = 0
        self.pruned_nodes = 0

    def solve(self):
        self.solutions = []
        self.nodes_visited = 0
        self.lp_calls = 0
        self.pruned_nodes = 0
        
        current_counts = {}
        current_k_values = [0] * self.num_hyperplanes
        
        self._backtrack(0, 0, current_counts, current_k_values)
        return self.solutions, self.nodes_visited, self.pruned_nodes, self.lp_calls

    def _backtrack(self, start_idx, current_n, current_counts, current_k_values):
        self.nodes_visited += 1
        
        # Base Case
        if current_n == self.n:
            if self._verify(current_k_values):
                self.solutions.append(current_counts.copy())
            return

        remaining_n = self.n - current_n
        
        # 1. Simple Bound Check
        if self.use_bounds and not self._check_bounds(current_k_values, remaining_n):
            self.pruned_nodes += 1
            return

        # 2. LP Pruning (The "Paper" feature)
        # 논문처럼 LP 솔버를 사용하여 가지치기 (비용 문제로 깊이가 얕을 때만 수행하도록 설정 가능)
        if self.use_lp and (GUROBI_AVAILABLE or ORTOOLS_AVAILABLE) and remaining_n > 2:
            # 너무 느려지지 않게 가끔씩만 호출 (실제 논문 구현체는 더 최적화되었을 수 있음)
            if current_n % 2 == 0: 
                if not self._check_lp(remaining_n, current_k_values, start_idx):
                    self.pruned_nodes += 1
                    return

        # Branching
        for i in range(start_idx, self.num_points):
            # Update
            for h in self.point_hyperplanes_map[i]:
                current_k_values[h] += 1
            current_counts[i] = current_counts.get(i, 0) + 1
            
            self._backtrack(i, current_n + 1, current_counts, current_k_values)
            
            # Restore
            current_counts[i] -= 1
            if current_counts[i] == 0:
                del current_counts[i]
            for h in self.point_hyperplanes_map[i]:
                current_k_values[h] -= 1
            
            if self.solutions: return # Compare finding first solution only

    def _check_bounds(self, current_k_values, remaining_n):
        min_allowed_k = self.n - self.max_weight
        max_allowed_k = self.n - self.min_weight
        for k in current_k_values:
            if k > max_allowed_k: return False
            if k + remaining_n < min_allowed_k: return False
        return True

    def _check_lp(self, remaining_n, current_k_values, start_idx):
        """LP Pruning을 위한 Dispatcher. Gurobi를 우선적으로 사용합니다."""
        if GUROBI_AVAILABLE:
            return self._check_lp_gurobi(remaining_n, current_k_values, start_idx)
        elif ORTOOLS_AVAILABLE:
            return self._check_lp_ortools(remaining_n, current_k_values, start_idx)
        return True # 사용 가능한 솔버가 없으면 Pruning하지 않음

    def _check_lp_gurobi(self, remaining_n, current_k_values, start_idx):
        """Gurobi를 사용하여 LP Relaxation 가능성을 검사합니다."""
        self.lp_calls += 1
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("lp_prune", env=env) as model:
                    # 변수는 LP 완화를 위해 연속형(Continuous)으로 선언
                    x = model.addVars(range(start_idx, self.num_points), vtype=GRB.CONTINUOUS, lb=0, ub=remaining_n, name="x")
                    
                    # 제약 1: 남은 점들의 합
                    model.addConstr(gp.quicksum(x[j] for j in range(start_idx, self.num_points)) == remaining_n)
                    
                    # 제약 2: 각 초평면에 대한 무게 제약
                    min_k = self.n - self.max_weight
                    max_k = self.n - self.min_weight
                    
                    for h in range(self.num_hyperplanes):
                        k_curr = current_k_values[h]
                        future_k_expr = gp.quicksum(x[j] for j in range(start_idx, self.num_points) if self.incidence[h][j] == 1)
                        
                        model.addConstr(k_curr + future_k_expr >= min_k)
                        model.addConstr(k_curr + future_k_expr <= max_k)
                        
                    model.optimize()
                    return model.Status != GRB.INFEASIBLE
        except Exception:
            # Gurobi에서 오류 발생 시 안전하게 Pruning하지 않음
            return True

    def _check_lp_ortools(self, remaining_n, current_k_values, start_idx):
        """OR-Tools를 사용하여 LP Relaxation 가능성을 검사합니다."""
        self.lp_calls += 1
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver: return True
        
        vars_map = {}
        for j in range(start_idx, self.num_points):
            vars_map[j] = solver.NumVar(0, remaining_n, f'x_{j}')
            
        solver.Add(solver.Sum(vars_map.values()) == remaining_n)
        
        min_k = self.n - self.max_weight
        max_k = self.n - self.min_weight
        
        # 모든 초평면 제약 추가 (시간이 많이 걸림)
        for h in range(self.num_hyperplanes):
            k_curr = current_k_values[h]
            # expr = sum(x_j) for j in hyperplane h
            expr = solver.Sum([vars_map[j] for j in range(start_idx, self.num_points) 
                               if self.incidence[h][j] == 1])
            solver.Add(k_curr + expr >= min_k)
            solver.Add(k_curr + expr <= max_k)
            
        status = solver.Solve()
        return status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE

    def _verify(self, k_values):
        allowed_k = {self.n - w for w in self.target_weights}
        return all(k in allowed_k for k in k_values)


# --- 2. Comparison Runner ---
def run_comparison():
    # 기본 파라미터 (Default)
    n, k, q = 7, 3, 2
    weights = {3, 4}

    # 커맨드 라인 인자가 있으면 덮어쓰기
    if len(sys.argv) >= 4:
        try:
            n = int(sys.argv[1])
            k = int(sys.argv[2])
            q = int(sys.argv[3])
            if len(sys.argv) >= 5:
                weights_str = sys.argv[4]
                # "3,4" 또는 "[3,4]" 형태 처리
                weights_str = weights_str.replace('[', '').replace(']', '').replace('"', '')
                weights = set(map(int, weights_str.split(',')))
            print(f"[*] Using custom parameters: n={n}, k={k}, q={q}, weights={weights}")
        except ValueError:
            print("[!] Invalid arguments. Using default parameters.")
    else:
        print(f"[*] Using default parameters: n={n}, k={k}, q={q}, weights={weights}")
        print("    (To use custom parameters: python compare_performance.py <n> <k> <q> <weights>)")
    
    print(f"[*] Generating Geometry for PG({k-1}, {q})...")
    points = generate_projective_points(k, q)
    # PG(k-1, q)에서 점과 초평면은 동형
    hyperplanes = points 
    incidence = get_incidence_matrix(points, hyperplanes, q)
    
    results = []
    
    print(f"\n[*] Starting Comparison for [n={n}, k={k}, q={q}, W={weights}]")
    print("-" * 60)

    # CSV Headers aligned with experiment_results.csv + Method column
    headers = ["Method", "Length(n)", "Dimension(k)", "Field(q)", "Target_Weights", 
               "Search_Time(s)", "Nodes_Visited", "Pruned_Nodes", "Total_Solutions", "Note"]

    # 1. Run Pure Backtracking (No Pruning)
    print(">>> Running Pure Backtracking (No Pruning)...")
    start_time = time.time()
    pure_solver = BaselineSolver(n, k, q, weights, points, incidence, use_lp=False, use_bounds=False)
    pure_sols, pure_nodes, pure_pruned, _ = pure_solver.solve()
    pure_time = time.time() - start_time
    print(f"    Done. Time: {pure_time:.4f}s, Nodes: {pure_nodes}, Pruned: {pure_pruned}")
    
    results.append({
        "Method": "Pure Backtracking",
        "Length(n)": n, "Dimension(k)": k, "Field(q)": q, "Target_Weights": str(list(weights)),
        "Search_Time(s)": f"{pure_time:.4f}",
        "Nodes_Visited": pure_nodes,
        "Pruned_Nodes": pure_pruned,
        "Total_Solutions": len(pure_sols),
        "Note": "No Pruning"
    })

    # 2. Run Baseline (Paper)
    print(">>> Running Baseline (Paper Approach: LP Pruning)...")
    start_time = time.time()
    baseline = BaselineSolver(n, k, q, weights, points, incidence, use_lp=True, use_bounds=True)
    base_sols, base_nodes, base_pruned, base_lp = baseline.solve()
    base_time = time.time() - start_time
    print(f"    Done. Time: {base_time:.4f}s, Nodes: {base_nodes}, Pruned: {base_pruned}, LP Calls: {base_lp}")
    
    results.append({
        "Method": "Baseline (Paper)",
        "Length(n)": n, "Dimension(k)": k, "Field(q)": q, "Target_Weights": str(list(weights)),
        "Search_Time(s)": f"{base_time:.4f}",
        "Nodes_Visited": base_nodes,
        "Pruned_Nodes": base_pruned,
        "Total_Solutions": len(base_sols),
        "Note": f"LP Calls: {base_lp}"
    })

    # 3. Run Proposed (Method 1+2)
    print("\n>>> Running Proposed (All Optimizations)...")
    start_time = time.time()
    extender = CodeExtender(n, k, q, weights)
    # CodeExtender returns (solutions, nodes, pruned)
    prop_sols, prop_nodes, prop_pruned, prop_lp_calls = extender.build_and_solve(points, hyperplanes)
    prop_time = time.time() - start_time
    print(f"    Done. Time: {prop_time:.4f}s, Nodes: {prop_nodes}, Pruned: {prop_pruned}, LP Calls: {prop_lp_calls}")

    results.append({
        "Method": "Proposed (All Optimizations)",
        "Length(n)": n, "Dimension(k)": k, "Field(q)": q, "Target_Weights": str(list(weights)),
        "Search_Time(s)": f"{prop_time:.4f}",
        "Nodes_Visited": prop_nodes,
        "Pruned_Nodes": prop_pruned,
        "Total_Solutions": len(prop_sols),
        "Note": f"All optimizations incl. LP Pruning (Calls: {prop_lp_calls})"
    })

    # Save to CSV
    csv_file = "comparison_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print("-" * 60)
    print(f"[*] Comparison saved to '{csv_file}'")

if __name__ == "__main__":
    run_comparison()