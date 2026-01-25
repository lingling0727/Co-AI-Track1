import time
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: 'gurobipy' not found. Phase 0 (Gurobi) check will be skipped.")

from geometry import get_projection_map, get_incidence_matrix

class CodeExtender:
    """
    Custom Branch-and-Bound Solver implementing:
    - Phase 0: Feasibility check using Gurobi (if available)
    - Phase 1: Recursive search with Method 1 (Incremental Update) & Method 2 (RCUB Pruning)
    """
    def __init__(self, n, k, q, target_weights):
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = sorted(list(target_weights))
        
        # Precompute allowed intersection sizes: k_H = n - w
        # 코드의 가중치 w는 n - (초평면 위의 점 개수) 이므로,
        # 허용되는 점 개수 k_H는 n - w 입니다.
        self.allowed_intersections = {n - w for w in self.target_weights}
        if not self.allowed_intersections:
            self.min_allowed_k = 0
            self.max_allowed_k = n
        else:
            self.min_allowed_k = min(self.allowed_intersections)
            self.max_allowed_k = max(self.allowed_intersections)

        self.solutions = []
        self.nodes_visited = 0
        self.pruned_nodes = 0

    def build_and_solve(self, points, hyperplanes, base_code_counts=None, points_km1=None):
        """
        Main entry point for the solver.
        """
        self.solutions = []
        self.nodes_visited = 0
        self.pruned_nodes = 0
        num_points = len(points)
        num_hyperplanes = len(hyperplanes)

        print(f"    > Initializing Custom Solver (n={self.n}, k={self.k}, q={self.q})")
        print(f"    > Allowed Intersection Sizes on Hyperplanes: {sorted(list(self.allowed_intersections))}")

        # 1. Build Sparse Incidence Map (Method 1 Preparation)
        # geometry.py의 get_incidence_matrix는 dense matrix를 반환하므로 sparse 형태로 변환
        # point_to_hyperplanes[p_idx] = [h_idx1, h_idx2, ...]
        print("    > Building sparse incidence map...")
        incidence_matrix = get_incidence_matrix(points, hyperplanes, self.q)
        point_to_hyperplanes = [[] for _ in range(num_points)]
        for h_idx, row in enumerate(incidence_matrix):
            for p_idx, val in enumerate(row):
                if val == 1:
                    point_to_hyperplanes[p_idx].append(h_idx)

        # 2. Phase 0: Gurobi Feasibility Check
        # 본격적인 탐색 전에, 이 파라미터 조합이 이론적으로 가능한지 LP/ILP로 확인
        if GUROBI_AVAILABLE:
            print("    > [Phase 0] Running Gurobi feasibility check...")
            if not self._check_phase0_gurobi(points, incidence_matrix):
                print("    > [Phase 0] Problem is INFEASIBLE. Stopping.")
                return [], 0, 0
            print("    > [Phase 0] Passed. Starting enumeration.")

        # 3. Setup Initial State for Recursion
        # current_counts: {point_idx: count}
        current_counts = {}
        current_n = 0
        
        # current_hyperplane_counts: 각 초평면에 현재 몇 개의 점이 있는지 저장 (Method 1)
        current_hyperplane_counts = [0] * num_hyperplanes
        
        # Extension 모드일 경우 초기 상태 설정
        start_idx = 0
        if base_code_counts:
            # TODO: Extension 로직은 복잡하므로, 현재는 Scratch Construction 위주로 구현
            # 필요시 base_code_counts를 current_counts에 반영하고 current_hyperplane_counts 업데이트 필요
            pass

        # 4. Start Recursive Search
        self._backtrack(start_idx, current_n, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points)
        
        print(f"    > Search finished. Visited {self.nodes_visited} nodes, Pruned {self.pruned_nodes} branches.")
        return self.solutions, self.nodes_visited, self.pruned_nodes

    def _check_phase0_gurobi(self, points, incidence_matrix):
        """
        Use Gurobi to check if a valid configuration exists (Relaxed or Integer).
        This is a 'feasibility only' check.
        """
        try:
            model = gp.Model("Phase0")
            model.setParam('OutputFlag', 0) # Silence output
            
            # Variables: x_i (integer counts for each point)
            x = model.addVars(len(points), vtype=GRB.INTEGER, lb=0, ub=self.n, name="x")
            
            # Constraint: Total length
            model.addConstr(x.sum() == self.n, "TotalLength")
            
            # Constraints: Hyperplane weights
            # For each hyperplane H, sum(x_P) must be in allowed_intersections
            # To keep Phase 0 fast, we use the relaxed bounds [min, max]
            # If stricter check is needed, we can use SOS1 or indicator constraints
            for h_idx, row in enumerate(incidence_matrix):
                expr = gp.LinExpr()
                for p_idx, val in enumerate(row):
                    if val == 1:
                        expr.add(x[p_idx])
                
                model.addConstr(expr >= self.min_allowed_k, f"H_min_{h_idx}")
                model.addConstr(expr <= self.max_allowed_k, f"H_max_{h_idx}")

            model.optimize()
            return model.Status != GRB.INFEASIBLE
        except Exception as e:
            print(f"    > [Phase 0 Error] {e}")
            return True # Assume feasible if check fails

    def _backtrack(self, start_idx, current_n, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points):
        """
        Recursive Branch-and-Bound Search
        """
        self.nodes_visited += 1
        
        # Base Case: Solution Found
        if current_n == self.n:
            # Final check (just to be safe, though RCUB should prevent invalid states)
            if all(k in self.allowed_intersections for k in current_hyperplane_counts):
                self.solutions.append(current_counts.copy())
            return

        # Method 2: RCUB (Remaining Capacity Upper Bound) Pruning
        # 남은 점의 개수
        remaining = self.n - current_n
        
        # 모든 초평면에 대해, 남은 점들을 모두 그 초평면에 쏟아부어도 최소 요구치를 못 채우거나,
        # 이미 최대치를 넘어섰는지 검사
        for k_h in current_hyperplane_counts:
            # 1. 이미 초과한 경우
            if k_h > self.max_allowed_k:
                self.pruned_nodes += 1
                return
            # 2. 남은 걸 다 더해도 부족한 경우 (Min Bound Check)
            if k_h + remaining < self.min_allowed_k:
                self.pruned_nodes += 1
                return
        
        # Branching
        # start_idx부터 점을 하나씩 추가 (Combination with repetition)
        # Method 3 (Orbit Branching) 제거됨 -> 단순 순회
        for p_idx in range(start_idx, num_points):
            
            # Method 1: Incremental Update (Watched Hyperplane)
            # 점 p_idx를 추가했을 때 영향을 받는 초평면만 업데이트
            affected_hyperplanes = point_to_hyperplanes[p_idx]
            
            # Forward Step
            for h_idx in affected_hyperplanes:
                current_hyperplane_counts[h_idx] += 1
            
            current_counts[p_idx] = current_counts.get(p_idx, 0) + 1
            
            # Recurse
            self._backtrack(p_idx, current_n + 1, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points)
            
            # Backtrack (Undo changes)
            current_counts[p_idx] -= 1
            if current_counts[p_idx] == 0:
                del current_counts[p_idx]
                
            for h_idx in affected_hyperplanes:
                current_hyperplane_counts[h_idx] -= 1
        
        return