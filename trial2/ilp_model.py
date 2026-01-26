import time
import math
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: 'gurobipy' not found. Phase 0 (Gurobi) check will be skipped.")

from geometry import get_projection_map, get_incidence_matrix, generate_linear_group, get_orbits

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
        self.lp_pruning_calls = 0
        
        self.early_exit = False

        # Persistent Gurobi Model Storage
        self.lp_model = None
        self.lp_vars = []
        self.lp_constrs_min = []
        self.lp_constrs_max = []
        self.lp_constr_total = None

    def build_and_solve(self, points, hyperplanes, base_code_counts=None, points_km1=None, early_exit=False):
        """
        Main entry point for the solver.
        """
        self.solutions = []
        self.nodes_visited = 0
        self.pruned_nodes = 0
        self.lp_pruning_calls = 0
        self.early_exit = early_exit
        num_points = len(points)
        num_hyperplanes = len(hyperplanes)

        print(f"    > Initializing Custom Solver (n={self.n}, k={self.k}, q={self.q})")
        print(f"    > Allowed Intersection Sizes on Hyperplanes: {sorted(list(self.allowed_intersections))}")

        precomp_start_time = time.time()

        # 0. Theoretical Bounds Check (Pre-computation)
        if not self._check_theoretical_bounds():
            print("    > [Pre-check] Theoretical bounds check failed. Stopping.")
            precomp_time = time.time() - precomp_start_time
            return [], 0, 0, 0, precomp_time, 0.0

        # 0.5. Phase 0.5: Recursive Residual Check
        if not self._phase_0_5_recursive_checks():
             print("    > [Phase 0.5] Recursive checks failed. Stopping.")
             precomp_time = time.time() - precomp_start_time
             return [], 0, 0, 0, precomp_time, 0.0
             
        # 0.6. Phase 0.5: Pless Power Moments Check (MacWilliams Identities)
        if not self._check_pless_moments():
            print("    > [Phase 0.5] Pless Power Moments check failed. Stopping.")
            precomp_time = time.time() - precomp_start_time
            return [], 0, 0, 0, precomp_time, 0.0

        # 0.7. Phase 0.5: Hamming Bound Check
        if not self._check_hamming_bound():
            print("    > [Phase 0.5] Hamming (Sphere-Packing) Bound check failed. Stopping.")
            precomp_time = time.time() - precomp_start_time
            return [], 0, 0, 0, precomp_time, 0.0

        # 0.8. Phase 0.5: Dual Projective Bound Check
        if not self._check_dual_projective_bound():
            print("    > [Phase 0.5] Dual Projective Bound check failed. Stopping.")
            precomp_time = time.time() - precomp_start_time
            return [], 0, 0, 0, precomp_time, 0.0

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
                precomp_time = time.time() - precomp_start_time
                return [], 0, 0, 0, precomp_time, 0.0
            print("    > [Phase 0] Passed. Starting enumeration.")
            
            # [Optimization] Initialize Persistent LP Model for Pruning
            print("    > [Optimization] Initializing Persistent Gurobi Model for LP Pruning...")
            self._init_persistent_lp_model(points, incidence_matrix)

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

        # [Method 3] Symmetry Breaking (Initial Branching Optimization)
        # 첫 번째 점을 선택할 때, 전체 점을 다 시도하는 대신 대칭 그룹의 궤도(Orbit) 대표값만 시도합니다.
        # 이는 탐색 트리의 최상단 너비를 획기적으로 줄여줍니다.
        initial_candidates = None
        if not base_code_counts: # 처음부터 생성하는 경우에만 적용
            print("    > [Symmetry] Generating Linear Group and Orbits for Symmetry Breaking...")
            try:
                # 그룹 생성 (너무 오래 걸리지 않도록 limit 설정)
                matrices = generate_linear_group(self.k, self.q, limit=5000)
                reps, _ = get_orbits(points, matrices, self.q)
                
                # 대표 점들을 인덱스로 변환
                point_to_idx = {p: i for i, p in enumerate(points)}
                initial_candidates = sorted([point_to_idx[p] for p in reps])
                print(f"    > [Symmetry] Reduced initial branching from {num_points} to {len(initial_candidates)} nodes.")
            except Exception as e:
                print(f"    > [Symmetry] Failed to apply symmetry breaking: {e}")
                initial_candidates = None

        precomp_time = time.time() - precomp_start_time

        # 4. Start Recursive Search
        search_start_time = time.time()
        self._backtrack(start_idx, current_n, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points, incidence_matrix, candidates=initial_candidates)
        search_time = time.time() - search_start_time
        
        print(f"    > Search finished. Visited {self.nodes_visited} nodes, Pruned {self.pruned_nodes} branches, LP Pruning Calls: {self.lp_pruning_calls}.")
        
        # Cleanup Gurobi Model
        if self.lp_model:
            self.lp_model.dispose()
            self.lp_model = None
            
        return self.solutions, self.nodes_visited, self.pruned_nodes, self.lp_pruning_calls, precomp_time, search_time

    def _check_theoretical_bounds(self):
        """
        Check theoretical bounds (e.g., Griesmer Bound) before starting the search.
        Returns True if the parameters are theoretically feasible, False otherwise.
        """
        # Griesmer Bound: n >= sum(ceil(d / q^i)) for i in 0..k-1
        if not self.target_weights:
            return True
            
        d = min(self.target_weights)
        griesmer_sum = sum(math.ceil(d / (self.q ** i)) for i in range(self.k))
        
        if self.n < griesmer_sum:
            print(f"    > [Griesmer Bound] FAILED. For [n={self.n}, k={self.k}, d={d}]_q={self.q}, length must be >= {griesmer_sum}.")
            return False
        return True

    def _phase_0_5_recursive_checks(self):
        """
        Phase 0.5: Refine allowed weights using recursive Griesmer bounds.
        If a hyperplane has weight w, the residual code has parameters [n-w, k-1, >= ceil(d/q)].
        If n-w is too small to support such a code, then weight w is impossible.
        """
        if not self.target_weights or self.k <= 1:
            return True

        d_min = min(self.target_weights)
        # Residual code minimum distance lower bound: d' >= ceil(d / q)
        d_res = math.ceil(d_min / self.q)
        
        # Calculate required length for residual code using Griesmer bound for dimension k-1
        # g_val = sum(ceil(d_res / q^i) for i in 0..k-2)
        g_val = sum(math.ceil(d_res / (self.q ** i)) for i in range(self.k - 1))
        
        to_remove = set()
        for k_h in self.allowed_intersections:
            # k_h is the length of the residual code (n - w)
            if k_h < g_val:
                to_remove.add(k_h)
        
        if to_remove:
            print(f"    > [Phase 0.5] Pruning impossible intersection sizes (Residual Griesmer): {to_remove}")
            self.allowed_intersections -= to_remove
            if not self.allowed_intersections:
                return False
            self.min_allowed_k = min(self.allowed_intersections)
            self.max_allowed_k = max(self.allowed_intersections)
            
        return True

    def _check_pless_moments(self):
        """
        Check if there exists a valid weight distribution {A_w} satisfying the first 3 Pless Power Moments.
        This is a necessary condition for the existence of a [n, k]_q projective code.
        """
        if not GUROBI_AVAILABLE:
            return True # Skip if solver is not available
            
        try:
            model = gp.Model("PlessMoments")
            model.setParam('OutputFlag', 0)
            
            # Variables: A_w (number of codewords with weight w)
            # A_w must be non-negative integers
            A = {w: model.addVar(vtype=GRB.INTEGER, lb=0, name=f"A_{w}") for w in self.target_weights}
            
            # Constants
            q = self.q
            k = self.k
            n = self.n
            
            # Moment 0: Sum(A_w) = q^k - 1
            m0_rhs = (q**k) - 1
            model.addConstr(gp.quicksum(A[w] for w in self.target_weights) == m0_rhs, "Moment0")
            
            # Moment 1: Sum(w * A_w) = n * q^(k-1) * (q-1)
            m1_rhs = n * (q**(k-1)) * (q-1)
            model.addConstr(gp.quicksum(w * A[w] for w in self.target_weights) == m1_rhs, "Moment1")
            
            # Moment 2: Sum(w^2 * A_w) = ... (Formula for Projective Codes)
            # Formula: n(q-1)q^(k-1) + n(n-1)(q-1)^2 q^(k-2)
            if k >= 2:
                term1 = n * (q-1) * (q**(k-1))
                term2 = n * (n-1) * ((q-1)**2) * (q**(k-2))
                m2_rhs = term1 + term2
                model.addConstr(gp.quicksum(w*w * A[w] for w in self.target_weights) == m2_rhs, "Moment2")
            
            model.optimize()
            
            if model.Status == GRB.INFEASIBLE:
                print(f"    > [Pless Moments] INFEASIBLE. No valid weight distribution exists for weights {self.target_weights}.")
                return False
                
            return True
            
        except Exception as e:
            print(f"    > [Pless Moments] Error: {e}")
            return True

    def _check_hamming_bound(self):
        """
        Checks the Hamming (sphere-packing) bound.
        q^(n-k) >= sum_{i=0 to t} C(n,i) * (q-1)^i, where t = floor((d-1)/2)
        """
        if not self.target_weights:
            return True
        
        d = min(self.target_weights)
        t = math.floor((d - 1) / 2)
        
        if t < 0:
            return True

        try:
            sphere_volume = 0
            for i in range(t + 1):
                # math.comb(n, k) is available in Python 3.8+
                term = math.comb(self.n, i) * ((self.q - 1)**i)
                sphere_volume += term
            
            if self.q**(self.n - self.k) < sphere_volume:
                print(f"    > [Hamming Bound] FAILED. For [n={self.n}, k={self.k}, d={d}]_q={self.q}, q^(n-k)={self.q**(self.n - self.k)} is smaller than sphere volume={sphere_volume}.")
                return False
        except (ValueError, OverflowError) as e:
            # Binomial coefficient can be very large. If it fails, we can't check.
            print(f"    > [Hamming Bound] Warning: Could not compute bound due to large numbers: {e}")
        
        return True

    def _check_dual_projective_bound(self):
        """
        Checks the Griesmer bound on the dual code, assuming the code is projective (d_perp >= 3).
        This provides another lower bound on n.
        """
        if self.k >= self.n: # Dual dimension must be positive
            return True
            
        d_perp = 3
        dual_k = self.n - self.k
        
        # Apply Griesmer to dual code [n, n-k, d_perp=3]
        griesmer_sum_dual = sum(math.ceil(d_perp / (self.q ** i)) for i in range(dual_k))
        
        if self.n < griesmer_sum_dual:
            print(f"    > [Dual Projective Bound] FAILED. Assuming d_perp>=3, for dual code [n={self.n}, k'={dual_k}]_q={self.q}, length must be >= {griesmer_sum_dual}.")
            return False
            
        return True

    def _init_persistent_lp_model(self, points, incidence_matrix):
        """
        Initialize a persistent Gurobi model for LP pruning.
        This avoids the overhead of creating a new model at every node.
        """
        try:
            self.lp_model = gp.Model("PersistentPrune")
            self.lp_model.setParam('OutputFlag', 0)
            self.lp_model.setParam('Threads', 1) # Single thread for low overhead
            
            num_points = len(points)
            # Variables: x[0]...x[num_points-1] (Continuous for Relaxation)
            self.lp_vars = self.lp_model.addVars(num_points, vtype=GRB.CONTINUOUS, lb=0.0, name="x").values()
            
            # Constraint 1: Total Sum (Placeholder RHS)
            # sum(x) == remaining_n
            self.lp_constr_total = self.lp_model.addConstr(gp.quicksum(self.lp_vars) == 0, "TotalSum")
            
            # Constraint 2: Hyperplanes (Placeholder RHS)
            # min_k <= current_k + sum(A_ij * x_j) <= max_k
            # => sum(A_ij * x_j) >= min_k - current_k
            # => sum(A_ij * x_j) <= max_k - current_k
            self.lp_constrs_min = []
            self.lp_constrs_max = []
            
            for h_idx, row in enumerate(incidence_matrix):
                expr = gp.quicksum(self.lp_vars[p_idx] for p_idx, val in enumerate(row) if val == 1)
                # Add constraints with dummy RHS
                c_min = self.lp_model.addConstr(expr >= 0, f"H_min_{h_idx}")
                c_max = self.lp_model.addConstr(expr <= 0, f"H_max_{h_idx}")
                self.lp_constrs_min.append(c_min)
                self.lp_constrs_max.append(c_max)
                
            self.lp_model.update()
            
        except Exception as e:
            print(f"    > [Gurobi Init Error] {e}")
            self.lp_model = None

    def _lp_prune(self, remaining_n, current_hyperplane_counts, start_idx, num_points, incidence_matrix):
        """
        Solves an LP relaxation to check if the current partial solution is extendable.
        If the LP is infeasible, this branch can be pruned.
        """
        if not GUROBI_AVAILABLE or self.lp_model is None or remaining_n < 2:
            return True

        self.lp_pruning_calls += 1
        try:
            # 1. Update Variable Bounds
            # Variables before start_idx must be 0 (since we only pick from start_idx onwards)
            # Variables from start_idx onwards can be up to remaining_n
            for j in range(num_points):
                self.lp_vars[j].UB = 0.0 if j < start_idx else float(remaining_n)

            # 2. Update Total Sum Constraint RHS
            self.lp_constr_total.RHS = float(remaining_n)

            # 3. Update Hyperplane Constraints RHS
            for h_idx, k_h in enumerate(current_hyperplane_counts):
                self.lp_constrs_min[h_idx].RHS = self.min_allowed_k - k_h
                self.lp_constrs_max[h_idx].RHS = self.max_allowed_k - k_h

            # 4. Solve
            self.lp_model.optimize()
            
            return self.lp_model.Status != GRB.INFEASIBLE

        except Exception as e:
            print(f"    > [LP Prune Error] {e}")
            return True # Fail-safe: if error, don't prune

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

    def _backtrack(self, start_idx, current_n, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points, incidence_matrix, candidates=None):
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

        # [NEW] Method 4: LP-based Pruning (Paper's approach)
        # Call it conditionally to manage overhead. e.g., every few levels
        if current_n > 0 and remaining > 2:
             if not self._lp_prune(remaining, current_hyperplane_counts, start_idx, num_points, incidence_matrix):
                 self.pruned_nodes += 1
                 return
        
        # Branching
        # start_idx부터 점을 하나씩 추가 (Combination with repetition)
        # candidates가 주어지면(주로 depth 0) 해당 후보만 순회, 아니면 range 순회
        loop_iterable = candidates if candidates is not None else range(start_idx, num_points)
        
        for p_idx in loop_iterable:
            
            # Method 1: Incremental Update (Watched Hyperplane)
            # 점 p_idx를 추가했을 때 영향을 받는 초평면만 업데이트
            affected_hyperplanes = point_to_hyperplanes[p_idx]
            
            # Forward Step
            for h_idx in affected_hyperplanes:
                current_hyperplane_counts[h_idx] += 1
            
            current_counts[p_idx] = current_counts.get(p_idx, 0) + 1
            
            # Recurse
            # 다음 깊이부터는 candidates 없이 순차 탐색 (중복 조합이므로 start_idx는 현재 p_idx)
            self._backtrack(p_idx, current_n + 1, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points, incidence_matrix, candidates=None)
            
            # Early exit for fair comparison
            if self.early_exit and self.solutions:
                return

            # Backtrack (Undo changes)
            current_counts[p_idx] -= 1
            if current_counts[p_idx] == 0:
                del current_counts[p_idx]
                
            for h_idx in affected_hyperplanes:
                current_hyperplane_counts[h_idx] -= 1
        
        return