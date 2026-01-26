import time
import math
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from geometry import get_incidence_matrix, generate_linear_group, get_orbits

class CodeExtender:
    def __init__(self, n, k, q, target_weights):
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = sorted(list(target_weights))
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
        self.solutions = []
        self.nodes_visited = 0
        self.pruned_nodes = 0
        
        phase0_time = 0.0
        phase0_5_time = 0.0
        phase1_5_prep_time = 0.0
        search_time = 0.0
        
        # --- Phase 0: ILP Feasibility Check (Baseline) ---
        incidence_matrix = get_incidence_matrix(points, hyperplanes, self.q)
        if GUROBI_AVAILABLE:
            start_time = time.time()
            print("    > [Phase 0] Running Gurobi feasibility check...")
            if not self._check_phase0_gurobi(points, incidence_matrix):
                print("    > [Phase 0] Infeasible. Stopping.")
                phase0_time = time.time() - start_time
                return [], 0, 0, phase0_time, phase0_5_time, phase1_5_prep_time, search_time
            phase0_time = time.time() - start_time
        
        # --- Phase 0.5: Theoretical Bounds Check ---
        start_time = time.time()
        print("    > [Phase 0.5] Checking theoretical bounds...")
        if not self._phase_0_5_checks():
            print("    > [Phase 0.5] Failed. Stopping.")
            phase0_5_time = time.time() - start_time
            return [], 0, 0, phase0_time, phase0_5_time, phase1_5_prep_time, search_time
        phase0_5_time = time.time() - start_time
        
        # --- Phase 1.5 Preparation: Symmetry Breaking ---
        initial_candidates = None
        if not base_code_counts:
            start_time = time.time()
            print("    > [Phase 1.5] Generating orbits for symmetry breaking...")
            try:
                matrices = generate_linear_group(self.k, self.q, limit=2000)
                reps = get_orbits(points, matrices, self.q)
                point_to_idx = {p: i for i, p in enumerate(points)}
                initial_candidates = sorted([point_to_idx[p] for p in reps])
                print(f"    > [Phase 1.5] Reduced initial branches: {len(points)} -> {len(initial_candidates)}")
            except Exception as e:
                print(f"    > [Phase 1.5] Symmetry breaking skipped: {e}")
            phase1_5_prep_time = time.time() - start_time

        # --- Phase 1: Backtracking ---
        print("    > [Phase 1] Starting recursive search...")
        search_start = time.time()
        
        point_to_hyperplanes = [[] for _ in range(len(points))]
        for h_idx, row in enumerate(incidence_matrix):
            for p_idx, val in enumerate(row):
                if val == 1: point_to_hyperplanes[p_idx].append(h_idx)

        current_hyperplane_counts = [0] * len(hyperplanes)
        self._backtrack(0, 0, {}, current_hyperplane_counts, point_to_hyperplanes, len(points), initial_candidates)
        
        search_time = time.time() - search_start
        return self.solutions, self.nodes_visited, self.pruned_nodes, phase0_time, phase0_5_time, phase1_5_prep_time, search_time

    def _phase_0_5_checks(self):
        """
        Phase 0.5: Griesmer Bound & Pless Power Moments
        """
        # 1. Griesmer Bound
        if self.target_weights:
            d = min(self.target_weights)
            griesmer = sum(math.ceil(d / (self.q ** i)) for i in range(self.k))
            if self.n < griesmer:
                print(f"      - Griesmer Bound Failed: n={self.n} < {griesmer}")
                return False

        # 2. Pless Power Moments (if Gurobi available)
        if GUROBI_AVAILABLE and self.target_weights:
            try:
                model = gp.Model("Pless")
                model.setParam('OutputFlag', 0)
                A = {w: model.addVar(vtype=GRB.INTEGER, lb=0) for w in self.target_weights}
                
                # Moment 0: Sum(A_i) = q^k - 1
                model.addConstr(gp.quicksum(A.values()) == (self.q**self.k) - 1)
                # Moment 1: Sum(w * A_i) = n * q^(k-1) * (q-1)
                target_m1 = self.n * (self.q**(self.k-1)) * (self.q - 1)
                model.addConstr(gp.quicksum(w * A[w] for w in self.target_weights) == target_m1)
                
                model.optimize()
                if model.Status == GRB.INFEASIBLE:
                    print("      - Pless Power Moments Failed.")
                    return False
            except Exception:
                pass
        
        return True

    def _check_phase0_gurobi(self, points, incidence_matrix):
        try:
            model = gp.Model("Phase0")
            model.setParam('OutputFlag', 0)
            x = model.addVars(len(points), vtype=GRB.INTEGER, lb=0, ub=self.n)
            model.addConstr(x.sum() == self.n)
            
            for h_idx, row in enumerate(incidence_matrix):
                expr = gp.LinExpr()
                for p_idx, val in enumerate(row):
                    if val == 1: expr.add(x[p_idx])
                model.addConstr(expr >= self.min_allowed_k)
                model.addConstr(expr <= self.max_allowed_k)
            
            model.optimize()
            return model.Status != GRB.INFEASIBLE
        except:
            return True

    def _backtrack(self, start_idx, current_n, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points, candidates=None):
        self.nodes_visited += 1
        
        if current_n == self.n:
            if all(k in self.allowed_intersections for k in current_hyperplane_counts):
                self.solutions.append(current_counts.copy())
            return

        remaining = self.n - current_n
        
        # --- Phase 1.5: Lookahead Pruning ---
        # Check if any hyperplane is already violated or cannot be satisfied
        for k_h in current_hyperplane_counts:
            if k_h > self.max_allowed_k:
                self.pruned_nodes += 1
                return
            if k_h + remaining < self.min_allowed_k:
                self.pruned_nodes += 1
                return

        # Branching
        loop_iterable = candidates if candidates is not None else range(start_idx, num_points)
        
        for p_idx in loop_iterable:
            # Update state
            affected = point_to_hyperplanes[p_idx]
            for h in affected: current_hyperplane_counts[h] += 1
            current_counts[p_idx] = current_counts.get(p_idx, 0) + 1
            
            # Recurse
            # Note: candidates is only for the first level (Phase 1.5 Symmetry)
            self._backtrack(p_idx, current_n + 1, current_counts, current_hyperplane_counts, point_to_hyperplanes, num_points, None)
            
            # Restore state
            current_counts[p_idx] -= 1
            if current_counts[p_idx] == 0: del current_counts[p_idx]
            for h in affected: current_hyperplane_counts[h] -= 1
            
            if self.solutions: return # Find one solution