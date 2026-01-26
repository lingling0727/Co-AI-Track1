import time
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from geometry import get_incidence_matrix

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
        if not GUROBI_AVAILABLE:
            print("  > [Error] 'gurobipy' is not installed.")
            return [], 0, 0, 0.0, 0.0

        incidence_matrix = get_incidence_matrix(points, hyperplanes, self.q)

        # --- Phase 0: Feasibility Check ---
        start_p0 = time.time()
        print("    > [Phase 0] Running Gurobi feasibility check...")
        if not self._check_phase0_gurobi(points, incidence_matrix):
            print("    > [Phase 0] Infeasible. Stopping.")
            phase0_time = time.time() - start_p0
            return [], 0, 0, phase0_time, 0.0
        phase0_time = time.time() - start_p0

        # --- Phase 1: Full Enumeration ---
        start_p1 = time.time()
        print("    > [Phase 1] Starting Gurobi search...")
        solutions, nodes = self._solve_gurobi(points, incidence_matrix, base_code_counts, points_km1)
        phase1_time = time.time() - start_p1
        
        self.solutions = solutions
        self.nodes_visited = nodes
        
        return self.solutions, self.nodes_visited, 0, phase0_time, phase1_time

    def _check_phase0_gurobi(self, points, incidence_matrix):
        try:
            model = gp.Model("Phase0_Baseline")
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
        except Exception as e:
            print(f"      > Gurobi Error in Phase 0: {e}")
            return True

    def _solve_gurobi(self, points, incidence_matrix, base_code_counts=None, points_km1=None):
        try:
            model = gp.Model("CodeClassification_Baseline")
            model.setParam('OutputFlag', 0)
            model.setParam(GRB.Param.PoolSearchMode, 2)
            model.setParam(GRB.Param.PoolGap, 0.0)
            model.setParam(GRB.Param.PoolSolutions, 2000000)
            
            x = model.addVars(len(points), vtype=GRB.INTEGER, lb=0, ub=self.n, name="x")
            model.addConstr(x.sum() == self.n, "Length")
            
            allowed_k = sorted(list(self.allowed_intersections))
            if not allowed_k: return [], 0

            for h_idx in range(len(incidence_matrix)):
                expr = gp.quicksum(x[p_idx] for p_idx, val in enumerate(incidence_matrix[h_idx]) if val == 1)
                
                z = model.addVars(allowed_k, vtype=GRB.BINARY, name=f"z_{h_idx}")
                model.addConstr(z.sum() == 1)
                model.addConstr(expr == gp.quicksum(k * z[k] for k in allowed_k))

            if base_code_counts and points_km1:
                from geometry import get_projection_map
                mapping, _ = get_projection_map(self.k, self.q, points, points_km1)
                for p_km1_idx, p_k_indices in mapping.items():
                    target = base_code_counts.get(p_km1_idx, 0)
                    model.addConstr(gp.quicksum(x[i] for i in p_k_indices) == target)

            model.optimize()
            
            solutions = []
            if model.Status in [GRB.OPTIMAL, GRB.SOLUTION_LIMIT]:
                n_solutions = model.SolCount
                for i in range(n_solutions):
                    model.setParam(GRB.Param.SolutionNumber, i)
                    sol = {j: int(round(x[j].Xn)) for j in range(len(points)) if x[j].Xn > 0.5}
                    solutions.append(sol)
            
            return solutions, int(model.NodeCount)
            
        except Exception as e:
            print(f"      > Gurobi Error in Baseline Solve: {e}")
            return [], 0