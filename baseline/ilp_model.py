import gurobipy as gp
from gurobipy import GRB
from geometry import get_incidence_matrix, get_projection_map, is_point_in_hyperplane

class CodeExtender:
    def __init__(self, n, k, q, target_weights):
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = sorted(list(target_weights))

    def build_and_solve(self, points, hyperplanes, base_code_counts=None, points_km1=None):
        """
        Builds and solves the ILP model using Gurobi.
        Returns a list of solutions (dicts of point counts).
        """
        # Create Model
        model = gp.Model("CodeClassification")
        model.setParam('OutputFlag', 0)  # Silent mode
        
        # Variables: x[i] is the multiplicity of point i
        x = {}
        for i in range(len(points)):
            x[i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}")
        
        # [Paper Match] Lemma 1, Eq (5): Systematic Generator Matrix Assumption
        # 기저 벡터(단위 벡터)는 반드시 코드에 포함되어야 한다고 가정 (x >= 1)
        if base_code_counts is None:
            for i, p in enumerate(points):
                if p.count(1) == 1 and p.count(0) == self.k - 1:
                    x[i].LB = 1

        # Constraint 1: Total Length n
        model.addConstr(gp.quicksum(x[i] for i in range(len(points))) == self.n, "Length")
        
        # Constraint 2: Weights
        # For each hyperplane H, sum(x_P for P in H) must be k_H
        # where n - k_H in target_weights.
        # So k_H in {n - w for w in target_weights}
        
        allowed_k = [self.n - w for w in self.target_weights]
        
        # Precompute incidence matrix for efficiency
        # (Assuming hyperplanes and points are aligned with indices)
        incidence = get_incidence_matrix(points, hyperplanes, self.q)
        
        for h_idx in range(len(hyperplanes)):
            # Calculate the sum of points in this hyperplane
            # Using incidence matrix: sum(x[i] * incidence[h][i])
            # Note: incidence[h][i] is 1 if point i is in hyperplane h
            
            expr = gp.quicksum(x[i] for i in range(len(points)) if incidence[h_idx][i] == 1)
            
            # The sum must be one of the allowed values
            # We use binary variables z to enforce this disjunction
            z = model.addVars(allowed_k, vtype=GRB.BINARY, name=f"z_{h_idx}")
            
            # Select exactly one allowed k
            model.addConstr(gp.quicksum(z[k_val] for k_val in allowed_k) == 1)
            
            # Link expr to the selected k
            model.addConstr(expr == gp.quicksum(k_val * z[k_val] for k_val in allowed_k))

        # Constraint 3: Extension (if applicable)
        if base_code_counts is not None and points_km1 is not None:
            mapping, ext_idx = get_projection_map(self.k, self.q, points, points_km1)
            
            # For each point P' in PG(k-2, q), the sum of multiplicities of points 
            # projecting to P' must equal the multiplicity of P' in the base code.
            for p_km1_idx, p_k_indices in mapping.items():
                target_count = base_code_counts.get(p_km1_idx, 0)
                model.addConstr(gp.quicksum(x[i] for i in p_k_indices) == target_count, 
                                name=f"Ext_{p_km1_idx}")
            
            # Note: The extension point (ext_idx) multiplicity is implicitly handled 
            # by the total length constraint and the sum of base code counts.

        # Solve Configuration
        # We want to find ALL feasible solutions (Classification)
        model.setParam(GRB.Param.PoolSearchMode, 2)  # 2 = Find all best solutions
        model.setParam(GRB.Param.PoolSolutions, 2000000)  # Large buffer
        model.setParam(GRB.Param.PoolGap, 0)  # Only optimal (feasible) solutions
        
        model.optimize()
        
        solutions = []
        if model.Status == GRB.OPTIMAL:
            n_solutions = model.SolCount
            for i in range(n_solutions):
                model.setParam(GRB.Param.SolutionNumber, i)
                sol = {}
                for j in range(len(points)):
                    val = x[j].Xn
                    if val > 0.5:
                        sol[j] = int(round(val))
                solutions.append(sol)
            
            # [Info] NodeCount가 0인 경우 사용자에게 이유를 알림
            if model.NodeCount == 0:
                print("    > [Gurobi] Solved during presolve or at root node (NodeCount=0).")
        
        # Return format: solutions, nodes_visited, pruned_nodes
        # Gurobi doesn't give "pruned_nodes" in the same sense as backtracking, 
        # so we return NodeCount for visited and 0 for pruned.
        return solutions, int(model.NodeCount), 0