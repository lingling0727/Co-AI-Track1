import time
import math
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from geometry import get_incidence_matrix, generate_gl_generators, get_orbits

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
        
        # --- Phase 0: ILP Feasibility Check ---
        incidence_matrix = get_incidence_matrix(points, hyperplanes, self.q)
        # (Gurobi가 없으면 실행 불가하므로 체크)
        if not GUROBI_AVAILABLE:
            print("    > [Error] Gurobi not found. Cannot run optimized search.")
            return [], 0, 0, 0, 0, 0, 0, "Gurobi_Not_Found"

        start_time = time.time()
        print("    > [Phase 0] Running Gurobi feasibility check...")
        if not self._check_phase0_gurobi(points, incidence_matrix):
            print("    > [Phase 0] Infeasible. Stopping.")
            phase0_time = time.time() - start_time
            return [], 0, 0, phase0_time, phase0_5_time, phase1_5_prep_time, search_time, "Infeasible_Phase0"
        phase0_time = time.time() - start_time
        
        # --- Phase 0.5: Theoretical Bounds Check ---
        start_time = time.time()
        print("    > [Phase 0.5] Checking theoretical bounds...")
        if not self._phase_0_5_checks():
            print("    > [Phase 0.5] Failed. Stopping.")
            phase0_5_time = time.time() - start_time
            return [], 0, 0, phase0_time, phase0_5_time, phase1_5_prep_time, search_time, "Infeasible_Phase0.5"
        phase0_5_time = time.time() - start_time
        
        # --- Phase 1.5: Symmetry Breaking (Orbital Branching) ---
        # 대칭성이 있는 점들(Orbit) 중 하나만 선택하도록 강제하여 Gurobi의 탐색 공간을 줄임
        orbits = []
        representatives = []
        if not base_code_counts:
            start_time = time.time()
            print("    > [Phase 1.5] Generating orbits for symmetry breaking...")
            try:
                matrices = generate_gl_generators(self.k, self.q)
                # get_orbits가 (reps, orbits_dict)를 반환한다고 가정 (geometry.py 확인 필요)
                # 현재 geometry.py의 get_orbits는 reps만 반환하므로, orbits_dict를 얻도록 수정하거나
                # 여기서는 reps만 있어도 되는지 확인. -> Orbital Branching을 하려면 Orbit 전체 목록이 필요함.
                # geometry.py의 get_orbits를 수정하지 않고 여기서 로직을 구현하기엔 복잡하므로
                # 간단히 reps만 사용하여 "첫 번째 선택 점"을 제한하는 전략 사용.
                
                # geometry.py의 get_orbits는 reps만 반환함.
                # 하지만 Orbital Branching을 제대로 하려면 "이전 Orbit의 점들은 모두 0"이라는 제약이 필요함.
                # 따라서 geometry.py의 get_orbits를 호출하되, 반환값을 활용해 직접 그룹핑을 다시 하거나
                # geometry.py를 수정해야 함. 
                # 여기서는 geometry.py를 수정하지 않고, reps만 사용하여 약식 Orbital Branching 수행.
                
                reps = get_orbits(points, matrices, self.q) # reps는 각 궤도의 대표점들
                point_to_idx = {p: i for i, p in enumerate(points)}
                
                # 대표점들의 인덱스
                rep_indices = [point_to_idx[p] for p in reps]
                print(f"    > [Phase 1.5] Found {len(rep_indices)} orbits. Applying Orbital Branching...")
                
            except Exception as e:
                print(f"    > [Phase 1.5] Symmetry breaking skipped: {e}")
                rep_indices = None
            
            phase1_5_prep_time = time.time() - start_time

        # --- Phase 1: Gurobi Search with Orbital Branching ---
        print("    > [Phase 1] Starting Gurobi search...")
        search_start = time.time()
        final_status = "Unknown"
        
        if rep_indices and not base_code_counts:
            # Orbital Branching: 각 궤도의 대표점에 대해 Gurobi를 개별 실행
            # 전략: "첫 번째로 선택되는 점(x >= 1)이 i번째 궤도의 대표점이다"라고 가정
            # 즉, 0 ~ i-1번째 궤도의 모든 점은 선택되지 않음(x=0)을 가정해야 완벽하지만,
            # 궤도 정보를 완벽히 모르면 "대표점 중 하나를 강제로 포함(x >= 1)"시키는 것만으로도 
            # 탐색 시작점을 분산시킬 수 있음. (완벽한 동형 제거는 Phase 2에서 수행)
            
            # 여기서는 "적어도 하나의 점은 존재해야 한다"는 가정 하에,
            # 그 "첫 번째 점"이 될 수 있는 후보를 대표점들로 한정합니다.
            
            # Gurobi 모델을 여러 번 푸는 대신, 하나의 모델에 "SOS1" 제약이나 "Indicator" 제약을 쓸 수도 있지만,
            # 명확한 분할을 위해 반복문으로 풉니다.
            
            # 주의: 단순히 x[rep] >= 1 만 걸면, 다른 궤도의 점들이 섞여 나올 때 중복이 발생할 수 있음.
            # 하지만 Baseline보다는 확실히 탐색 공간을 쪼개는 효과가 있음.
            
            status_set = set()
            
            for i, r_idx in enumerate(rep_indices):
                # 이번 분기: r_idx 점을 반드시 포함 (x >= 1)
                # 그리고 이전 대표점들은 포함하지 않음 (x = 0) -> 이를 위해서는 궤도 전체를 알아야 함.
                # 궤도 전체를 모르는 상태에서는 "이전 대표점들(r_0 ... r_{i-1})은 사용 안 함" 제약만 추가.
                # (이것만으로도 상당한 가지치기 효과)
                
                forbidden_indices = rep_indices[:i]
                forced_index = r_idx
                
                print(f"      > Branch {i+1}/{len(rep_indices)}: Force x_{forced_index} >= 1, Forbid previous reps")
                
                sub_solutions, sub_nodes, sub_status = self._solve_gurobi(points, incidence_matrix, 
                                                            forced_index=forced_index, 
                                                            forbidden_indices=forbidden_indices,
                                                            base_code_counts=base_code_counts,
                                                            points_km1=points_km1)
                self.solutions.extend(sub_solutions)
                self.nodes_visited += sub_nodes
                status_set.add(sub_status)
                
                # 만약 충분한 해를 찾았다면 조기 종료 가능 (옵션)
            
            # 상태 결정 로직
            if GRB.SOLUTION_LIMIT in status_set:
                final_status = "Solution_Limit"
            elif GRB.TIME_LIMIT in status_set:
                final_status = "Time_Limit"
            elif GRB.OPTIMAL in status_set:
                final_status = "Optimal"
            else:
                final_status = "Feasible" if self.solutions else "Infeasible"

        else:
            # 기본 Gurobi 실행 (Symmetry Breaking 없이)
            sub_solutions, sub_nodes, sub_status = self._solve_gurobi(points, incidence_matrix, 
                                                        base_code_counts=base_code_counts,
                                                        points_km1=points_km1)
            self.solutions.extend(sub_solutions)
            self.nodes_visited += sub_nodes
            
            if sub_status == GRB.OPTIMAL: final_status = "Optimal"
            elif sub_status == GRB.SOLUTION_LIMIT: final_status = "Solution_Limit"
            elif sub_status == GRB.TIME_LIMIT: final_status = "Time_Limit"
            else: final_status = "Feasible" if self.solutions else "Infeasible"
        
        search_time = time.time() - search_start
        return self.solutions, self.nodes_visited, self.pruned_nodes, phase0_time, phase0_5_time, phase1_5_prep_time, search_time, final_status

    def _solve_gurobi(self, points, incidence_matrix, forced_index=None, forbidden_indices=None, base_code_counts=None, points_km1=None):
        """
        Gurobi를 사용하여 ILP를 풉니다. (Baseline 로직 + 제약 조건 추가)
        """
        try:
            model = gp.Model("CodeClassification")
            model.setParam('OutputFlag', 0)
            model.setParam(GRB.Param.PoolSearchMode, 2)
            model.setParam(GRB.Param.PoolSolutions, 2000000)
            # [Optimization] Gurobi Tuning
            model.setParam('Symmetry', 2)  # Aggressive symmetry breaking
            model.setParam('Presolve', 2)  # Aggressive presolve
            
            x = model.addVars(len(points), vtype=GRB.INTEGER, lb=0, ub=self.n, name="x")
            
            # 1. 기본 제약: 길이 n
            model.addConstr(x.sum() == self.n, "Length")
            
            # 2. Phase 1.5 제약 (Orbital Branching)
            if forced_index is not None:
                model.addConstr(x[forced_index] >= 1, name="Force_Rep")
            
            if forbidden_indices:
                for idx in forbidden_indices:
                    model.addConstr(x[idx] == 0, name=f"Forbid_{idx}")
            
            # 3. 가중치 제약 (Incidence Matrix 활용)
            # Baseline과 동일하게 구현
            allowed_k = sorted(list(self.allowed_intersections))
            if not allowed_k: return [], 0 # 불가능
            
            # 최적화: 미리 incidence list 생성
            hyperplane_indices = [[] for _ in range(len(incidence_matrix))]
            for h_idx, row in enumerate(incidence_matrix):
                for p_idx, val in enumerate(row):
                    if val == 1: hyperplane_indices[h_idx].append(p_idx)
            
            for h_idx in range(len(incidence_matrix)):
                expr = gp.LinExpr()
                for p_idx in hyperplane_indices[h_idx]:
                    expr.add(x[p_idx])
                
                # [Optimization] If only one allowed k, use direct constraint (Huge speedup for fixed-weight codes)
                if len(allowed_k) == 1:
                    model.addConstr(expr == allowed_k[0], name=f"H_{h_idx}")
                else:
                    # z 변수를 사용하여 Disjunction 구현: expr == k1 OR expr == k2 ...
                    z = model.addVars(allowed_k, vtype=GRB.BINARY, name=f"z_{h_idx}")
                    model.addConstr(z.sum() == 1)
                    model.addConstr(expr == gp.quicksum(k * z[k] for k in allowed_k))

            # 4. 확장 제약 (Extension)
            if base_code_counts and points_km1:
                from geometry import get_projection_map
                mapping, _ = get_projection_map(self.k, self.q, points, points_km1)
                for p_km1_idx, p_k_indices in mapping.items():
                    target = base_code_counts.get(p_km1_idx, 0)
                    model.addConstr(gp.quicksum(x[i] for i in p_k_indices) == target)

            model.optimize()
            
            solutions = []
            # OPTIMAL 또는 SOLUTION_LIMIT(해를 찾다가 멈춤) 상태 모두 처리
            if model.Status in [GRB.OPTIMAL, GRB.SOLUTION_LIMIT]:
                n_solutions = model.SolCount
                for i in range(n_solutions):
                    model.setParam(GRB.Param.SolutionNumber, i)
                    sol = {}
                    for j in range(len(points)):
                        val = int(round(x[j].Xn))
                        if val > 0: sol[j] = val
                    solutions.append(sol)
                
                if model.NodeCount == 0:
                    print("      > [Gurobi] Solved at root node (Presolve Success).")
            
            return solutions, int(model.NodeCount), model.Status
            
        except Exception as e:
            print(f"      > Gurobi Error: {e}")
            return [], 0, -1

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