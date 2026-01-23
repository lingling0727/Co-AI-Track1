from ortools.sat.python import cp_model
from geometry import is_point_in_hyperplane, get_projection_map

class CodeExtender:
    """
    논문의 Lemma 1을 기반으로 ILP 모델을 생성하고 해를 열거하는 클래스.
    Phase 0 (Feasibility)와 Phase 1 (Enumeration)을 담당합니다.
    """
    def __init__(self, n, k, q, target_weights):
        """
        n: 확장될 코드의 목표 길이
        k: 확장될 코드의 목표 차원
        q: 유한체 크기
        target_weights: 확장된 코드 C'에서 허용되는 가중치 집합
        """
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = target_weights

    def build_and_solve(self, points, hyperplanes, base_code_counts=None, points_km1=None):
        """
        논문 Lemma 1에 기반한 CP-SAT 모델 생성 및 해 열거
        points: PG(k-1, q)의 점 리스트
        hyperplanes: PG(k-1, q)의 초평면 리스트
        base_code_counts: (Optional) 확장 전 k-1차원 코드의 점 중복도 {idx: count}
        points_km1: (Optional) 확장 전 PG(k-2, q)의 점 리스트
        """
        model = cp_model.CpModel()

        # 1. 변수 정의: x_p
        # x_p는 점 p가 코드에 포함되는 횟수(multiplicity)를 나타냅니다. (논문의 x_P)
        # x_p >= 0 (Eq 6)
        x = {p: model.NewIntVar(0, self.n, f'x_{i}') for i, p in enumerate(points)}

        # 2. 제약 조건 (전체 길이)
        # sum(x_p) = n
        model.Add(sum(x.values()) == self.n)

        # 3. 제약 조건 (가중치) - 논문 Eq (3)
        # 모든 초평면 H에 대해, 코드 C'의 가중치 wt(c_H)는 target_weights에 속해야 합니다.
        # wt(c_H) = n - |C' ∩ H| = n - sum_{p in H} x_p
        # 따라서, sum_{p in H} x_p = n - w (여기서 w는 허용된 가중치)
        
        # 허용되는 교차(intersection) 크기 집합을 미리 계산합니다.
        allowed_intersection_sizes = [self.n - w for w in self.target_weights if w != 0]
        intersection_domain = cp_model.Domain.FromValues(allowed_intersection_sizes)

        for h in hyperplanes:
            # 초평면 h에 포함되는 점들의 x_p 변수 합
            intersection_sum_expr = sum(x[p] for p in points if is_point_in_hyperplane(p, h, self.q))
            
            # 이 합은 `intersection_domain`에 속해야 합니다.
            # 이 제약이 Phase 0의 역할을 수행하여, 불가능한 경우를 빠르게 걸러냅니다.
            model.AddLinearExpressionInDomain(intersection_sum_expr, intersection_domain)
            
        # 4. 확장 제약 (Extension Constraints) - 논문 Eq (4)
        # sum_{q in Fq} x_{(u|q)} = c(u)
        if base_code_counts is not None and points_km1 is not None:
            print("    > Applying Extension Constraints (Lemma 1, Eq 4)...")
            # 투영 매핑 계산
            proj_map, ext_point_idx = get_projection_map(self.k, self.q, points, points_km1)
            
            # 각 u in P_{k-1}에 대해 제약 조건 추가
            for u_idx, u_point in enumerate(points_km1):
                c_u = base_code_counts.get(u_idx, 0)
                
                # u로 투영되는 P_k의 점들의 합 == c(u)
                if u_idx in proj_map:
                    model.Add(sum(x[points[p_idx]] for p_idx in proj_map[u_idx]) == c_u)
            
            # 확장 중심점(Extension Point)에 대한 처리는 논문에서 c(0) = r로 정의됨
            # r = n_new - n_old
            n_old = sum(base_code_counts.values())
            r = self.n - n_old

            # --- 논문 Eq (2) "Canonical Length Extension" 제약 추가 ---
            # 모든 점 P에 대해, x_P = 0 또는 x_P >= r 이어야 함.
            # 이는 ILP로 선형화하기 위해 보조 이진 변수 u_P를 사용.
            if r > 1: # r=1이면 모든 non-zero x_p >= 1 이므로 자명함.
                print(f"    > Applying Canonical Length Extension Constraint (r={r})...")
                u = {p: model.NewBoolVar(f'u_{i}') for i, p in enumerate(points)}

                for p, x_var in x.items():
                    # u[p]=1 이면 x_var >= r 을 강제
                    # u[p]=0 이면 x_var = 0 을 강제 (아래 제약과 결합하여)
                    model.Add(x_var >= r * u[p])
                    # Big-M 제약: x_var는 u[p]가 1일 때만 non-zero 값을 가질 수 있음
                    model.Add(x_var <= self.n * u[p])

            if ext_point_idx != -1:
                model.Add(x[points[ext_point_idx]] == r)
                
        else:
            # 바닥부터 생성(Scratch) 시 대칭성 제거
            for i in range(self.k):
                basis_vector = tuple(1 if j == i else 0 for j in range(self.k))
                if basis_vector in x:
                    model.Add(x[basis_vector] >= 1)

        # 5. 솔버 설정 및 실행 (Phase 1: 모든 해 열거)
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        
        solution_printer = CodeSolutionPrinter(x, points)
        status = solver.Solve(model, solution_printer)
        
        print(f"    > Solver status: {solver.StatusName(status)}")
        
        return solution_printer.solutions

class CodeSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """CP-SAT 솔버가 찾은 모든 해를 수집하는 콜백 클래스"""
    def __init__(self, variables, points):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__points = {p: i for i, p in enumerate(points)} # 빠른 조회를 위한 튜플->인덱스 맵
        self.solutions = []
        self.solution_count = 0
        self.solution_limit = 1000 # 너무 많은 해가 나올 경우를 대비한 안전장치

    def on_solution_callback(self):
        self.solution_count += 1
        
        # 해를 {point_index: count} 형태의 딕셔너리로 저장
        current_solution = {self.__points[p]: self.Value(var) for p, var in self.__variables.items() if self.Value(var) > 0}
        self.solutions.append(current_solution)
        
        if self.solution_count >= self.solution_limit:
            print(f"    > Stopping search after reaching the solution limit of {self.solution_limit}.")
            self.StopSearch()