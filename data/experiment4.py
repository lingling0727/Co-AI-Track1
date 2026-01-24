import numpy as np
import itertools
from scipy.optimize import linprog

class OptimizedCodeEnumerator:
    def __init__(self):
        # --- 1. 초기 파라미터 설정 (Proposition 4) ---
        self.n_target = 153
        self.k_target = 7
        self.q = 2
        
        # 허용된 가중치 (Lemma 4)
        self.allowed_weights = {76, 80, 92, 96, 100}
        
        # 기하 구조 생성 (PG(6, 2))
        print("1. 기하 구조(PG(6,2)) 생성 중...")
        self.points, self.A_matrix = self._generate_geometry(self.k_target, self.q)
        self.num_vars = len(self.points)
        print(f"   -> 전체 변수(점) 개수: {self.num_vars}")
        
        # 초평면 허용 교점 수 계산 (Intersection sizes)
        # |H ∩ C| = n - wt(c)
        self.allowed_intersections = {self.n_target - w for w in self.allowed_weights}
        print(f"   -> 허용된 초평면 교점 수: {sorted(list(self.allowed_intersections))}")

    def _generate_geometry(self, k, q):
        """PG(k-1, q)의 점과 접속 행렬(Incidence Matrix) 생성"""
        # 모든 0이 아닌 벡터 생성
        raw_vectors = list(itertools.product(range(q), repeat=k))
        raw_vectors.remove((0,)*k)
        
        points = []
        seen = set()
        for v in raw_vectors:
            # 정규화 (첫 0이 아닌 성분을 1로)
            first_nz = next((x for x in v if x != 0), None)
            factor = pow(first_nz, -1, q)
            normalized = tuple((x * factor) % q for x in v)
            if normalized not in seen:
                seen.add(normalized)
                points.append(normalized)
        
        # PG(k-1, q)에서 초평면의 개수는 점의 개수와 동일
        hyperplanes = points 
        num_items = len(points)
        
        # 접속 행렬 A 생성 (A_ij = 1 if P_j in H_i)
        A = np.zeros((num_items, num_items), dtype=int)
        for i, h in enumerate(hyperplanes):
            for j, p in enumerate(points):
                if sum(h[x]*p[x] for x in range(k)) % q == 0:
                    A[i, j] = 1
        return points, A

    def _add_gap_constraints(self, A_base, b_base_min, b_base_max):
        """
        [가지치기 단계]
        논문의 식 (13) 활용: 가중치 Gap을 이용한 Cut 추가.
        단순히 min/max만 보는 것이 아니라, '중간에 비어있는 값'을 불가능하게 만드는 제약은
        일반적인 LP로는 힘들므로, 여기서는 Tight한 Min/Max range로 변환하여 적용합니다.
        
        실제 논문에서는 정수 변수 z_H를 도입하지만, 
        여기서는 LP Relaxation을 위해 Hull(Convex Hull)을 좁히는 방식을 사용합니다.
        """
        # 현재 설정된 교점 수의 최소/최대값
        min_inters = min(self.allowed_intersections)
        max_inters = max(self.allowed_intersections)
        
        # 기본 제약조건: min_inters <= A*x <= max_inters
        # 이를 행렬 형태로 변환
        # -A*x <= -min_inters
        # A*x <= max_inters
        
        A_gap = np.vstack([ -A_base, A_base ])
        b_gap = np.concatenate([ -b_base_min, b_base_max ])
        
        return A_gap, b_gap

    def reduce_search_space(self):
        """
        [격자점 열거 최적화 핵심]
        LP 기반의 Bound Tightening (OBBT)을 수행하여
        각 점(변수) x_P가 가질 수 있는 범위를 계산하고,
        불필요한 변수(항상 0이어야 하는 점)를 제거합니다.
        """
        print("\n2. 변수(점) 최적화 및 제거 과정 시작 (LP-based Pruning)...")
        
        # 1. 기본 제약 조건 설정
        # sum(x) = 153
        A_eq = np.ones((1, self.num_vars))
        b_eq = np.array([self.n_target])
        
        # 초평면 제약 (Intersection Sizes)
        # 각 초평면 H에 대해: k_min <= sum(x in H) <= k_max
        min_k = min(self.allowed_intersections)
        max_k = max(self.allowed_intersections)
        
        b_lower = np.full(self.A_matrix.shape[0], min_k)
        b_upper = np.full(self.A_matrix.shape[0], max_k)
        
        # Gap Constraints 적용 (Inequalities)
        A_ub, b_ub = self._add_gap_constraints(self.A_matrix, b_lower, b_upper)
        
        # 2. 각 변수별 유효 범위 계산 (Variable Probing)
        # 각 변수 x_i에 대해 Minimize x_i, Maximize x_i를 수행
        valid_indices = []
        fixed_vars = 0
        pruned_vars = 0
        
        print("   -> 각 변수별 최대 가능값 계산 중 (이 과정은 시간이 조금 걸립니다)...")
        
        # 변수 범위: 0 <= x_i <= 2 (Point Multiplicity Constraint from Prop 4)
        bounds = [(0, 2) for _ in range(self.num_vars)]
        
        for i in range(self.num_vars):
            # 목적 함수: Maximize x_i (Minimize -x_i)
            c = np.zeros(self.num_vars)
            c[i] = -1 
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                max_val = -res.fun
                # 정수 제약 조건 적용 (LP 결과가 0.001이면 정수로는 0임)
                int_max = int(np.floor(max_val + 1e-9))
                
                if int_max == 0:
                    pruned_vars += 1
                    # print(f"      [Pruned] Point {i} cannot be selected (Max possible = 0)")
                else:
                    valid_indices.append(i)
            else:
                # Infeasible한 경우 (이론상 발생하면 안 됨)
                pass

        print(f"\n3. 최적화 결과:")
        print(f"   -> 초기 변수 개수: {self.num_vars}")
        print(f"   -> 제거된 변수 (Always 0): {pruned_vars}")
        print(f"   -> 남은 유효 변수: {len(valid_indices)}")
        
        reduced_ratio = (pruned_vars / self.num_vars) * 100
        print(f"   -> 검색 공간 축소율: {reduced_ratio:.2f}%")
        
        return valid_indices

# --- 실행 ---
optimizer = OptimizedCodeEnumerator()
valid_points = optimizer.reduce_search_space()

print("\n[결론] 격자점 열거 알고리즘은 이제 127차원이 아닌")
print(f"{len(valid_points)}차원 공간에서만 탐색을 수행하면 됩니다.")
