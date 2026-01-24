import pulp
import itertools
import time

class ILPCodePruner:
    def __init__(self):
        # 1. Proposition 4 파라미터 설정
        self.n = 153
        self.k = 7
        self.q = 2
        
        # 허용된 가중치 및 초평면 교점 수
        self.allowed_weights = {76, 80, 92, 96, 100}
        self.allowed_intersections = sorted([self.n - w for w in self.allowed_weights])
        # 예: [53, 57, 61, 73, 77] (이 값들만 허용됨, 54, 55 등은 불가능)
        
        print(f"[설정] n={self.n}, k={self.k}, 허용 교점 수={self.allowed_intersections}")
        
        # 2. 기하 구조 생성 (PG(6,2))
        self.points = self._generate_points()
        self.hyperplanes = self.points # Self-dual geometry
        self.num_points = len(self.points)
        self.num_hyperplanes = len(self.hyperplanes)
        
        print(f"[준비] 변수(점) 개수: {self.num_points}, 초평면 개수: {self.num_hyperplanes}")

    def _generate_points(self):
        """PG(k-1, q)의 점 생성"""
        raw_vectors = list(itertools.product(range(self.q), repeat=self.k))
        raw_vectors.remove((0,)*self.k)
        
        points = []
        seen = set()
        for v in raw_vectors:
            first_nz = next((x for x in v if x != 0), None)
            factor = pow(first_nz, -1, self.q)
            normalized = tuple((x * factor) % self.q for x in v)
            if normalized not in seen:
                seen.add(normalized)
                points.append(normalized)
        return points

    def create_ilp_model(self, fixed_basis=None):
        """
        ILP 모델 생성
        fixed_basis: 리스트 [(index, value), ...] 형태로 특정 점의 값을 고정 (대칭성 깨기 용도)
        """
        # 문제 정의 (최대화/최소화는 나중에 설정)
        prob = pulp.LpProblem("Code_Extension_Pruning", pulp.LpMaximize)
        
        # 1. 메인 변수 x (각 점의 중복도)
        # 정수 조건(Integer) 강제: 0, 1, 2
        x = pulp.LpVariable.dicts("x", range(self.num_points), 
                                  lowBound=0, upBound=2, cat=pulp.LpInteger)
        
        # 2. 보조 변수 z (Gap Constraints 용)
        # z[h][s] = 1 이면 초평면 h의 교점 수가 s임
        z = pulp.LpVariable.dicts("z", 
                                  (range(self.num_hyperplanes), self.allowed_intersections), 
                                  cat=pulp.LpBinary)

        # 제약조건 1: 전체 길이 n
        prob += pulp.lpSum(x[i] for i in range(self.num_points)) == self.n, "Total_Length"

        # 제약조건 2 & 3: 초평면 교점 수 제약 (Gap Constraints)
        for i, h in enumerate(self.hyperplanes):
            # 해당 초평면에 포함된 점들의 합 계산
            points_in_h = [j for j, p in enumerate(self.points) 
                           if sum(h[k]*p[k] for k in range(self.k)) % self.q == 0]
            
            sum_x_in_h = pulp.lpSum(x[j] for j in points_in_h)
            
            # Sum(x) = Sum(s * z_s)  (교점 수는 허용된 값 중 하나여야 함)
            # 예: x_1 + ... + x_k = 53*z_53 + 57*z_57 + ...
            prob += sum_x_in_h == pulp.lpSum(s * z[i][s] for s in self.allowed_intersections), f"Hyperplane_Value_{i}"
            
            # 정확히 하나의 교점 값만 선택되어야 함: Sum(z) = 1
            prob += pulp.lpSum(z[i][s] for s in self.allowed_intersections) == 1, f"Hyperplane_Select_{i}"

        # (선택) Basis Fixing: 대칭성을 깨기 위해 일부 점 고정
        if fixed_basis:
            for idx, val in fixed_basis:
                prob += x[idx] == val, f"Fixed_Basis_{idx}"

        return prob, x

    def probe_variable(self, target_idx, fixed_basis=None):
        """특정 변수(x_target)가 가질 수 있는 최대값을 계산하여 Pruning 여부 확인"""
        prob, x = self.create_ilp_model(fixed_basis)
        
        # 목적 함수: 해당 변수 최대화
        prob += x[target_idx]
        
        # Solver 실행 (CBC Solver 사용, 시간 제한 설정 가능)
        # msg=False로 로그 숨김
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10)) 
        
        status = pulp.LpStatus[prob.status]
        
        if status == 'Infeasible':
            return -1  # 불가능 (전역적으로 해가 없음)
        elif status == 'Optimal':
            return pulp.value(x[target_idx])
        else:
            return 999 # 시간 초과 등

    def run_pruning(self):
        print("\n--- ILP Variable Probing 시작 ---")
        print("참고: 대칭성이 있으므로 Basis를 고정하지 않으면 모든 점의 결과가 동일할 수 있습니다.")
        
        # 예시: 기저(Basis)를 고정하지 않은 상태에서 첫 번째 점(x_0) 확인
        print("\n[Case 1] Basis 고정 없이 x_0 탐색 중...")
        start = time.time()
        max_val = self.probe_variable(0)
        end = time.time()
        
        if max_val == -1:
            print(f"   -> 결과: Infeasible! (모든 점이 불가능 -> 존재하지 않음 증명됨)")
        else:
            print(f"   -> 결과: x_0의 최대 가능 값 = {int(max_val)} (소요시간: {end-start:.2f}초)")
            print("      (해석: 0이 아니므로 이 점은 제거되지 않음)")

        # 예시: Basis를 일부 고정하고 나머지 탐색 (실제 논문의 방식)
        # 첫 번째 점을 1로 고정했다고 가정
        print("\n[Case 2] x_0 = 1로 고정 후 x_1 탐색 중...")
        fixed = [(0, 1)] 
        max_val_2 = self.probe_variable(1, fixed_basis=fixed)
        
        if max_val_2 == -1:
            print(f"   -> 결과: x_0=1일 때 x_1은 불가능함 (x_1은 반드시 0이어야 하거나 모순)")
        else:
            print(f"   -> 결과: x_0=1일 때 x_1의 최대 가능 값 = {int(max_val_2)}")

# 실행
if __name__ == "__main__":
    try:
        pruner = ILPCodePruner()
        pruner.run_pruning()
    except ImportError:
        print("오류: pulp 라이브러리가 필요합니다. 'pip install pulp'를 실행해주세요.")
        pruner = ILPCodePruner()
        pruner.run_pruning()
    except ImportError:
        print("오류: pulp 라이브러리가 필요합니다. 'pip install pulp'를 실행해주세요.")
