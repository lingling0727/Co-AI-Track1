import pulp
import itertools
import time

class ILPCodePruner:
    def __init__(self):
        # 1. Proposition 4 파라미터 설정 (논문 Lemma 4 참조)
        self.n = 153          # 부호 길이
        self.k = 7            # 차원
        self.q = 2            # 유한체 크기
        
        # 허용된 가중치 (Weights)
        self.allowed_weights = {76, 80, 92, 96, 100}
        
        # 초평면 교점 수 (Intersection Sizes)
        # 공식: |H ∩ C| = n - wt(c)
        self.allowed_intersections = sorted([self.n - w for w in self.allowed_weights])
        # 결과: [53, 57, 61, 73, 77]
        
        print(f"[설정] n={self.n}, k={self.k}")
        print(f"[설정] 허용된 초평면 교점 수: {self.allowed_intersections}")
        
        # 2. 기하 구조 생성 (PG(6,2))
        self.points = self._generate_points()
        self.hyperplanes = self.points # Self-dual geometry
        self.num_points = len(self.points)
        self.num_hyperplanes = len(self.hyperplanes)
        
        print(f"[준비] 전체 변수(점) 개수: {self.num_points}")

    def _generate_points(self):
        """PG(k-1, q)의 점 생성 (정규화된 좌표)"""
        raw_vectors = list(itertools.product(range(self.q), repeat=self.k))
        raw_vectors.remove((0,)*self.k)
        
        points = []
        seen = set()
        for v in raw_vectors:
            # 첫 번째 0이 아닌 성분을 1로 정규화
            first_nz = next((x for x in v if x != 0), None)
            factor = pow(first_nz, -1, self.q)
            normalized = tuple((x * factor) % self.q for x in v)
            if normalized not in seen:
                seen.add(normalized)
                points.append(normalized)
        return points

    def create_ilp_model(self, target_idx=None):
        """
        ILP 모델 생성
        target_idx가 주어지면 해당 점(x[target_idx])을 최대화하는 문제로 설정
        """
        # 최대화 문제 생성
        prob = pulp.LpProblem("Code_Pruning", pulp.LpMaximize)
        
        # --- [수정된 부분] 문법 오류 해결 ---
        # 1. 메인 변수 x (각 점의 중복도)
        # 점의 개수는 정수(0, 1, 2)여야 함 (Proposition 4의 조건)
        x = pulp.LpVariable.dicts("x", 
                                  range(self.num_points), 
                                  lowBound=0, 
                                  upBound=2, 
                                  cat=pulp.LpInteger)
        
        # 2. 보조 변수 z (Gap Constraints 용)
        # z[h][s] = 1 이면 초평면 h의 교점 수가 s임
        # 이 변수가 없으면 53과 57 사이의 값(54 등)이 허용되어 버림
        z = pulp.LpVariable.dicts("z", 
                                  (range(self.num_hyperplanes), self.allowed_intersections), 
                                  cat=pulp.LpBinary)

        # 제약조건 1: 전체 길이 n = 153
        prob += pulp.lpSum(x[i] for i in range(self.num_points)) == self.n, "Total_Length"

        # 제약조건 2: 각 초평면의 교점 수는 허용된 값 중 하나여야 함
        for i, h in enumerate(self.hyperplanes):
            # 초평면 h에 포함된 점들의 인덱스 찾기
            points_in_h = [j for j, p in enumerate(self.points) 
                           if sum(h[k]*p[k] for k in range(self.k)) % self.q == 0]
            
            # 해당 점들의 합 (교점 수)
            intersection_sum = pulp.lpSum(x[j] for j in points_in_h)
            
            # Sum(x) = 53*z_53 + 57*z_57 + ... + 77*z_77
            prob += intersection_sum == pulp.lpSum(s * z[i][s] for s in self.allowed_intersections), f"Hyperplane_Val_{i}"
            
            # z 변수 중 딱 하나만 1이어야 함 (교점 수는 유일함)
            prob += pulp.lpSum(z[i][s] for s in self.allowed_intersections) == 1, f"Hyperplane_Select_{i}"

        # 목적 함수 설정
        if target_idx is not None:
            prob += x[target_idx]
        else:
            prob += 0 # Feasibility Check only

        return prob, x

    def run_pruning(self):
        print("\n--- 변수별 최대값 탐색 (Variable Probing) 시작 ---")
        print("목표: 각 점을 최대로 몇 개까지 쓸 수 있는지 확인하여 불가능한 점(0) 제거")
        
        pruned_count = 0
        valid_indices = []
        
        # 시간 절약을 위해 앞쪽 5개 점만 테스트 (실제로는 self.num_points 전체 반복)
        # 대칭성이 있어 초기에는 결과가 모두 같을 수 있음
        test_range = range(min(5, self.num_points)) 
        
        for i in test_range:
            prob, x = self.create_ilp_model(target_idx=i)
            
            # Solver 실행 (CBC Solver, 로그 끔, 시간제한 10초)
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
            
            status = pulp.LpStatus[prob.status]
            
            if status == 'Infeasible':
                print(f"[점 {i}] Infeasible (이 조건의 코드는 존재하지 않음)")
                return # 하나라도 불가능하면 전체 불가능
            
            elif status == 'Optimal':
                max_val = pulp.value(x[i])
                print(f"[점 {i}] 최대 가능 개수: {int(max_val)}")
                
                if max_val < 0.9: # 0개
                    print(f"   -> 제거됨 (Pruned)!")
                    pruned_count += 1
                else:
                    valid_indices.append(i)
            else:
                print(f"[점 {i}] 시간 초과 또는 오류")
                valid_indices.append(i) # 안전하게 유지

        print(f"\n[결과 요약]")
        print(f"테스트한 점 개수: {len(test_range)}")
        print(f"제거된 점 개수: {pruned_count}")
        print(f"남은 유효 점 개수: {len(valid_indices)}")

# 실행
if __name__ == "__main__":
    pruner = ILPCodePruner()
    pruner.run_pruning()
