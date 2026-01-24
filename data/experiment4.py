import pulp
import itertools
import time
import sys
import io

# [수정 1] 한글 깨짐 방지 (강제 UTF-8 설정)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

class ILPCodePruner:
    def __init__(self):
        # [cite_start]1. 파라미터 설정 (Proposition 4) [cite: 1, 142]
        self.n = 153
        self.k = 7
        self.q = 2
        # [cite_start]Lemma 4와 Proposition 4에 명시된 허용 가중치 [cite: 180, 186]
        self.allowed_weights = {76, 80, 92, 96, 100}
        self.allowed_intersections = sorted([self.n - w for w in self.allowed_weights])
        
        print(f"[설정] n={self.n}, k={self.k}")
        print(f"[설정] 허용된 교점 수: {self.allowed_intersections}")
        
        # 2. 기하 구조 생성 (PG(6,2))
        self.points = self._generate_points()
        self.hyperplanes = self.points 
        self.num_points = len(self.points)
        self.num_hyperplanes = len(self.hyperplanes)
        
        print(f"[준비] 전체 변수 개수: {self.num_points}")

    def _generate_points(self):
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

    def create_feasibility_model(self, target_idx):
        """
        [전략 변경] 최대화 문제(Maximize)가 아니라, 
        'x[target_idx] >= 1'인 해가 존재하는지 묻는 '가능성 판별(Feasibility)' 모델
        """
        prob = pulp.LpProblem("Feasibility_Check", pulp.LpMinimize) # 목적함수 없음 (0)
        
        # 변수 설정 (정수 제약 포함)
        x = pulp.LpVariable.dicts("x", range(self.num_points), 
                                  lowBound=0, upBound=2, cat=pulp.LpInteger)
        
        z = pulp.LpVariable.dicts("z", 
                                  (range(self.num_hyperplanes), self.allowed_intersections), 
                                  cat=pulp.LpBinary)

        # 제약 1: 전체 길이
        prob += pulp.lpSum(x[i] for i in range(self.num_points)) == self.n

        # [cite_start]제약 2: Gap Constraints (빈 구간 제약) [cite: 126, 131]
        for i, h in enumerate(self.hyperplanes):
            points_in_h = [j for j, p in enumerate(self.points) 
                           if sum(h[k]*p[k] for k in range(self.k)) % self.q == 0]
            
            intersection_sum = pulp.lpSum(x[j] for j in points_in_h)
            
            # 교점 수는 허용된 값 중 하나여야 함
            prob += intersection_sum == pulp.lpSum(s * z[i][s] for s in self.allowed_intersections)
            prob += pulp.lpSum(z[i][s] for s in self.allowed_intersections) == 1

        # [핵심] "이 점을 1개 이상 쓸 수 있는가?" 제약 추가
        prob += x[target_idx] >= 1
        
        return prob

    def run_pruning(self):
        print("\n--- 빠른 가지치기 (Feasibility Check) 시작 ---")
        print("목표: 각 점을 1개라도 쓰는 것이 '가능'한지 확인 (불가능하면 제거)")
        
        pruned_count = 0
        valid_indices = []
        
        # 전체 점에 대해 수행
        test_range = range(self.num_points)
        
        total_start = time.time()
        
        for i in test_range:
            prob = self.create_feasibility_model(target_idx=i)
            
            # Solver 실행: 시간제한 5초 (가능성 확인은 금방 끝남)
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))
            
            status = pulp.LpStatus[prob.status]
            
            if status == 'Infeasible':
                print(f"[점 {i}] 불가능(Infeasible) -> 제거됨 (Pruned)!")
                pruned_count += 1
            elif status == 'Optimal' or status == 'Feasible':
                # 1개 이상 쓰는 해를 찾음 -> 유효한 점
                valid_indices.append(i)
            else:
                # 시간 초과 등 (보수적으로 유지)
                print(f"[점 {i}] 시간 초과/알수없음 -> 유지 (Status: {status})")
                valid_indices.append(i)

            # 진행 상황 표시 (10개마다)
            if (i+1) % 10 == 0:
                print(f"... {i+1}개 점 확인 완료 ({time.time() - total_start:.1f}초 경과)")

        print(f"\n[최종 결과]")
        print(f"전체 점: {self.num_points}")
        print(f"제거된 점(무조건 0): {pruned_count}")
        print(f"남은 후보 점: {len(valid_indices)}")
        print(f"소요 시간: {time.time() - total_start:.2f}초")

if __name__ == "__main__":
    pruner = ILPCodePruner()
    pruner.run_pruning()
