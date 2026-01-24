import pulp
import itertools
import time
import sys
import io

# [필수] 한글 깨짐 방지 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

class ILPCodePruner:
    def __init__(self):
        # 1. 파라미터 설정 (Proposition 4)
        self.n = 153
        self.k = 7
        self.q = 2
        # 허용된 가중치 (Lemma 4)
        self.allowed_weights = {76, 80, 92, 96, 100}
        self.allowed_intersections = sorted([self.n - w for w in self.allowed_weights])
        
        print(f"[설정] n={self.n}, k={self.k}")
        print(f"[설정] 허용된 교점 수: {self.allowed_intersections}")
        
        # 2. 기하 구조 생성 (PG(6,2))
        self.points = self._generate_points()
        self.hyperplanes = self.points 
        self.num_points = len(self.points)
        self.num_hyperplanes = len(self.hyperplanes)
        
        print(f"[준비] 전체 변수(점) 개수: {self.num_points}")

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
        [핵심 전략 변경] 
        최대값을 구하지 않고, 'x[target_idx] >= 1'인 해가 존재하는지만 확인합니다.
        계산 속도가 100배 이상 빠릅니다.
        """
        # 목적함수가 없는 '가능성 판별(Feasibility)' 문제로 설정
        prob = pulp.LpProblem("Feasibility_Check", pulp.LpMinimize)
        
        # 변수 설정 (정수 제약: 0, 1, 2)
        x = pulp.LpVariable.dicts("x", range(self.num_points), 
                                  lowBound=0, upBound=2, cat=pulp.LpInteger)
        
        # Gap Constraints용 보조 변수
        z = pulp.LpVariable.dicts("z", 
                                  (range(self.num_hyperplanes), self.allowed_intersections), 
                                  cat=pulp.LpBinary)

        # 제약 1: 전체 길이 153
        prob += pulp.lpSum(x[i] for i in range(self.num_points)) == self.n

        # 제약 2: 빈 구간 제약 (Gap Constraints)
        for i, h in enumerate(self.hyperplanes):
            points_in_h = [j for j, p in enumerate(self.points) 
                           if sum(h[k]*p[k] for k in range(self.k)) % self.q == 0]
            
            intersection_sum = pulp.lpSum(x[j] for j in points_in_h)
            
            # 교점 수는 반드시 허용된 값 중 하나여야 함
            prob += intersection_sum == pulp.lpSum(s * z[i][s] for s in self.allowed_intersections)
            prob += pulp.lpSum(z[i][s] for s in self.allowed_intersections) == 1

        # [질문] "이 점을 1개 이상 쓰는 것이 가능한가?"
        prob += x[target_idx] >= 1
        
        return prob

    def run_pruning(self):
        print("\n--- 빠른 가지치기 (Feasibility Check) 시작 ---")
        print("목표: 각 점을 1개라도 쓰는 것이 '가능'한지 확인 (불가능하면 제거)")
        
        pruned_count = 0
        valid_indices = []
        
        # 전체 127개 점에 대해 수행
        test_range = range(self.num_points)
        
        total_start = time.time()
        
        for i in test_range:
            prob = self.create_feasibility_model(target_idx=i)
            
            # 시간제한 2초 (가능성 확인은 금방 끝남)
            # msg=False로 불필요한 로그 숨김
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=2))
            
            status = pulp.LpStatus[prob.status]
            
            if status == 'Infeasible':
                # "1개 이상 쓰는 게 불가능하다" -> 즉, 무조건 0이어야 함
                print(f"[점 {i}] 불가능(Infeasible) -> 제거됨 (Pruned)!")
                pruned_count += 1
            elif status == 'Optimal' or status == 'Feasible':
                # 가능한 경우
                # print(f"[점 {i}] 가능 -> 유지")
                valid_indices.append(i)
            else:
                # 시간 초과 등 (안전하게 유지)
                print(f"[점 {i}] 판단 불가(Status: {status}) -> 유지")
                valid_indices.append(i)

            # 10개마다 진행 상황 출력
            if (i+1) % 10 == 0:
                print(f"... {i+1}개 점 확인 완료 ({time.time() - total_start:.1f}초 경과)")

        print(f"\n[최종 결과]")
        print(f"전체 점: {self.num_points}")
        print(f"제거된 점(무조건 0): {pruned_count}")
        print(f"남은 후보 점: {len(valid_indices)}")
        print(f"총 소요 시간: {time.time() - total_start:.2f}초")

if __name__ == "__main__":
    pruner = ILPCodePruner()
    pruner.run_pruning()
