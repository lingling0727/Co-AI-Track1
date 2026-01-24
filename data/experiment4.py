import itertools
import time
import sys

class CustomSolverProp4:
    def __init__(self):
        # 1. 파라미터 설정
        self.n = 153
        self.k = 7
        self.q = 2
        
        # 목표: 허용된 교점 수 (Target Intersections)
        # Gap Constraints: 이 값들 외에는 절대 허용하지 않음
        self.allowed_values = {53, 57, 61, 73, 77}
        self.min_allowed = min(self.allowed_values) # 53
        self.max_allowed = max(self.allowed_values) # 77
        
        # 2. 기하 구조 생성
        print("1. 기하 구조 생성 중...")
        self.points = self._generate_points()
        self.num_points = len(self.points)
        
        # 3. [방법 1 준비] Watched-Hyperplane을 위한 인덱싱
        # point_to_hyperplanes[p]: 점 p가 포함된 초평면들의 리스트
        # hyperplane_counts[h]: 초평면 h에 포함된 점의 개수 (초기값)
        self.point_to_hyperplanes = [[] for _ in range(self.num_points)]
        self.hyperplanes = self.points # Self-dual
        
        for h_idx, h_vec in enumerate(self.hyperplanes):
            for p_idx, p_vec in enumerate(self.points):
                if sum(h_vec[i]*p_vec[i] for i in range(self.k)) % self.q == 0:
                    self.point_to_hyperplanes[p_idx].append(h_idx)
                    
        print(f"2. 인덱싱 완료. 전체 변수: {self.num_points}")
        
        # 대칭성 깨기 (앞서 했던 작업)
        self.basis_indices = self._find_basis_indices()
        
        # 솔루션 저장용
        self.found_solution = None

    def _generate_points(self):
        raw = list(itertools.product(range(self.q), repeat=self.k))
        raw.remove((0,)*self.k)
        points = []
        seen = set()
        for v in raw:
            first = next((x for x in v if x != 0), None)
            inv = pow(first, -1, self.q)
            norm = tuple((x * inv) % self.q for x in v)
            if norm not in seen:
                seen.add(norm)
                points.append(norm)
        return points

    def _find_basis_indices(self):
        # (1,0,0...), (0,1,0...) 등 찾기
        indices = []
        for i in range(self.k):
            vec = [0]*self.k
            vec[i] = 1
            try:
                indices.append(self.points.index(tuple(vec)))
            except: pass
        return set(indices)

    def solve(self):
        # 상태 초기화
        # current_solution[i]: 점 i의 선택 개수 (-1이면 미정)
        current_solution = [-1] * self.num_points
        
        # [방법 1 상태값] current_h_sums[h]: 현재까지 확정된 초평면 h의 교점 수 합
        current_h_sums = [0] * self.num_points
        
        # [방법 2 상태값] remaining_h_capacity[h]: 앞으로 더 추가될 수 있는 초평면 h의 최대 교점 수
        # 초기에는 모든 점이 미정이므로, 해당 초평면에 속한 점의 개수 * 2(최대 중복도)
        remaining_h_capacity = [0] * self.num_points
        for p_idx in range(self.num_points):
            for h_idx in self.point_to_hyperplanes[p_idx]:
                remaining_h_capacity[h_idx] += 1 # 여기서는 일단 binary(0,1)로 가정하고 품 (속도 위해)
                # 만약 중복도 2까지 허용하면 += 2 해야 함. (Proposition 4는 max 2)

        # 1. 대칭성 깨기: 기저 벡터는 무조건 1개 선택
        print("3. 탐색 시작 (Symmetry Breaking 적용)")
        for b_idx in self.basis_indices:
            current_solution[b_idx] = 1
            # 상태 업데이트 (Propagation)
            for h_idx in self.point_to_hyperplanes[b_idx]:
                current_h_sums[h_idx] += 1
                remaining_h_capacity[h_idx] -= 1 # 이미 결정됐으므로 미래 가능성에서 제외

        # 2. 재귀 탐색 시작
        # 기저 벡터가 아닌 첫 번째 인덱스부터 시작
        start_idx = 0
        while start_idx in self.basis_indices: start_idx += 1
        
        self.start_time = time.time()
        self.nodes_visited = 0
        
        if self._backtrack(start_idx, current_solution, current_h_sums, remaining_h_capacity):
            print("\n[성공] 해를 찾았습니다!")
            print(f"Solution: {self.found_solution}")
        else:
            print("\n[종료] 가능한 해가 없거나 탐색을 완료했습니다.")

    def _backtrack(self, p_idx, solution, h_sums, h_remains):
        self.nodes_visited += 1
        if self.nodes_visited % 100000 == 0:
            print(f"... {self.nodes_visited} 노드 탐색 중 ({time.time()-self.start_time:.1f}초)")

        # 1. 모든 변수가 결정되었는가?
        if p_idx >= self.num_points:
            # 최종 검증 (길이 153 체크 등)
            if sum(solution) == self.n:
                self.found_solution = solution[:]
                return True
            return False

        # 다음 탐색할 인덱스 (기저 벡터 건너뛰기)
        next_idx = p_idx + 1
        while next_idx < self.num_points and next_idx in self.basis_indices:
            next_idx += 1

        # 2. 분기 (Branching): x_p = 1 또는 0 (Proposition 4는 max 2지만, 일단 0/1로 좁혀서 탐색)
        # 1을 먼저 시도 (가능한 해를 빨리 찾기 위해)
        for val in [1, 0]: 
            
            # --- [방법 1: Watched-Hyperplane Propagation] ---
            # 값이 바뀌면 영향을 받는 초평면만 확인합니다.
            # val을 선택했을 때 위배되는 조건이 있는지 즉시 확인 (Forward Checking)
            is_valid_move = True
            
            affected_hyperplanes = self.point_to_hyperplanes[p_idx]
            
            for h_idx in affected_hyperplanes:
                new_sum = h_sums[h_idx] + val
                new_remain = h_remains[h_idx] - 1 # 이 점은 이제 결정되므로 미래 용량 감소
                
                # --- [방법 2: RCUB Pruning (미래 예측)] ---
                # "현재 합(new_sum) + 남은 모든 점을 다 1로 채워도(new_remain)" < 최소 요구치(53) 라면?
                # 가망이 없으므로 즉시 가지치기.
                if new_sum + new_remain < self.min_allowed:
                    is_valid_move = False
                    break
                
                # 상한 초과 가지치기
                if new_sum > self.max_allowed:
                    is_valid_move = False
                    break
                
                # (고급) Gap Pruning:
                # 더 이상 추가할 점이 없는데(new_remain == 0), 현재 합이 허용된 값 집합에 없다면?
                if new_remain == 0 and new_sum not in self.allowed_values:
                    is_valid_move = False
                    break

            if is_valid_move:
                # 상태 업데이트 (Do)
                solution[p_idx] = val
                for h_idx in affected_hyperplanes:
                    h_sums[h_idx] += val
                    h_remains[h_idx] -= 1
                
                # 재귀 호출
                if self._backtrack(next_idx, solution, h_sums, h_remains):
                    return True
                
                # 원상 복구 (Undo / Backtrack)
                for h_idx in affected_hyperplanes:
                    h_sums[h_idx] -= val
                    h_remains[h_idx] += 1
                solution[p_idx] = -1
        
        return False

# 실행
solver = CustomSolverProp4()
solver.solve()


# 이거 phrase1에 준원이 방법론 섞어서 했는데 실행시간 10분 넘게 걸림.
