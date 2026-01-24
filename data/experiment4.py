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
                                  lowBound=0, up
