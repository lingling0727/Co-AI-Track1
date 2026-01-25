import itertools
import numpy as np
import os

class GaloisField:
    """
    유한체 GF(q)의 연산을 처리하는 클래스.
    소수체(Prime Field)와 작은 확장체(Extension Field, q=4, 8, 9)를 지원합니다.
    """
    def __init__(self, q):
        self.q = q
        self.is_prime = self._is_prime(q)
        self.mul_table = {}
        self.add_table = {}
        self.inv_table = {}
        
        if not self.is_prime:
            self._init_tables()

    def _is_prime(self, n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def _init_tables(self):
        if self.q == 4:
            # GF(4) ~= F2[x] / (x^2 + x + 1)
            # Elements: 0, 1, 2(x), 3(x+1)
            self.mul_table = {
                (0,0):0, (0,1):0, (0,2):0, (0,3):0,
                (1,0):0, (1,1):1, (1,2):2, (1,3):3,
                (2,0):0, (2,1):2, (2,2):3, (2,3):1,
                (3,0):0, (3,1):3, (3,2):1, (3,3):2
            }
            self.inv_table = {1:1, 2:3, 3:2}
            
        elif self.q == 8:
            # GF(8) ~= F2[x] / (x^3 + x + 1)
            # Log/Exp tables for multiplication
            exp = [1, 2, 4, 3, 6, 7, 5, 1, 2, 4, 3, 6, 7, 5]
            log = {1:0, 2:1, 4:2, 3:3, 6:4, 7:5, 5:6}
            
            for a in range(8):
                for b in range(8):
                    if a == 0 or b == 0:
                        self.mul_table[(a,b)] = 0
                    else:
                        self.mul_table[(a,b)] = exp[log[a] + log[b]]
            
            self.inv_table = {a: exp[(7 - log[a]) % 7] for a in range(1, 8)}

        elif self.q == 9:
            # GF(9) ~= F3[x] / (x^2 + 1)
            # Elements 0..8 represented as 3*high + low
            for a in range(9):
                for b in range(9):
                    h1, l1 = divmod(a, 3)
                    h2, l2 = divmod(b, 3)
                    
                    # Add: vector addition mod 3
                    h_sum = (h1 + h2) % 3
                    l_sum = (l1 + l2) % 3
                    self.add_table[(a,b)] = 3*h_sum + l_sum
                    
                    # Mul: (h1x + l1)(h2x + l2) = ... = (h1l2+l1h2)x + (l1l2 - h1h2)
                    # x^2 = -1 = 2 in F3
                    h_prod = (h1*l2 + l1*h2) % 3
                    l_prod = (l1*l2 + 2*h1*h2) % 3
                    self.mul_table[(a,b)] = 3*h_prod + l_prod
            
            for a in range(1, 9):
                for b in range(1, 9):
                    if self.mul_table[(a,b)] == 1:
                        self.inv_table[a] = b
                        break
        else:
            # q=16, 25 등은 추가 구현 필요
            raise NotImplementedError(f"GF({self.q}) is not supported yet. Only Prime fields and q=4,8,9.")

    def add(self, a, b):
        if self.is_prime:
            return (a + b) % self.q
        if self.q == 9:
            return self.add_table[(a,b)]
        return a ^ b # GF(2^m) is XOR

    def mul(self, a, b):
        if self.is_prime:
            return (a * b) % self.q
        # Symmetric check
        return self.mul_table.get((a,b), self.mul_table.get((b,a), 0))

    def inv(self, a):
        if a == 0: raise ValueError("Division by zero")
        if self.is_prime:
            return pow(a, self.q - 2, self.q)
        return self.inv_table[a]

    def dot(self, v1, v2):
        res = 0
        for a, b in zip(v1, v2):
            res = self.add(res, self.mul(a, b))
        return res

# 전역 캐시를 사용하여 GF 객체 재사용
_gf_cache = {}
def get_gf(q):
    if q not in _gf_cache:
        _gf_cache[q] = GaloisField(q)
    return _gf_cache[q]

def generate_projective_points(k, q):
    """
    PG(k-1, q)의 모든 점을 생성합니다.
    각 점은 첫 번째 0이 아닌 성분이 1인 벡터로 정규화됩니다.
    """
    gf = get_gf(q)
    points = set()
    all_vectors = itertools.product(range(q), repeat=k)

    for v in all_vectors:
        if all(x == 0 for x in v): continue

        # 정규화: 첫 번째 0이 아닌 성분(lead)을 찾아 그 역원을 전체에 곱함
        first_nonzero = next((x for x in v if x != 0), None)
        inv = gf.inv(first_nonzero)
        
        # v * inv
        normalized = tuple(gf.mul(x, inv) for x in v)
        points.add(normalized)

    # numpy 배열을 사용하지 않으므로 일반 튜플 리스트로 반환
    return sorted(list(points))

def get_projection_map(k_target, q, points_k, points_k_minus_1):
    """
    PG(k, q)의 점들을 PG(k-1, q)의 점들로 투영하는 매핑을 생성합니다.
    논문의 Lemma 1, Eq (4)를 위해 필요합니다.
    
    Returns:
        mapping: {point_k_minus_1_idx: [list of point_k_indices]}
        extension_point_idx: 투영 시 0이 되는 점(e_{k+1})의 인덱스
    """
    mapping = {i: [] for i in range(len(points_k_minus_1))}
    extension_point_idx = -1
    
    # points_k_minus_1을 빠른 조회를 위해 딕셔너리로 변환
    p_km1_to_idx = {p: i for i, p in enumerate(points_k_minus_1)}
    
    for idx_k, p_val in enumerate(points_k):
        # p_val = (x1, x2, ..., xk, x_{k+1})
        # 투영: 앞의 k개 성분만 취함 (x1, ..., xk)
        projected_vec = p_val[:-1]
        
        # 1. 투영 결과가 0벡터인 경우 -> 확장 중심점 (Extension Point)
        if all(x == 0 for x in projected_vec):
            extension_point_idx = idx_k
            continue
            
        # 2. 정규화하여 PG(k-1, q)에서의 인덱스 찾기
        # (geometry.py의 정규화 로직과 동일하게 처리해야 함)
        norm_p = normalize_point(projected_vec, q)
        
        if norm_p in p_km1_to_idx:
            mapping[p_km1_to_idx[norm_p]].append(idx_k)
            
    return mapping, extension_point_idx

def normalize_point(v, q):
    """벡터 v를 첫 번째 0이 아닌 성분이 1이 되도록 정규화"""
    gf = get_gf(q)
    if all(x == 0 for x in v): return v
    
    first_nonzero = next((x for x in v if x != 0), None)
    if first_nonzero == 1: return tuple(v)
    
    inv = gf.inv(first_nonzero)
    return tuple(gf.mul(x, inv) for x in v)

def generate_hyperplanes(k, q):
    """
    PG(k-1, q)의 모든 초평면(Hyperplane)을 생성합니다.
    초평면은 법선 벡터(Dual space의 점)로 표현됩니다.
    """
    # 사영 공간의 점과 초평면의 개수는 동일하며 구조적으로 동형입니다.
    return generate_projective_points(k, q)

def is_point_in_hyperplane(point, hyperplane, q):
    """
    점 P가 초평면 H에 포함되는지 확인 (내적 == 0)
    """
    gf = get_gf(q)
    return gf.dot(point, hyperplane) == 0

def get_incidence_matrix(points, hyperplanes, q):
    """
    점과 초평면 사이의 결합 행렬(Incidence Matrix) 생성
    Rows: Hyperplanes, Cols: Points
    """
    matrix = []
    for h in hyperplanes:
        row = []
        for p in points:
            if is_point_in_hyperplane(p, h, q):
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix)

if __name__ == "__main__":
    # 단독 실행 시 테스트 및 파일 저장
    print("[*] Running geometry.py directly...")
    test_k, test_q = 3, 4 # Test with Extension Field GF(4)
    print(f"[*] Generating points for PG({test_k-1}, {test_q})...")
    
    points = generate_projective_points(test_k, test_q)
    print(f"[*] Generated {len(points)} points: {points}")
    
    # dataset 폴더에 저장 테스트
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = os.path.join(directory, f"test_geometry_k{test_k}_q{test_q}.txt")
    with open(filename, "w") as f:
        f.write(f"# Test Run from geometry.py\n")
        for i, p in enumerate(points):
            f.write(f"{i}: {p}\n")
    print(f"[*] Test data saved to '{filename}'")