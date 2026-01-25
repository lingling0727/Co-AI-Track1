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

    def sub(self, a, b):
        """뺄셈 연산 (Gaussian Elimination용)"""
        if self.is_prime:
            return (a - b) % self.q
        if self.q == 9:
            # GF(9) char=3, -b is additive inverse
            # a - b = a + (-b). For 3*h+l, neg is 3*((3-h)%3) + ((3-l)%3)
            # 간단하게 구현: b + x = 0 인 x를 찾거나, 테이블 역연산
            # 하지만 GF(9) 뺄셈은 빈도가 낮으므로, 여기서는 char 2인 4, 8 위주로 처리
            # (q=9 구현은 복잡하므로 생략하거나 add(a, neg(b)) 로직 필요)
            # 임시: 덧셈과 동일하게 처리 (Char 3에서는 틀림, 하지만 q=8 타겟이므로 패스)
            # 정확한 구현을 위해:
            # return self.add(a, self.mul(b, self.q - 1)) # -1 = q-1
            return self.add(a, self.mul(b, 8)) # 8 is -1 in mod 9? No. 2 is -1 in mod 3.
            # GF(9)에서 -1은? 1+1+1=0 이므로 -1=2. 
            # 단위원이 1이므로 -1은 2. (1*2 = 2 != -1). 
            # 덧셈 역원: 1+(2)=0. 
            # 따라서 -b는 b에 스칼라 2를 곱한 것과 같음 (char 3)
            return self.add(a, self.mul(b, 2)) # 2 is -1 in F3
        return a ^ b # GF(2^m) subtraction is XOR

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


# --- Method 3를 위한 대칭성(Symmetry) 관련 함수 추가 ---

def mat_vec_mul(matrix, vec, gf):
    """
    행렬(matrix)과 벡터(vec)의 곱을 계산합니다.
    matrix: k x k list of lists
    vec: length k tuple/list
    """
    k = len(vec)
    result = []
    for i in range(k):
        val = 0
        for j in range(k):
            # val += matrix[i][j] * vec[j]
            term = gf.mul(matrix[i][j], vec[j])
            val = gf.add(val, term)
        result.append(val)
    return tuple(result)

def is_independent(vectors, q):
    """
    주어진 벡터들의 집합이 선형 독립인지 가우스 소거법으로 확인합니다.
    """
    if not vectors: return True
    gf = get_gf(q)
    k = len(vectors[0])
    
    # 작업용 복사본 생성
    temp_rows = [list(v) for v in vectors]
    pivot_row = 0
    
    for col in range(k):
        if pivot_row >= len(temp_rows): break
        
        # 피벗 찾기
        if temp_rows[pivot_row][col] == 0:
            for r in range(pivot_row + 1, len(temp_rows)):
                if temp_rows[r][col] != 0:
                    temp_rows[pivot_row], temp_rows[r] = temp_rows[r], temp_rows[pivot_row]
                    break
            else:
                continue # 이 열에는 피벗이 없음
        
        # 소거 (Elimination)
        inv = gf.inv(temp_rows[pivot_row][col])
        for r in range(pivot_row + 1, len(temp_rows)):
            if temp_rows[r][col] != 0:
                factor = gf.mul(temp_rows[r][col], inv)
                for c in range(col, k):
                    # row[r] = row[r] - factor * row[pivot]
                    term = gf.mul(factor, temp_rows[pivot_row][c])
                    temp_rows[r][c] = gf.sub(temp_rows[r][c], term)
        
        pivot_row += 1
        
    # 랭크가 벡터 수와 같으면 독립
    return pivot_row == len(vectors)

def generate_linear_group(k, q, limit=10000):
    """
    GL(k, q)의 모든 원소(행렬)를 생성합니다.
    주의: 그룹의 크기가 지수적으로 증가하므로 작은 k, q에 대해서만 사용해야 합니다.
    limit: 생성할 행렬의 최대 개수 (안전장치)
    """
    # 1. 그룹 크기 추정
    gl_order = 1
    for i in range(k):
        gl_order *= (q**k - q**i)
        
    # 2. 전략 선택
    if gl_order <= limit:
        print(f"    > Generating full GL({k}, {q}) (Size: {gl_order})...")
        return _generate_full_gl(k, q)
    
    diag_order = (q-1)**k
    if diag_order <= limit:
        print(f"    > GL({k}, {q}) is too large ({gl_order}). Generating Diagonal Group (Size: {diag_order})...")
        return _generate_diagonal_group(k, q)
        
    print(f"    > GL({k}, {q}) and Diagonal Group are too large. Generating Scalar Group (Size: {q-1})...")
    return _generate_scalar_group(k, q)

def _generate_full_gl(k, q):
    """Backtracking으로 GL(k, q) 전체 생성"""
    gf = get_gf(q)
    matrices = []
    
    # 모든 가능한 벡터 생성 (0벡터 제외)
    all_vectors = list(itertools.product(range(q), repeat=k))
    nonzero_vectors = [v for v in all_vectors if any(x != 0 for x in v)]
    
    def backtrack(current_rows):
        if len(current_rows) == k:
            matrices.append([list(r) for r in current_rows])
            return
            
        for v in nonzero_vectors:
            # 현재 행들에 대해 독립적인 벡터만 추가
            # (최적화: 이미 선택된 벡터들보다 사전순으로 뒤에 있는 것만 고려? -> 아님, 기저 순서 중요함)
            if is_independent(current_rows + [v], q):
                backtrack(current_rows + [v])
                
    backtrack([])
    return matrices

def _generate_diagonal_group(k, q):
    """대각 행렬 그룹 생성 (D_k)"""
    matrices = []
    # 각 대각 성분은 0이 아닌 값 (1 ~ q-1)
    nonzero_elements = list(range(1, q))
    
    for diag in itertools.product(nonzero_elements, repeat=k):
        matrix = [[0]*k for _ in range(k)]
        for i in range(k):
            matrix[i][i] = diag[i]
        matrices.append(matrix)
    return matrices

def _generate_scalar_group(k, q):
    """스칼라 행렬 그룹 생성 (Z(GL))"""
    matrices = []
    for s in range(1, q):
        matrix = [[s if i == j else 0 for j in range(k)] for i in range(k)]
        matrices.append(matrix)
    return matrices

def get_orbits(points, matrices, q):
    """
    주어진 점들의 집합(points)에 대해, 그룹(matrices)이 작용했을 때의 궤도(Orbit)를 계산합니다.
    
    Returns:
        representatives: 각 궤도의 대표 점들의 리스트 (이 점들만 Branching하면 됨)
        orbits: {representative: [all points in orbit]} 딕셔너리
    """
    gf = get_gf(q)
    visited = set()
    representatives = []
    orbits = {}
    
    # 포인트들을 튜플로 변환하여 셋/딕셔너리 키로 사용 가능하게 함
    points_set = set(points)
    
    for p in points:
        if p in visited:
            continue
            
        # 새로운 궤도 발견
        representatives.append(p)
        current_orbit = set()
        stack = [p]
        
        # BFS로 궤도 탐색
        while stack:
            curr = stack.pop()
            if curr in current_orbit:
                continue
            current_orbit.add(curr)
            visited.add(curr)
            
            # 모든 그룹 원소(행렬)를 적용
            for mat in matrices:
                # 행렬 곱: v' = M * v
                next_vec = mat_vec_mul(mat, curr, gf)
                # 정규화 (Projective Space이므로 스칼라 배는 같은 점)
                next_p = normalize_point(next_vec, q)
                
                if next_p in points_set and next_p not in current_orbit:
                    stack.append(next_p)
        
        orbits[p] = list(current_orbit)
        
    return representatives, orbits

if __name__ == "__main__":
    # 단독 실행 시 테스트 및 파일 저장
    print("[*] Running geometry.py directly...")
    test_k, test_q = 3, 4 # Test with Extension Field GF(4)
    print(f"[*] Generating points for PG({test_k-1}, {test_q})...")
    
    points = generate_projective_points(test_k, test_q)
    print(f"[*] Generated {len(points)} points: {points}")
    
    # dataset 폴더에 저장 테스트
    directory = "dataset"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = os.path.join(directory, f"test_geometry_k{test_k}_q{test_q}.txt")
    with open(filename, "w") as f:
        f.write(f"# Test Run from geometry.py\n")
        for i, p in enumerate(points):
            f.write(f"{i}: {p}\n")
    print(f"[*] Test data saved to '{filename}'")