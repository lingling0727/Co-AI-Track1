import itertools
import ast
import os
import random
import datetime
from collections import Counter

class GaloisMath:
    """
    Minimal Galois Field arithmetic for dataset generation.
    Supports Prime fields and small composite fields (4, 8, 9).
    """
    def __init__(self, q):
        self.q = q
        self.is_prime = self._check_prime(q)
        self.gf8_exp = [1, 2, 4, 3, 6, 7, 5] * 2
        self.gf8_log = {1:0, 2:1, 4:2, 3:3, 6:4, 7:5, 5:6}
        self.gf4_mul = {
            (0,0):0, (0,1):0, (0,2):0, (0,3):0,
            (1,0):0, (1,1):1, (1,2):2, (1,3):3,
            (2,0):0, (2,1):2, (2,2):3, (2,3):1,
            (3,0):0, (3,1):3, (3,2):1, (3,3):2
        }

    def _check_prime(self, n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def mul(self, a, b):
        if a == 0 or b == 0: return 0
        if self.is_prime:
            return (a * b) % self.q
        elif self.q == 4:
            return self.gf4_mul.get((a, b), 0)
        elif self.q == 8:
            return self.gf8_exp[self.gf8_log[a] + self.gf8_log[b]]
        return (a * b) % self.q # Fallback

    def dot(self, v1, v2):
        val = 0
        for a, b in zip(v1, v2):
            term = self.mul(a, b)
            if self.is_prime:
                val = (val + term) % self.q
            else:
                val ^= term # XOR for characteristic 2
        return val

class ProjectivePoint:
    """
    Represents a point in PG(k-1, q).
    Pre-computes incidence with all hyperplanes.
    """
    def __init__(self, id, vector, q, k):
        self.id = id
        self.vector = vector
        self.q = q
        self.k = k
        # incidence_vector[h_idx] = 1 if dot(this, h_vec) == 0 else 0
        self.incidence_vector = []

    def compute_incidence(self, all_points, gf_math):
        # In PG(k-1, q), points and hyperplanes are isomorphic.
        # We test this point against every other point (acting as a hyperplane normal).
        self.incidence_vector = [1 if gf_math.dot(self.vector, h) == 0 else 0 for h in all_points]

def generate_projective_points(k, q):
    """
    Step 1: 기하학적 환경 준비 (Projective Space Data)
    PG(k-1, q)의 모든 점을 생성합니다.
    
    [수학적 원리]
    투영 공간의 점은 벡터 v와 kv (k는 0이 아닌 스칼라)가 동일한 점을 나타냅니다.
    따라서 각 1차원 부분공간에서 유일한 대표원(Canonical Representative)을 뽑아야 합니다.
    
    1. q가 소수(Prime)인 경우: 모듈러 역원을 이용해 정규화 (v * first_nonzero^-1 mod q)
    2. q가 합성수(Composite)인 경우: 갈루아 체(Galois Field) 연산이 필요하므로, 
       여기서는 '첫 번째 0이 아닌 성분이 1'인 벡터를 필터링하는 방식으로 처리합니다.
    """
    
    # 소수 판별 함수 (간단한 구현)
    def is_prime(n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    q_is_prime = is_prime(q)
    points = set()
    
    # 0부터 q-1까지의 숫자로 만들 수 있는 길이 k의 모든 벡터 생성
    for vector in itertools.product(range(q), repeat=k):
        # 영벡터 제외
        if all(x == 0 for x in vector):
            continue
        
        # 첫 번째 0이 아닌 성분 찾기
        first_nonzero_idx = -1
        first_nonzero_val = 0
        for i, val in enumerate(vector):
            if val != 0:
                first_nonzero_idx = i
                first_nonzero_val = val
                break
        
        if q_is_prime:
            # [Case 1: Prime Field] 모듈러 역원을 이용한 수학적 정규화
            # v' = v * (val)^-1 (mod q)
            try:
                inv = pow(first_nonzero_val, -1, q)
                normalized_vector = tuple((v * inv) % q for v in vector)
                points.add(normalized_vector)
            except ValueError:
                # 역원이 없는 경우 (이론상 발생하지 않아야 함)
                continue
        else:
            # [Case 2: Composite Field (e.g., 4, 8)]
            # GF(q) 연산 라이브러리 없이 구현하기 위해, 
            # "첫 성분이 1"인 벡터만 선택하는 방식(Filtering) 사용
            # 이 방식은 모든 1차원 부분공간에서 정확히 하나의 대표원을 선택함을 보장합니다.
            if first_nonzero_val == 1:
                points.add(vector)

    # 정렬하여 일관된 순서 보장
    sorted_vectors = sorted(list(points))
    
    # ProjectivePoint 객체 생성 및 Incidence 계산
    gf = GaloisMath(q)
    projective_objects = []
    
    print(f"[*] Computing Incidence Matrix for {len(sorted_vectors)} points...")
    for idx, vec in enumerate(sorted_vectors):
        p_obj = ProjectivePoint(idx, vec, q, k)
        p_obj.compute_incidence(sorted_vectors, gf)
        projective_objects.append(p_obj)
        
    return projective_objects

def save_dataset(n, k, q):
    """
    데이터셋 파일 생성
    """
    # 현재 실행 중인 스크립트 파일의 위치를 기준으로 dataset 폴더 경로 설정
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(base_path, exist_ok=True)
    
    # 1. Projective Space Data 저장
    p_objects = generate_projective_points(k, q)
    points_filename = os.path.join(base_path, f"projective_space_k{k}_q{q}.txt")

    # metadata
    seed = random.randint(0, 2**31 - 1)
    timestamp = datetime.datetime.now().isoformat()
    expected_count = (q ** k - 1) // (q - 1) if q != 1 else 0

    with open(points_filename, "w", encoding="utf-8") as f:
        f.write(f"# Projective Space PG({k-1}, {q})\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# k: {k}, q: {q}\n")
        f.write(f"# Expected Points: {expected_count}\n")
        f.write(f"# Total Points: {len(p_objects)}\n")
        f.write("ID\tVector\tIncidence_BitString\n")
        for p in p_objects:
            inc_str = "".join(map(str, p.incidence_vector))
            f.write(f"{p.id}\t{p.vector}\t{inc_str}\n")

    print(f"[Generated] {points_filename}")

def load_dataset(n, k, q):
    """
    생성된 데이터셋 파일을 읽어서 메모리로 로드하는 함수 (실험 코드에서 사용)
    Returns:
        points (list): 투영 공간의 점 리스트
        constraints (list): 제약 조건 메타데이터 리스트
    """
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    points_file = os.path.join(base_path, f"projective_space_k{k}_q{q}.txt")
    
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"Dataset for (n={n}, k={k}, q={q}) not found. Run generate_dataset.py first.")

    # 1. Load Points
    points = []
    incidence_matrix = []
    
    with open(points_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or "ID\tVector" in line:
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # 문자열 벡터 "(0, 0, 1)"을 실제 튜플/리스트로 변환
                vec = ast.literal_eval(parts[1])
                points.append(vec)
                if len(parts) >= 3:
                    incidence_matrix.append([int(c) for c in parts[2]])

    print(f"[Loaded] Dataset k={k}, q={q} (Points: {len(points)})")

    # validation
    valid, messages = validate_points(points, k, q)
    if not valid:
        msg = "; ".join(messages)
        raise ValueError(f"Dataset validation failed for k={k}, q={q}: {msg}")

    return points, incidence_matrix


def validate_points(points, k, q):
    """
    Validate a list of projective points:
    - check expected count ((q^k - 1)/(q - 1))
    - check duplicates
    - check coordinate ranges (0..q-1)
    Returns (bool, [messages])
    """
    messages = []
    # expected count
    try:
        expected = (q ** k - 1) // (q - 1) if q != 1 else 0
    except Exception:
        expected = None

    if expected is not None and len(points) != expected:
        messages.append(f"point count mismatch (got {len(points)}, expected {expected})")

    # duplicates
    cnt = Counter(points)
    dups = [p for p, c in cnt.items() if c > 1]
    if dups:
        messages.append(f"duplicates found: {len(dups)} (examples: {dups[:3]})")

    # coordinate ranges
    out_of_range = []
    for p in points:
        for coord in p:
            if not (0 <= coord < q):
                out_of_range.append(p)
                break
    if out_of_range:
        messages.append(f"out-of-range coordinates in {len(out_of_range)} points (examples: {out_of_range[:3]})")

    return (len(messages) == 0, messages)

def main():
    # experiment_parameters.txt 파일을 읽어서 데이터셋 생성
    # 파라미터 파일은 스크립트와 동일한 디렉토리에 있다고 가정
    param_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_parameters.txt")
    
    if not os.path.exists(param_file):
        print(f"Parameter file '{param_file}' not found. Please ensure it exists in the project root.")
        return

    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # 파싱
            parts = line.split(",", 3)
            n = int(parts[0].strip())
            k = int(parts[1].strip())
            q = int(parts[2].strip())
            weights = parts[3].strip().replace("{", "").replace("}", "")
            
            save_dataset(n, k, q)

if __name__ == "__main__":
    main()