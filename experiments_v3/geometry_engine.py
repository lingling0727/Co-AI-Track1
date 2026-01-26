# geometry_engine.py
import numpy as np
import galois
import time
from itertools import product

def generate_geometry_data(n: int, k: int, q: int, weights: list[int]):
    """
    PG(k-1, q)의 기하 구조와 MIP 솔버에 필요한 데이터를 만들어주는 함수임.

    Args:
        n (int): 코드의 전체 길이임.
        k (int): 코드의 차원임.
        q (int): 필드 사이즈임.
        weights (list[int]): 허용 용량을 계산할 가중치 리스트임.

    Returns:
        dict: 솔버에 필요한 모든 데이터가 담긴 딕셔너리임.
    """
    print(f"Generating geometry for PG({k-1}, {q})...")
    start_time = time.time()

    if not all(isinstance(i, int) for i in [n, k, q]) or not isinstance(weights, list):
        raise TypeError("n, k, q는 정수, weights는 리스트여야 함.")

    GF = galois.GF(q)

    # 벡터 공간 V = GF(q)^k의 0이 아닌 모든 벡터들을 생성함
    all_vectors = list(product(GF.elements, repeat=k))
    non_zero_vectors = [v for v in all_vectors if any(e != 0 for e in v)]

    # 1차원 부분 공간(사영 공간의 점)의 대표 원소들을 구함
    points_set = set()
    for vector in non_zero_vectors:
        first_nonzero = next(c for c in vector if c != 0)
        normalized_vector = tuple(c / first_nonzero for c in vector)
        points_set.add(normalized_vector)
    
    # 순서를 일관되게 유지하기 위해 정렬함
    points = np.array([list(p) for p in sorted(list(points_set))], dtype=int)
    points_gf = GF(points)
    num_points = len(points_gf)

    # 쌍대성(duality) 때문에, 초평면도 점과 동일한 벡터로 표현됨
    hyperplanes_gf = points_gf
    num_hypers = num_points

    expected_points = (q**k - 1) // (q - 1)
    if num_points != expected_points:
        print(f"Warning: 생성된 점 개수({num_points})와 예상 개수({expected_points})가 다름.")

    print(f"  - 점 {num_points}개, 초평면 {num_hypers}개 찾음.")

    # 인시던스 행렬 생성: 점 P가 초평면 H 위에 있으면 1 (P . H = 0)
    incidence_matrix = (hyperplanes_gf @ points_gf.T == 0).astype(np.int8)

    # 허용 용량 S_H = n - w 를 계산함
    allowed_capacities = sorted(list(set([n - w for w in weights])))

    # 새 문제니까, 점의 중복도(multiplicity) 최소값은 일단 전부 0임
    lower_bounds = np.zeros(num_points, dtype=np.int32)

    end_time = time.time()
    print(f"기하 데이터 생성 완료. ({end_time - start_time:.4f}초)")

    return {
        "n": n,
        "k": k,
        "q": q,
        "weights": weights,
        "allowed_capacities": allowed_capacities,
        "num_points": num_points,
        "num_hypers": num_hypers,
        "incidence_matrix": incidence_matrix,
        "lower_bounds": lower_bounds,
    }