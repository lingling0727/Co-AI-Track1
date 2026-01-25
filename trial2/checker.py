import collections
from collections import Counter
from geometry import is_point_in_hyperplane, generate_linear_group, mat_vec_mul, normalize_point, get_gf

def verify_solution(solution_counts, n, target_weights, points, hyperplanes, q):
    """
    ILP 솔버가 찾은 해(multiset)가 실제로 목표 가중치를 갖는지 검증합니다.
    (이론적으로 ILP 모델이 정확하다면 항상 통과해야 합니다.)
    
    solution_counts: {point_idx: count} 딕셔너리
    """

    for h in hyperplanes:
        intersection_size = 0
        for p_idx, count in solution_counts.items():
            point_vector = points[p_idx]
            if is_point_in_hyperplane(point_vector, h, q):
                intersection_size += count
        
        weight = n - intersection_size
        if weight != 0 and weight not in target_weights:
            # 0이 아닌 가중치가 목표 집합에 없는 경우
            print(f"  > Verification FAILED for hyperplane {h}. Weight: {weight}")
            return False
            
    return True

def get_weight_distribution(solution_counts, n, points, hyperplanes, q):
    """
    주어진 해(solution)에 해당하는 부호의 가중치 분포를 계산합니다.
    가중치 분포는 (가중치, 해당 가중치를 갖는 부호어의 개수)의 집합으로,
    강력한 동형 불변량(isomorphism invariant)으로 사용됩니다.
    """
    
    weights = []
    for h in hyperplanes:
        intersection_size = 0
        for p_idx, count in solution_counts.items():
            point_vector = points[p_idx]
            if is_point_in_hyperplane(point_vector, h, q):
                intersection_size += count
        
        weight = n - intersection_size
        if weight != 0:
            weights.append(weight)
            
    # 각 가중치의 개수를 세어 정규형(canonical form)으로 반환
    # 예: [3, 4, 3, 4, 4] -> ((3, 2), (4, 3))
    return tuple(sorted(Counter(weights).items()))

def are_codes_isomorphic(sol1, sol2, points, point_map, group, q):
    """
    두 해(sol1, sol2)가 기하학적으로 동형인지(선형 변환으로 겹쳐지는지) 확인합니다.
    논문의 Phase 2에서 수행하는 엄밀한 동치 판단입니다.
    """
    # 1. 점의 개수(Multiplicity) 구성이 다르면 바로 False
    if sorted(sol1.values()) != sorted(sol2.values()):
        return False

    gf = get_gf(q)
    
    # 그룹 내의 모든 행렬 A에 대해 테스트
    for mat in group:
        # sol1의 모든 점을 A로 변환했을 때 sol2와 일치하는지 확인
        is_match = True
        
        # sol1의 점들을 변환하여 sol2에 있는지 확인
        # (최적화: sol1의 점 중 하나라도 sol2에 없거나 개수가 다르면 즉시 중단)
        for p_idx, count in sol1.items():
            p_vec = points[p_idx]
            
            # 변환: P' = A * P
            trans_vec = mat_vec_mul(mat, p_vec, gf)
            norm_vec = normalize_point(trans_vec, q)
            
            # 변환된 점이 sol2에 같은 개수만큼 있어야 함
            p_prime_idx = point_map.get(norm_vec)
            
            if p_prime_idx is None or sol2.get(p_prime_idx, 0) != count:
                is_match = False
                break
        
        if is_match:
            return True # 변환 A를 찾음 -> 동형임
            
    return False

def filter_isomorphic_solutions(solutions, n, points, hyperplanes, q):
    """
    1차로 가중치 분포를 비교하고, 2차로 기하학적 변환을 통해 엄밀하게 동형을 필터링합니다.
    """
    if not solutions: return []
    
    # 0. 준비: 벡터 -> 인덱스 맵 및 대칭성 그룹 생성
    point_map = {p: i for i, p in enumerate(points)}
    k = len(points[0])
    # 주의: k, q가 크면 전체 그룹 생성이 불가능하므로 부분군만 사용됨 (이 경우 완벽한 분류는 아님)
    group = generate_linear_group(k, q)
    
    # 1. 가중치 분포로 그룹화 (빠른 필터링)
    grouped_solutions = collections.defaultdict(list)
    for sol in solutions:
        wd = get_weight_distribution(sol, n, points, hyperplanes, q)
        grouped_solutions[wd].append(sol)
    
    final_unique_solutions = []
    
    # 2. 각 그룹 내에서 엄밀한 동형성 검사 (Brute-force Check)
    for wd, candidates in grouped_solutions.items():
        unique_in_group = []
        for cand in candidates:
            # 기존에 찾은 유니크한 해들과 비교
            if not any(are_codes_isomorphic(cand, existing, points, point_map, group, q) for existing in unique_in_group):
                unique_in_group.append(cand)
        final_unique_solutions.extend(unique_in_group)
            
    return final_unique_solutions