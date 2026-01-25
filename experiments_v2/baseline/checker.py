import collections
from collections import Counter
def verify_solution(solution_counts, n, target_weights, points, hyperplanes, q):
    """
    ILP 솔버가 찾은 해(multiset)가 실제로 목표 가중치를 갖는지 검증합니다.
    (이론적으로 ILP 모델이 정확하다면 항상 통과해야 합니다.)
    
    solution_counts: {point_idx: count} 딕셔너리
    """
    # 순환 참조 방지를 위해 함수 내에서 import
    from geometry import is_point_in_hyperplane

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
    # 순환 참조 방지를 위해 함수 내에서 import
    from geometry import is_point_in_hyperplane
    
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

def filter_isomorphic_solutions(solutions, n, points, hyperplanes, q):
    """
    가중치 분포를 비교하여 동형일 가능성이 있는 해들을 필터링합니다.
    """
    unique_solutions = {}
    for sol in solutions:
        canonical_key = get_weight_distribution(sol, n, points, hyperplanes, q)
        if canonical_key not in unique_solutions:
            unique_solutions[canonical_key] = sol
            
    return list(unique_solutions.values())