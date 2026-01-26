import collections
from collections import Counter

def verify_solution(solution_counts, n, target_weights, points, hyperplanes, q):
    from geometry import is_point_in_hyperplane

    for h in hyperplanes:
        intersection_size = 0
        for p_idx, count in solution_counts.items():
            point_vector = points[p_idx]
            if is_point_in_hyperplane(point_vector, h, q):
                intersection_size += count
        
        weight = n - intersection_size
        if weight != 0 and weight not in target_weights:
            return False
            
    return True

def get_weight_distribution(solution_counts, n, points, hyperplanes, q):
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
            
    return tuple(sorted(Counter(weights).items()))

def filter_isomorphic_solutions(solutions, n, points, hyperplanes, q):
    unique_solutions = {}
    for sol in solutions:
        canonical_key = get_weight_distribution(sol, n, points, hyperplanes, q)
        if canonical_key not in unique_solutions:
            unique_solutions[canonical_key] = sol
            
    return list(unique_solutions.values())