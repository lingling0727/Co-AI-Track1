"""
Linear Code Classification: Enhanced Phase 0 with LP-based Upper Bounds

LP를 사용하여 각 x_P의 타이트한 상한을 계산
"""

from math import comb
import numpy as np
from itertools import product

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Install with: pip install scipy")


class ProjectiveSpace:
    """사영공간 PG(k-1, q) 관련 계산"""

    def __init__(self, k, q):
        self.k = k
        self.q = q
        self.dimension = k - 1

    def num_points(self):
        """PG(k-1, q)의 점 개수"""
        return (self.q**self.k - 1) // (self.q - 1)

    def generate_points(self):
        """PG(k-1, q)의 모든 점을 생성"""
        points = []
        for vec in self._generate_vectors():
            normalized = self._normalize_vector(vec)
            is_duplicate = False
            for p in points:
                if np.array_equal(p, normalized):
                    is_duplicate = True
                    break
            if not is_duplicate:
                points.append(normalized)
        return np.array(points)

    def _generate_vectors(self):
        """F_q^k의 모든 non-zero 벡터 생성"""
        for vec in product(range(self.q), repeat=self.k):
            if any(v != 0 for v in vec):
                yield np.array(vec)

    def _normalize_vector(self, vec):
        """벡터를 정규화"""
        vec = vec.copy()
        for i in range(len(vec)):
            if vec[i] != 0:
                vec = vec % self.q
                break
        return vec

    def generate_hyperplanes(self, points):
        """각 hyperplane에 포함되는 점들의 인덱스 반환"""
        hyperplanes = []
        for normal in self._generate_vectors():
            point_indices = []
            for i, point in enumerate(points):
                if np.dot(point, normal) % self.q == 0:
                    point_indices.append(i)
            point_set = frozenset(point_indices)
            if point_set not in [frozenset(h) for h in hyperplanes]:
                if len(point_indices) > 0:
                    hyperplanes.append(point_indices)
        return hyperplanes


def count_without_phase0(n, k, q):
    """Phase 0 없이 전체 경우의 수 계산"""
    pg = ProjectiveSpace(k, q)
    m = pg.num_points()
    count = comb(n + m - 1, m - 1)
    return count, m


def compute_upper_bounds_naive(n, k, q, d):
    """
    단순 상한 계산 (기존 방식)
    각 점에 대해 x_P ≤ n - d
    """
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)
    
    # 모든 점에 대해 동일한 상한
    upper_bounds = [n - d] * m
    
    point_to_hyperplanes = [[] for _ in range(m)]
    for h_idx, h_points in enumerate(hyperplanes):
        for p_idx in h_points:
            point_to_hyperplanes[p_idx].append(h_idx)
    
    return upper_bounds, hyperplanes, points, point_to_hyperplanes


def compute_upper_bounds_lp(n, k, q, d):
    """
    LP를 사용한 타이트한 상한 계산
    
    각 x_P에 대해 다음 LP를 풀어서 상한 계산:
    
    maximize: x_P
    subject to:
        ∑x_Q = n                    (equality)
        ∑(Q∈H) x_Q ≤ n - d  for all H  (inequalities)
        x_Q ≥ 0              for all Q  (non-negativity)
    
    Returns:
        upper_bounds: LP로 계산된 각 x_P의 상한
        hyperplanes: hyperplane 정보
        points: 사영공간의 점들
        point_to_hyperplanes: 매핑
    """
    if not SCIPY_AVAILABLE:
        print("  ⚠️ scipy 없음 - 단순 상한 사용")
        return compute_upper_bounds_naive(n, k, q, d)
    
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)
    
    point_to_hyperplanes = [[] for _ in range(m)]
    for h_idx, h_points in enumerate(hyperplanes):
        for p_idx in h_points:
            point_to_hyperplanes[p_idx].append(h_idx)
    
    print(f"  - LP 기반 상한 계산 중... (점: {m}개, hyperplane: {len(hyperplanes)}개)")
    
    # 제약조건 행렬 구성
    # Inequality constraints: A_ub @ x <= b_ub
    # ∑(Q∈H) x_Q ≤ n - d for all H
    A_ub = []
    b_ub = []
    for h_points in hyperplanes:
        row = [0] * m
        for p_idx in h_points:
            row[p_idx] = 1
        A_ub.append(row)
        b_ub.append(n - d)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Equality constraint: ∑x_Q = n
    # A_eq @ x = b_eq
    A_eq = np.array([[1] * m])
    b_eq = np.array([n])
    
    # Bounds: x_Q ≥ 0
    bounds = [(0, None) for _ in range(m)]
    
    # 각 x_P에 대해 LP로 상한 계산
    upper_bounds = []
    for p_idx in range(m):
        # Objective: maximize x_P = minimize -x_P
        c = [0] * m
        c[p_idx] = -1  # maximize x_P
        
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            # LP optimal value는 -x_P이므로 부호 반전
            ub = -result.fun
            # 정수로 내림 (LP는 실수 해를 줌)
            ub_int = int(np.floor(ub + 1e-6))  # 부동소수점 오차 고려
            upper_bounds.append(ub_int)
        else:
            # LP 실패 시 기본 상한 사용
            upper_bounds.append(n - d)
    
    return upper_bounds, hyperplanes, points, point_to_hyperplanes


def count_with_basic_phase0(n, k, q, d):
    """기본 Phase 0: hyperplane 제약만 사용"""
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)
    
    count = 0
    
    def enumerate_solutions(remaining, pos, current):
        nonlocal count
        if pos == m:
            if remaining == 0:
                if check_hyperplane_constraints(current, hyperplanes, n, d):
                    count += 1
            return
        for val in range(remaining + 1):
            current[pos] = val
            enumerate_solutions(remaining - val, pos + 1, current)
    
    current = [0] * m
    enumerate_solutions(n, 0, current)
    return count


def count_with_enhanced_phase0(n, k, q, d, use_lp=True):
    """
    향상된 Phase 0: LP 기반 상한 적용
    
    Args:
        use_lp: True이면 LP 사용, False면 단순 상한 사용
    """
    if use_lp:
        upper_bounds, hyperplanes, points, point_to_hyperplanes = compute_upper_bounds_lp(n, k, q, d)
    else:
        upper_bounds, hyperplanes, points, point_to_hyperplanes = compute_upper_bounds_naive(n, k, q, d)
    
    m = len(points)
    
    print(f"  - 상한 계산 완료:")
    print(f"    평균 상한: {sum(upper_bounds) / len(upper_bounds):.2f}")
    print(f"    최소 상한: {min(upper_bounds)}")
    print(f"    최대 상한: {max(upper_bounds)}")
    print(f"    상한 분포: {sorted(set(upper_bounds))}")
    
    count = 0
    
    def enumerate_solutions(remaining, pos, current):
        nonlocal count
        if pos == m:
            if remaining == 0:
                if check_hyperplane_constraints(current, hyperplanes, n, d):
                    count += 1
            return
        
        # 상한 적용
        max_val = min(remaining, upper_bounds[pos])
        for val in range(max_val + 1):
            current[pos] = val
            enumerate_solutions(remaining - val, pos + 1, current)
    
    current = [0] * m
    enumerate_solutions(n, 0, current)
    return count


def check_hyperplane_constraints(solution, hyperplanes, n, d):
    """hyperplane 제약 검증"""
    for h_points in hyperplanes:
        h_sum = sum(solution[p] for p in h_points)
        if h_sum > n - d:
            return False
    return True


def compare_three_methods(n, k, q, d):
    """
    세 가지 방법 비교:
    1. Phase 0 없이
    2. 기본 Phase 0 (상한 없음)
    3. 향상된 Phase 0 - 단순 상한 (x_P ≤ n-d)
    4. 향상된 Phase 0 - LP 상한 (타이트)
    """
    print("="*70)
    print(f"Linear Code Parameters: [n={n}, k={k}, d={d}]_{q}")
    print("="*70)
    
    # 1. Phase 0 없이
    count_without, m = count_without_phase0(n, k, q)
    print(f"\n[방법 1: Phase 0 없이]")
    print(f"  - PG({k-1}, {q})의 점 개수: {m}")
    print(f"  - 이론적 전체 경우의 수: {count_without:,}")
    
    # 2. 기본 Phase 0
    print(f"\n[방법 2: 기본 Phase 0 (상한 없음)]")
    print(f"  - 계산 중...")
    count_basic = count_with_basic_phase0(n, k, q, d)
    print(f"  - 격자점 개수: {count_basic:,}")
    reduction1 = count_without - count_basic
    rate1 = (reduction1 / count_without * 100) if count_without > 0 else 0
    print(f"  - 감소량: {reduction1:,} ({rate1:.2f}%)")
    
    # 3. 향상된 Phase 0 - 단순 상한
    print(f"\n[방법 3: 향상된 Phase 0 - 단순 상한 (x_P ≤ n-d)]")
    print(f"  - 계산 중...")
    count_enhanced_naive = count_with_enhanced_phase0(n, k, q, d, use_lp=False)
    print(f"  - 격자점 개수: {count_enhanced_naive:,}")
    reduction2 = count_without - count_enhanced_naive
    rate2 = (reduction2 / count_without * 100) if count_without > 0 else 0
    print(f"  - 감소량: {reduction2:,} ({rate2:.2f}%)")
    
    # 4. 향상된 Phase 0 - LP 상한
    print(f"\n[방법 4: 향상된 Phase 0 - LP 기반 타이트 상한]")
    print(f"  - 계산 중...")
    count_enhanced_lp = count_with_enhanced_phase0(n, k, q, d, use_lp=True)
    print(f"  - 격자점 개수: {count_enhanced_lp:,}")
    reduction3 = count_without - count_enhanced_lp
    rate3 = (reduction3 / count_without * 100) if count_without > 0 else 0
    print(f"  - 감소량: {reduction3:,} ({rate3:.2f}%)")
    
    # 비교
    print(f"\n[비교 결과]")
    print(f"  - 이론적 전체: {count_without:,}")
    print(f"  - 기본 Phase 0: {count_basic:,} (감소율 {rate1:.2f}%)")
    print(f"  - 단순 상한: {count_enhanced_naive:,} (감소율 {rate2:.2f}%)")
    print(f"  - LP 상한: {count_enhanced_lp:,} (감소율 {rate3:.2f}%)")
    
    # 검증
    if count_basic == count_enhanced_naive == count_enhanced_lp:
        print(f"  ✅ 검증 성공: 모든 방법이 동일한 결과")
    else:
        print(f"  ⚠️ 결과 비교:")
        print(f"     기본={count_basic}, 단순={count_enhanced_naive}, LP={count_enhanced_lp}")
    
    # 추가 감소량
    additional_naive = count_basic - count_enhanced_naive
    additional_lp = count_basic - count_enhanced_lp
    
    print(f"\n[효과 분석]")
    print(f"  - 단순 상한의 추가 감소: {additional_naive:,}")
    print(f"  - LP 상한의 추가 감소: {additional_lp:,}")
    
    if count_basic > 0:
        print(f"  - 단순 상한 효과: {(additional_naive/count_basic*100):.2f}%")
        print(f"  - LP 상한 효과: {(additional_lp/count_basic*100):.2f}%")
    
    print("="*70)
    
    return {
        'without': count_without,
        'basic': count_basic,
        'naive': count_enhanced_naive,
        'lp': count_enhanced_lp,
        'num_points': m
    }


# 테스트
if __name__ == "__main__":
    print("\n테스트 1: 작은 파라미터")
    compare_three_methods(n=7, k=3, q=2, d=3)
    
    print("\n\n테스트 2: 조금 더 큰 파라미터")
    compare_three_methods(n=10, k=3, q=2, d=4)
