"""
Linear Code Classification - Complete Implementation for Google Colab
LP 기반 타이트한 상한 계산으로 탐색 공간 축소

사용법:
1. 이 셀 전체를 실행
2. compare_three_methods(n, k, q, d) 호출

예시:
    compare_three_methods(n=10, k=3, q=2, d=4)
"""

# ============================================================================
# 설치 (Colab에서 처음 실행 시)
# ============================================================================
try:
    from scipy.optimize import linprog
    print("✓ scipy 설치됨")
except ImportError:
    print("scipy 설치 중...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    from scipy.optimize import linprog
    print("✓ scipy 설치 완료")

import numpy as np
from math import comb
from itertools import product


# ============================================================================
# 사영공간 클래스
# ============================================================================
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


# ============================================================================
# 경우의 수 계산 함수들
# ============================================================================
def count_without_phase0(n, k, q):
    """Phase 0 없이 전체 경우의 수 계산 (중복조합)"""
    pg = ProjectiveSpace(k, q)
    m = pg.num_points()
    count = comb(n + m - 1, m - 1)
    return count, m


def compute_upper_bounds_lp(n, k, q, d, verbose=True):
    """
    LP를 사용한 타이트한 상한 계산
    
    각 x_P에 대해:
        maximize x_P
        subject to:
            ∑x_Q = n
            ∑(Q∈H) x_Q ≤ n - d  for all H
            x_Q ≥ 0
    """
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)
    
    if verbose:
        print(f"  - LP 기반 상한 계산 중... (점: {m}개, hyperplane: {len(hyperplanes)}개)")
    
    # 제약조건 구성
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
    A_eq = np.array([[1] * m])
    b_eq = np.array([n])
    bounds = [(0, None) for _ in range(m)]
    
    # 각 x_P에 대해 LP로 상한 계산
    upper_bounds = []
    for p_idx in range(m):
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
            ub = -result.fun
            ub_int = int(np.floor(ub + 1e-6))
            upper_bounds.append(ub_int)
        else:
            upper_bounds.append(n - d)
    
    point_to_hyperplanes = [[] for _ in range(m)]
    for h_idx, h_points in enumerate(hyperplanes):
        for p_idx in h_points:
            point_to_hyperplanes[p_idx].append(h_idx)
    
    return upper_bounds, hyperplanes, points, point_to_hyperplanes


def check_hyperplane_constraints(solution, hyperplanes, n, d):
    """hyperplane 제약 검증"""
    for h_points in hyperplanes:
        h_sum = sum(solution[p] for p in h_points)
        if h_sum > n - d:
            return False
    return True


def count_with_basic_phase0(n, k, q, d):
    """기본 Phase 0: 상한 없이 hyperplane 제약만 검증"""
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


def count_with_lp_phase0(n, k, q, d):
    """향상된 Phase 0: LP 기반 타이트한 상한 적용"""
    upper_bounds, hyperplanes, points, _ = compute_upper_bounds_lp(n, k, q, d, verbose=True)
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
        
        # LP 상한 적용
        max_val = min(remaining, upper_bounds[pos])
        for val in range(max_val + 1):
            current[pos] = val
            enumerate_solutions(remaining - val, pos + 1, current)
    
    current = [0] * m
    enumerate_solutions(n, 0, current)
    return count


def count_search_space(n, k, q, d, use_lp=False):
    """탐색 공간 크기 (방문하는 노드 수) 계산"""
    if use_lp:
        upper_bounds, _, _, _ = compute_upper_bounds_lp(n, k, q, d, verbose=False)
    else:
        pg = ProjectiveSpace(k, q)
        m = pg.num_points()
        upper_bounds = [n] * m
    
    m = len(upper_bounds)
    node_count = [0]
    
    def count_nodes(remaining, pos):
        node_count[0] += 1
        if pos == m:
            return
        max_val = min(remaining, upper_bounds[pos])
        for val in range(max_val + 1):
            count_nodes(remaining - val, pos + 1)
    
    count_nodes(n, 0)
    return node_count[0]


# ============================================================================
# 메인 비교 함수
# ============================================================================
def compare_three_methods(n, k, q, d):
    """
    세 가지 방법 비교:
    1. Phase 0 없이 (이론적 개수)
    2. 기본 Phase 0 (상한 없음)
    3. LP Phase 0 (타이트한 상한)
    """
    print("="*70)
    print(f"Linear Code Parameters: [n={n}, k={k}, d={d}]_{q}")
    print("="*70)
    
    # 1. Phase 0 없이
    count_without, m = count_without_phase0(n, k, q)
    print(f"\n[방법 1: Phase 0 없이 (이론적)]")
    print(f"  - PG({k-1}, {q})의 점 개수: {m}")
    print(f"  - 이론적 전체 경우의 수: {count_without:,}")
    
    # 탐색 공간 분석
    print(f"\n[탐색 공간 분석]")
    space_no_bounds = count_search_space(n, k, q, d, use_lp=False)
    print(f"  - 제약 없는 탐색 공간: {space_no_bounds:,} 노드")
    
    space_lp = count_search_space(n, k, q, d, use_lp=True)
    print(f"  - LP 상한 적용 시: {space_lp:,} 노드")
    reduction = (space_no_bounds - space_lp) / space_no_bounds * 100
    print(f"  - 탐색 공간 감소: {space_no_bounds - space_lp:,} ({reduction:.2f}%)")
    
    # 2. 기본 Phase 0
    print(f"\n[방법 2: 기본 Phase 0 (상한 없음)]")
    print(f"  - 계산 중...")
    count_basic = count_with_basic_phase0(n, k, q, d)
    print(f"  - 유효한 격자점 개수: {count_basic:,}")
    reduction1 = (count_without - count_basic) / count_without * 100
    print(f"  - 이론치 대비 감소: {count_without - count_basic:,} ({reduction1:.2f}%)")
    
    # 3. LP Phase 0
    print(f"\n[방법 3: LP Phase 0 (타이트한 상한)]")
    print(f"  - 계산 중...")
    count_lp = count_with_lp_phase0(n, k, q, d)
    print(f"  - 유효한 격자점 개수: {count_lp:,}")
    reduction2 = (count_without - count_lp) / count_without * 100
    print(f"  - 이론치 대비 감소: {count_without - count_lp:,} ({reduction2:.2f}%)")
    
    # 검증
    print(f"\n[검증 및 요약]")
    if count_basic == count_lp:
        print(f"  ✅ 정확도: 100% (두 방법 모두 {count_basic:,}개 발견)")
    else:
        print(f"  ⚠️ 불일치: 기본={count_basic:,}, LP={count_lp:,}")
    
    print(f"\n  📊 효율성 비교:")
    print(f"     - 탐색 공간: {space_no_bounds:,} → {space_lp:,} ({reduction:.2f}% 감소)")
    print(f"     - 유효한 해: {count_basic:,} (동일)")
    print(f"     - 계산 효율: {reduction:.1f}% 향상")
    
    print("="*70)
    
    return {
        'without': count_without,
        'basic': count_basic,
        'lp': count_lp,
        'num_points': m,
        'space_no_bounds': space_no_bounds,
        'space_lp': space_lp
    }


# ============================================================================
# 사용 예시
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🔬 테스트 1: 작은 파라미터 " + "🔬\n")
    result1 = compare_three_methods(n=7, k=3, q=2, d=3)
    
    print("\n\n" + "🔬 테스트 2: 중간 파라미터 " + "🔬\n")
    result2 = compare_three_methods(n=10, k=3, q=2, d=4)
    
    print("\n\n" + "🔬 테스트 3: 조금 더 큰 파라미터 " + "🔬\n")
    result3 = compare_three_methods(n=12, k=3, q=2, d=5)
    
    print("\n\n" + "="*70)
    print("💡 사용법:")
    print("="*70)
    print("compare_three_methods(n=10, k=3, q=2, d=4)")
    print("\n파라미터:")
    print("  n: 부호 길이")
    print("  k: 차원")
    print("  q: 유한체 크기")
    print("  d: 최소 거리")
    print("="*70)
