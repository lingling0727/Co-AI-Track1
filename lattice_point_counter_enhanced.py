"""
Linear Code Classification: Enhanced Phase 0 with Upper Bounds

각 x_P에 대한 상한을 계산하여 탐색 공간을 축소 (수정 버전)
"""

from math import comb
import numpy as np
from itertools import product

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
        """
        PG(k-1, q)의 모든 점을 생성
        각 점은 F_q^k의 1차원 부분공간을 대표하는 벡터
        """
        points = []
        # F_q^k의 모든 non-zero 벡터 생성
        for vec in self._generate_vectors():
            # 정규화된 대표원소 찾기 (첫 번째 non-zero 성분을 1로)
            normalized = self._normalize_vector(vec)
            # 이미 추가된 점인지 확인
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
        """벡터를 정규화 (첫 non-zero 성분을 1로)"""
        vec = vec.copy()
        for i in range(len(vec)):
            if vec[i] != 0:
                # F_q에서의 역원 계산 (간단히 modular inverse)
                factor = vec[i]
                # 간단한 경우: 그냥 첫 성분을 1로 만들기 위해 나눔
                vec = vec % self.q
                break
        return vec

    def generate_hyperplanes(self, points):
        """
        각 hyperplane에 포함되는 점들의 인덱스 반환
        """
        hyperplanes = []
        # 각 hyperplane을 정의하는 법벡터 생성
        for normal in self._generate_vectors():
            # 이 hyperplane에 속하는 점들 찾기
            point_indices = []
            for i, point in enumerate(points):
                if np.dot(point, normal) % self.q == 0:
                    point_indices.append(i)

            # 중복 제거
            point_set = frozenset(point_indices)
            if point_set not in [frozenset(h) for h in hyperplanes]:
                if len(point_indices) > 0:  # 빈 hyperplane 제외
                    hyperplanes.append(point_indices)

        return hyperplanes


def count_without_phase0(n, k, q):
    """
    Phase 0 없이 전체 경우의 수 계산
    제약: ∑x_P = n
    """
    pg = ProjectiveSpace(k, q)
    m = pg.num_points()
    # Stars and bars: C(n + m - 1, m - 1)
    count = comb(n + m - 1, m - 1)
    return count, m


def compute_upper_bounds(n, k, q, d):
    """
    각 x_P에 대한 상한(upper bound) 계산

    올바른 방법:
    - 각 점 P가 속한 모든 hyperplane H에 대해
    - ∑(Q∈H) x_Q ≤ n - d
    - 다른 점들의 최소값(0)을 고려하면:
    - x_P ≤ (n - d) - 0 = n - d

    Returns:
        upper_bounds: list of upper bounds for each x_P
        hyperplanes: list of hyperplane memberships
        points: projective space points
        point_to_hyperplanes: mapping from points to hyperplanes
    """
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)

    # 초기 상한: ∑x_P = n 제약에서 각 점은 최대 n개까지 가능
    upper_bounds = [n] * m

    # 각 점이 속한 hyperplane들 기록
    point_to_hyperplanes = [[] for _ in range(m)]
    for h_idx, h_points in enumerate(hyperplanes):
        for p_idx in h_points:
            point_to_hyperplanes[p_idx].append(h_idx)

    # 각 점 P에 대해 상한 계산
    for p_idx in range(m):
        bounds_for_p = [n]  # 기본 상한

        # P가 속한 각 hyperplane H에 대해
        for h_idx in point_to_hyperplanes[p_idx]:
            # 이 hyperplane의 제약: ∑(Q∈H) x_Q ≤ n - d
            # 다른 점들의 최소값 = 0이므로
            # x_P ≤ (n - d) - 0 = n - d
            bound_from_h = n - d
            bounds_for_p.append(bound_from_h)

        # 모든 제약 중 가장 타이트한 것 선택
        upper_bounds[p_idx] = min(bounds_for_p)

    return upper_bounds, hyperplanes, points, point_to_hyperplanes


def count_with_basic_phase0(n, k, q, d):
    """
    기본 Phase 0: hyperplane 제약만 사용
    """
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)

    count = 0

    # 모든 가능한 조합 생성
    def enumerate_solutions(remaining, pos, current):
        nonlocal count

        if pos == m:
            if remaining == 0:
                # Hyperplane 제약 확인
                if check_hyperplane_constraints(current, hyperplanes, n, d):
                    count += 1
            return

        # x_pos 값 시도 (0부터 remaining까지)
        for val in range(remaining + 1):
            current[pos] = val
            enumerate_solutions(remaining - val, pos + 1, current)

    current = [0] * m
    enumerate_solutions(n, 0, current)

    return count


def count_with_enhanced_phase0(n, k, q, d):
    """
    향상된 Phase 0: 각 x_P의 상한을 계산하여 탐색 공간 축소
    """
    upper_bounds, hyperplanes, points, point_to_hyperplanes = compute_upper_bounds(n, k, q, d)
    m = len(points)

    print(f"   - 상한 계산 완료:")
    print(f"     평균 상한: {sum(upper_bounds) / len(upper_bounds):.2f}")
    print(f"     최소 상한: {min(upper_bounds)}")
    print(f"     최대 상한: {max(upper_bounds)}")

    count = 0

    # 모든 가능한 조합 생성 (상한 적용)
    def enumerate_solutions(remaining, pos, current):
        nonlocal count

        if pos == m:
            if remaining == 0:
                # Hyperplane 제약 확인
                if check_hyperplane_constraints(current, hyperplanes, n, d):
                    count += 1
            return

        # x_pos 값 시도 (0부터 min(remaining, upper_bound)까지)
        max_val = min(remaining, upper_bounds[pos])
        for val in range(max_val + 1):
            current[pos] = val
            enumerate_solutions(remaining - val, pos + 1, current)

    current = [0] * m
    enumerate_solutions(n, 0, current)

    return count


def check_hyperplane_constraints(solution, hyperplanes, n, d):
    """
    주어진 solution이 모든 hyperplane 제약을 만족하는지 확인
    ∑(P∈H) x_P ≤ n - d for all H
    """
    for h_points in hyperplanes:
        h_sum = sum(solution[p] for p in h_points)
        if h_sum > n - d:
            return False
    return True


def compare_three_methods(n, k, q, d):
    """
    세 가지 방법 비교:
    1. Phase 0 없이 (이론적 개수)
    2. 기본 Phase 0 (hyperplane 제약만)
    3. 향상된 Phase 0 (상한 계산 적용)
    """
    print("="*70)
    print(f"Linear Code Parameters: [n={n}, k={k}, d={d}]_{q}")
    print("="*70)

    # 1. Phase 0 없이
    count_without, m = count_without_phase0(n, k, q)
    print(f"\n[방법 1: Phase 0 없이]")
    print(f"   - PG({k-1}, {q})의 점 개수: {m}")
    print(f"   - 이론적 전체 경우의 수: {count_without:,}")

    # 2. 기본 Phase 0
    print(f"\n[방법 2: 기본 Phase 0 (hyperplane 제약)]")
    print(f"   - 계산 중...")
    count_basic = count_with_basic_phase0(n, k, q, d)
    print(f"   - 격자점 개수: {count_basic:,}")
    reduction1 = count_without - count_basic
    rate1 = (reduction1 / count_without * 100) if count_without > 0 else 0
    print(f"   - 감소량: {reduction1:,} ({rate1:.2f}%)")

    # 3. 향상된 Phase 0
    print(f"\n[방법 3: 향상된 Phase 0 (상한 계산 적용)]")
    print(f"   - 계산 중...")
    count_enhanced = count_with_enhanced_phase0(n, k, q, d)
    print(f"   - 격자점 개수: {count_enhanced:,}")
    reduction2 = count_without - count_enhanced
    rate2 = (reduction2 / count_without * 100) if count_without > 0 else 0
    print(f"   - 감소량: {reduction2:,} ({rate2:.2f}%)")

    # 비교
    print(f"\n[비교 결과]")
    print(f"   - 이론적 전체: {count_without:,}")
    print(f"   - 기본 Phase 0: {count_basic:,} (감소율 {rate1:.2f}%)")
    print(f"   - 향상된 Phase 0: {count_enhanced:,} (감소율 {rate2:.2f}%)")

    # 검증: 두 방법이 같은 결과를 내는지 확인
    if count_basic == count_enhanced:
        print(f"   ✅ 검증 성공: 두 방법 모두 동일한 결과")
    else:
        print(f"   ⚠️ 불일치: 기본={count_basic}, 향상={count_enhanced}")

    additional_reduction = count_basic - count_enhanced
    print(f"   - 추가 감소: {additional_reduction:,}")
    print("="*70)

    return {
        'without': count_without,
        'basic': count_basic,
        'enhanced': count_enhanced,
        'num_points': m
    }


# 테스트
if __name__ == "__main__":
    # 작은 예시로 테스트
    print("\n테스트 1: 작은 파라미터")
    compare_three_methods(n=7, k=3, q=2, d=3)

    print("\n\n테스트 2: 조금 더 큰 파라미터")
    compare_three_methods(n=10, k=3, q=2, d=4)
