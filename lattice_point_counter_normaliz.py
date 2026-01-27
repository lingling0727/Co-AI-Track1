"""
Linear Code Classification: Lattice Point Enumeration
Phase 0 적용 전후의 격자점 개수 비교

Using PyNormaliz for lattice point counting (신버전 API)
"""

from math import comb
from itertools import combinations
import numpy as np

try:
    import PyNormaliz as pn
    NORMALIZ_AVAILABLE = True
except ImportError:
    NORMALIZ_AVAILABLE = False
    print("Warning: PyNormaliz not installed. Install with: pip install PyNormaliz")


class ProjectiveSpace:
    """사영공간 PG(k-1, q) 관련 계산"""

    def __init__(self, k, q):
        self.k = k
        self.q = q
        self.dimension = k - 1

    def num_points(self):
        """PG(k-1, q)의 점 개수"""
        return (self.q**self.k - 1) // (self.q - 1)

    def num_hyperplanes(self):
        """PG(k-1, q)의 hyperplane 개수"""
        return self.num_points()  # Dual space

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
        import itertools
        for vec in itertools.product(range(self.q), repeat=self.k):
            if any(v != 0 for v in vec):
                yield np.array(vec)

    def _normalize_vector(self, vec):
        """벡터를 정규화 (첫 non-zero 성분을 1로)"""
        vec = vec.copy()
        for i in range(len(vec)):
            if vec[i] != 0:
                # F_q에서의 역원 (간단히 1로 나눔)
                factor = vec[i]
                vec = (vec // factor) % self.q
                break
        return vec

    def generate_hyperplanes(self, points):
        """
        각 hyperplane에 포함되는 점들의 인덱스 반환
        Hyperplane H = {P : <P, normal> = 0} in dual space
        """
        hyperplanes = []
        n_points = len(points)

        # 각 hyperplane을 정의하는 법벡터 생성
        for normal in self._generate_vectors():
            normal_normalized = self._normalize_vector(normal)

            # 이 hyperplane에 속하는 점들 찾기
            point_indices = []
            for i, point in enumerate(points):
                if np.dot(point, normal_normalized) % self.q == 0:
                    point_indices.append(i)

            # 중복 제거
            point_indices_set = frozenset(point_indices)
            if point_indices_set not in [frozenset(h) for h in hyperplanes]:
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


def generate_normaliz_constraints(n, k, q, d):
    """
    Normaliz를 위한 제약조건 생성

    제약조건:
    1. ∑x_P = n (equality)
    2. ∑(P∈H) x_P ≤ n - d for all hyperplanes H (inequality)
    3. x_P ≥ 0 for all P (non-negativity)

    Returns:
        inequalities: list of inequalities (Ax ≥ b 형태)
        equations: list of equations (Ax = b 형태)
        num_points: 점 개수
        num_hyperplanes: hyperplane 개수
    """
    pg = ProjectiveSpace(k, q)
    points = pg.generate_points()
    hyperplanes = pg.generate_hyperplanes(points)
    m = len(points)

    print(f"Points: {m}, Hyperplanes: {len(hyperplanes)}")

    # Normaliz 형식:
    # inequalities: [b, -A] for b + Ax ≥ 0, i.e., Ax ≥ -b
    # equations: [b, -A] for b + Ax = 0, i.e., Ax = -b

    inequalities = []
    equations = []

    # 1. Equation: ∑x_P = n
    # n - ∑x_P = 0 => ∑x_P = n
    eq = [n] + [-1] * m
    equations.append(eq)

    # 2. Inequalities: ∑(P∈H) x_P ≤ n - d for all H
    # (n - d) - ∑(P∈H) x_P ≥ 0
    for hyperplane in hyperplanes:
        ineq = [n - d] + [0] * m
        for p_idx in hyperplane:
            ineq[p_idx + 1] = -1
        inequalities.append(ineq)

    # 3. Non-negativity: x_P ≥ 0
    # 0 + 1·x_P ≥ 0
    for i in range(m):
        ineq = [0] + [0] * m
        ineq[i + 1] = 1
        inequalities.append(ineq)

    return inequalities, equations, m, len(hyperplanes)


def count_with_normaliz(inequalities, equations):
    """
    PyNormaliz를 사용하여 격자점 개수 계산

    Args:
        inequalities: list of inequality constraints [b, -A] for Ax ≥ -b
        equations: list of equation constraints [b, -A] for Ax = -b

    Returns:
        격자점 개수 (정수) 또는 None (실패 시)
    """
    if not NORMALIZ_AVAILABLE:
        print("Error: PyNormaliz가 설치되어 있지 않습니다.")
        print("설치: pip install PyNormaliz")
        return None

    try:
        # Normaliz cone 생성
        cone_input = {}
        if inequalities:
            cone_input["inequalities"] = inequalities
        if equations:
            cone_input["equations"] = equations

        # ✅ 수정: NmzCone → Cone
        cone = pn.Cone(**cone_input)

        # ✅ 수정: Compute 메서드 사용
        cone.Compute("HilbertBasis")

        # Hilbert basis 가져오기
        hilbert_basis = cone.HilbertBasis()

        if hilbert_basis is not None:
            return len(hilbert_basis)

        # 다른 방법: 직접 열거 방식으로 폴백
        return count_lattice_points_direct(inequalities, equations)

    except Exception as e:
        print(f"Normaliz Error: {e}")
        # 실패 시 직접 열거 시도
        return count_lattice_points_direct(inequalities, equations)


def count_lattice_points_direct(inequalities, equations):
    """
    제약조건을 만족하는 격자점을 직접 열거하여 개수 계산
    (작은 문제에만 적합)

    Args:
        inequalities: [[b, -A]] for Ax ≥ -b
        equations: [[b, -A]] for Ax = -b

    Returns:
        격자점 개수
    """
    if not equations:
        print("Error: equation이 필요합니다 (∑x_P = n)")
        return None

    # equation에서 n 값 추출
    # [n, -1, -1, ..., -1] 형태
    eq = equations[0]
    n = eq[0]
    m = len(eq) - 1  # 변수 개수

    print(f"  - 직접 열거 방식 사용 (n={n}, m={m})")
    print(f"  - 경고: 큰 문제는 시간이 오래 걸릴 수 있습니다")

    count = 0

    # 모든 가능한 조합 생성 (stars and bars)
    def generate_partitions(remaining, num_vars, current):
        """재귀적으로 분할 생성"""
        nonlocal count

        if num_vars == 0:
            if remaining == 0:
                # 모든 inequality 체크
                if check_inequalities(current, inequalities):
                    count += 1
            return

        if num_vars == 1:
            current.append(remaining)
            if check_inequalities(current, inequalities):
                count += 1
            current.pop()
            return

        # x_P 값 시도
        for val in range(remaining + 1):
            current.append(val)
            generate_partitions(remaining - val, num_vars - 1, current)
            current.pop()

    generate_partitions(n, m, [])

    return count


def check_inequalities(point, inequalities):
    """
    주어진 점이 모든 inequality를 만족하는지 확인

    Args:
        point: [x_1, x_2, ..., x_m]
        inequalities: [[b, -A]] for Ax ≥ -b

    Returns:
        True if all inequalities satisfied
    """
    for ineq in inequalities:
        b = ineq[0]
        A = ineq[1:]
        # b + A·x ≥ 0 체크
        value = b + sum(a * x for a, x in zip(A, point))
        if value < 0:
            return False
    return True


def compare_phase0_effect(n, k, q, d):
    """
    Phase 0 적용 전후의 격자점 개수 비교
    """
    print("="*60)
    print(f"Linear Code Parameters: [n={n}, k={k}, d={d}]_{q}")
    print("="*60)

    # Phase 0 없이 계산
    count_without, m = count_without_phase0(n, k, q)
    print(f"\n[Phase 0 없이]")
    print(f"  - PG({k-1}, {q})의 점 개수: {m}")
    print(f"  - 전체 격자점 개수: {count_without:,}")

    # Phase 0 적용
    print(f"\n[Phase 0 적용]")
    inequalities, equations, num_points, num_hyperplanes = generate_normaliz_constraints(n, k, q, d)
    print(f"  - Hyperplane 개수: {num_hyperplanes}")
    print(f"  - 제약조건: {len(inequalities)} inequalities, {len(equations)} equations")
    print(f"  - 격자점 개수 계산 중...")

    count_with = count_with_normaliz(inequalities, equations)

    if count_with is not None:
        print(f"  - Phase 0 적용 후 격자점 개수: {count_with:,}")

        reduction = count_without - count_with
        reduction_rate = (reduction / count_without) * 100 if count_without > 0 else 0

        print(f"\n[결과]")
        print(f"  - 제거된 격자점 개수: {reduction:,}")
        print(f"  - 감소율: {reduction_rate:.2f}%")
    else:
        print(f"  - 계산 실패")

    print("="*60)


# 테스트
if __name__ == "__main__":
    # 예시: [7, 3, 3]_2 코드
    compare_phase0_effect(n=7, k=3, q=2, d=3)

    print("\n\n")

    # 다른 예시
    compare_phase0_effect(n=10, k=3, q=2, d=4)
