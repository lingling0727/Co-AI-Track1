# Codebase Documentation: Linear Code Classification

본 문서는 논문 *Computer Classification of Linear Codes Based on Lattice Point Enumeration and Integer Linear Programming*을 기반으로 구현된 Python 코드의 구조, 진행 과정, 변수 및 함수 설정을 설명합니다.

## 1. 프로젝트 개요 (Project Overview)

이 프로젝트는 유한체 $\mathbb{F}_q$ 위에서 특정 가중치 집합(Weight Set)을 만족하는 선형 부호 $[n, k]_q$를 분류하거나 존재성을 판별합니다.

### 전체 진행 프로세스
1.  **Geometry Generation**: 사영 기하학 $PG(k-1, q)$의 점과 초평면을 생성합니다.
2.  **Phase 0 & 1 (ILP Solving)**: 기하학적 제약 조건과 가중치 조건을 만족하는 정수 해(Integer Solution)를 열거합니다.
3.  **Phase 2 (Verification)**: 찾아낸 해가 유효한지 검증하고, 동형(Isomorphic)인 중복 해를 제거합니다.

---

## 2. 파일별 상세 설명 (Detailed Description)

### 2.1 `geometry.py`
유한체 및 사영 기하학적 구조를 생성하는 모듈입니다.

*   **`generate_projective_points(k, q)`**
    *   **기능**: $k$차원 벡터 공간에서 1차원 부분 공간(사영 점)들을 대표하는 벡터들을 생성합니다.
    *   **알고리즘**: 모든 $q^k$개 벡터 중 0벡터를 제외하고, 첫 번째 0이 아닌 성분이 1이 되도록 정규화(Normalization)하여 중복을 제거합니다.
    *   **반환값**: $PG(k-1, q)$의 점 리스트 (크기: $\frac{q^k-1}{q-1}$).

*   **`get_projection_map(k_target, q, points_k, points_k_minus_1)`**
    *   **기능**: $k$차원 공간의 점들을 $k-1$차원 공간으로 투영(Projection)하는 매핑 테이블을 만듭니다.
    *   **용도**: 논문의 **Lemma 1 (Extension)** 알고리즘에서, 이전 차원의 코드 구조를 유지하며 확장할 때 사용됩니다.
    *   **반환값**: `{k-1차원_점_인덱스: [k차원_점_인덱스_리스트]}` 딕셔너리.

*   **`is_point_in_hyperplane(point, hyperplane, q)`**
    *   **기능**: 점과 초평면의 내적(Dot Product)이 0인지 확인하여 포함 여부를 판별합니다.
    *   **주의**: 현재 구현은 `sum(p*h) % q` 방식을 사용하므로, $q$가 소수(Prime)일 때만 정확합니다.

### 2.2 `ilp_model.py`
Google OR-Tools의 CP-SAT 솔버를 사용하여 부호의 존재성을 탐색하는 핵심 모듈입니다.

*   **Class `CodeExtender`**
    *   **`__init__(n, k, q, target_weights)`**:
        *   `n`: 목표 부호 길이.
        *   `k`: 목표 차원.
        *   `target_weights`: 허용되는 부호어의 가중치 집합 (예: $\{3, 4\}$).
    
    *   **`build_and_solve(points, hyperplanes, base_code_counts, points_km1)`**:
        *   **변수 (`x`)**: 각 사영 점 $P$가 부호 생성 행렬의 열(Column)로 몇 번 선택되었는지를 나타내는 정수 변수 ($x_P \ge 0$).
        *   **제약 조건 1 (Length)**: $\sum x_P = n$.
        *   **제약 조건 2 (Weight)**: 모든 초평면 $H$에 대해, $H$에 포함된 점들의 개수 합 $S_H$는 $n - w$ ($w \in \text{Weights}$)여야 합니다.
            *   `model.AddLinearExpressionInDomain`을 사용하여 Phase 0(Feasibility Check) 역할을 수행합니다.
        *   **제약 조건 3 (Extension)**: `base_code_counts`가 주어지면, 투영된 점들의 합이 이전 코드의 구성과 일치해야 한다는 제약(Lemma 1, Eq 4)을 추가합니다.
        *   **제약 조건 4 (Symmetry Breaking)**: 처음부터 생성(Scratch)할 경우, 기저 벡터를 강제로 포함시켜 탐색 공간을 줄입니다.

### 2.3 `checker.py`
솔버가 찾은 해를 후처리하는 모듈입니다.

*   **`verify_solution(...)`**:
    *   찾아낸 해(점들의 중복 집합)를 바탕으로 실제 모든 초평면(부호어)의 가중치를 다시 계산하여 `target_weights`에 포함되는지 검증합니다.
*   **`filter_isomorphic_solutions(...)`**:
    *   찾아낸 여러 해 중에서 구조적으로 동일한(Isomorphic) 해를 제거합니다.
    *   현재는 점들의 선택 횟수 분포(Canonical Form)를 비교하는 간단한 방식을 사용합니다.

### 2.4 `main.py`
전체 프로그램을 실행하는 진입점입니다.

*   **`run_classification(...)`**:
    1.  `geometry.py`를 호출하여 기하학 데이터 생성.
    2.  `ilp_model.py`를 호출하여 해 탐색 (Phase 1).
    3.  `checker.py`를 호출하여 검증 및 필터링 (Phase 2).
    4.  결과를 `experiment_results.csv`에 저장.
*   **실행 모드**:
    *   인자 없이 실행 시: 논문 재현을 위한 테스트 케이스(Extension Test) 수행.
    *   인자 포함 실행 시: 사용자 지정 파라미터로 분류 수행.

---

## 3. 주요 변수 설명 (Key Variables)

*   **`n` (Length)**: 선형 부호의 길이. 생성 행렬 $G$의 열(Column)의 개수와 같습니다.
*   **`k` (Dimension)**: 선형 부호의 차원. 생성 행렬 $G$의 행(Row)의 개수와 같습니다.
*   **`q` (Field Size)**: 유한체의 크기. (예: $q=2$는 이진 부호).
*   **`d` (Minimum Distance)**: 부호의 최소 거리. 본 코드에서는 `target_weights`의 최솟값에 해당합니다.
*   **`x_P` (Multiplicity)**: 사영 공간의 점 $P$가 생성 행렬에 몇 번 등장하는지를 나타내는 결정 변수입니다.
*   **`Incidence Matrix`**: 점과 초평면의 포함 관계를 나타내는 행렬. $M_{ij} = 1$이면 점 $j$가 초평면 $i$에 포함됨을 의미합니다.