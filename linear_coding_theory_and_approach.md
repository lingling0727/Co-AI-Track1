# 선형 부호 이론과 기하학적 접근법을 통한 부호 분류 연구

본 문서는 **선형 부호(Linear Code)**의 기초 이론부터, 이를 **사영 기하학(Projective Geometry)**의 관점에서 해석하고, **정수계획법(Integer Linear Programming, ILP)**과 **휴리스틱(Heuristic)**을 결합하여 부호의 존재성을 규명하는 본 연구의 방법론을 상세히 기술한다.

---

## 1. 선형 부호 이론의 기초 (Fundamentals of Linear Coding Theory)

### 1.1 선형 부호의 정의
선형 부호 $C$는 유한체(Finite Field) $\mathbb{F}_q$ 위에서의 벡터 공간 $\mathbb{F}_q^n$의 부분 공간(Subspace)으로 정의된다.

*   **파라미터 $[n, k, d]_q$**:
    *   **$n$ (Length)**: 부호어(Codeword)의 길이 (벡터의 성분 개수).
    *   **$k$ (Dimension)**: 부호 $C$의 차원. 즉, 정보 비트의 수. $|C| = q^k$.
    *   **$d$ (Minimum Distance)**: 서로 다른 두 부호어 사이의 최소 해밍 거리(Hamming Distance). 오류 정정 능력을 결정하는 핵심 지표.
    *   **$q$ (Field Size)**: 사용하는 유한체의 크기 (예: $q=2$이면 이진 부호).

### 1.2 생성 행렬 (Generator Matrix)
선형 부호 $C$는 $k \times n$ 크기의 행렬 $G$에 의해 생성된다.
$$ C = \{ uG \mid u \in \mathbb{F}_q^k \} $$
여기서 $G$의 행(Row)들은 $C$의 기저(Basis)를 이룬다.

### 1.3 부호어의 무게 (Weight)
*   **해밍 무게 (Hamming Weight)** $w(c)$: 벡터 $c$에서 0이 아닌 성분의 개수.
*   **무게 스펙트럼 (Weight Spectrum)** $W$: 부호 $C$에 속하는 모든 비자명(non-zero) 부호어들이 가질 수 있는 무게들의 집합.
    *   본 연구의 핵심 문제는 **"특정 무게 스펙트럼 $W$를 만족하는 $[n, k]_q$ 부호가 존재하는가?"**를 판별하는 것이다.

---

## 2. 기하학적 관점: 부호와 투영 공간 (Geometric View)

전통적인 대수적 접근(행렬 연산) 대신, 본 연구는 **기하학적 접근법**을 채택한다. 이는 생성 행렬 $G$의 **열(Column)** 벡터들을 기하학적 공간의 **점(Point)**으로 해석하는 방식이다.

### 2.1 생성 행렬과 투영 공간의 대응
생성 행렬 $G = [g_1, g_2, \dots, g_n]$의 각 열 $g_i$는 $\mathbb{F}_q^k$의 벡터이다. 영벡터가 아닌 $g_i$는 $k$차원 벡터 공간의 1차원 부분 공간을 생성하며, 이는 **$(k-1)$차원 투영 공간 $PG(k-1, q)$의 점**에 대응된다.

*   **부호 $C$ $\iff$ 점들의 중복 집합 (Multiset of Points)**
    *   $[n, k]_q$ 부호는 $PG(k-1, q)$에서 $n$개의 점을 선택하는 것과 동치이다. (중복 선택 가능)
    *   두 열 벡터가 스칼라 배 관계($v = \lambda u$)에 있다면, 이들은 투영 공간에서 **같은 점**이다.

### 2.2 부호어와 초평면 (Codewords and Hyperplanes)
투영 기하학에서 **초평면(Hyperplane)**은 $(k-2)$차원의 부분 공간을 의미한다.

*   **쌍대성 (Duality)**: $\mathbb{F}_q^k$의 비자명 벡터 $u$는 부호어 $c = uG$를 생성한다. 동시에 $u$는 투영 공간의 초평면 $H_u = \{ x \in PG(k-1, q) \mid u \cdot x = 0 \}$을 정의한다.
*   **기하학적 무게 공식**:
    부호어 $c$의 무게 $w(c)$는 전체 길이 $n$에서 해당 초평면 $H_u$ 위에 있는 점들의 개수를 뺀 것과 같다.
    $$ w(c) = n - |\{ i \mid g_i \in H_u \}| $$
    여기서 $|\cdot|$는 중복도(Multiplicity)를 포함한 개수이다.

---

## 3. 연구 문제의 재정의 (Problem Statement)

위의 기하학적 대응 원리를 통해, 부호 분류 문제는 다음과 같은 **조합론적 최적화 문제**로 변환된다.

**문제:**
투영 공간 $PG(k-1, q)$의 점들 중에서 $n$개의 점(중복 허용)을 선택하여 다중 집합 $\mathcal{K}$를 구성하라. 단, 공간 내의 **모든 초평면 $H$**에 대하여 다음 조건을 만족해야 한다.

$$ n - |\mathcal{K} \cap H| \in W $$

여기서 $W$는 허용된 무게의 집합(Target Weights)이다.

---

## 4. 해결 방법론 1: 정수계획법 (ILP) 모델링

이 문제를 풀기 위해 **정수 선형 계획법(Integer Linear Programming)**을 도입한다. 이는 "Phase 0"라고 불리며, 격자점 열거(Lattice Point Enumeration) 전에 불가능한 후보를 빠르게 제거하는 역할을 한다.

### 4.1 변수 정의
투영 공간 $PG(k-1, q)$의 모든 점을 $P_1, P_2, \dots, P_N$이라 하자. ($N = \frac{q^k - 1}{q - 1}$)

*   **결정 변수 $x_i$**: 점 $P_i$가 부호(생성 행렬)에 포함되는 횟수 (정수, $x_i \ge 0$).

### 4.2 제약 조건 (Constraints)

1.  **길이 제약 (Length Constraint)**:
    선택된 점들의 총합은 $n$이어야 한다.
    $$ \sum_{i=1}^{N} x_i = n $$

2.  **무게 제약 (Weight Constraints)**:
    모든 초평면 $H_j$ ($j=1, \dots, N$)에 대하여,
    $$ k_{H_j} = \sum_{P_i \in H_j} x_i $$
    $$ w_j = n - k_{H_j} $$
    $$ w_j \in W $$

    이를 선형 부등식으로 표현하기 위해 보조 이진 변수 $b_{j, w}$를 사용한다. ($w \in W$)
    $$ \sum_{w \in W} b_{j, w} = 1 $$
    $$ \sum_{w \in W} w \cdot b_{j, w} = n - \sum_{P_i \in H_j} x_i $$

### 4.3 ILP의 장점
*   **엄밀성**: 해가 존재하지 않음(Infeasible)을 수학적으로 증명할 수 있다.
*   **유연성**: 무게 조건 외에도 자기 쌍대성(Self-duality), 분할 가능성(Divisibility) 등 추가 제약을 쉽게 반영할 수 있다.

---

## 5. 해결 방법론 2: 하이브리드 알고리즘 (Hybrid Algorithm)

문제의 크기($n, k, q$)가 커지면 투영 공간의 점 개수 $N$이 급증하여 순수 ILP로 풀기 어려워진다. 이를 해결하기 위해 본 연구에서는 **휴리스틱과 ILP를 결합**한다.

### 5.1 Phase 1: 휴리스틱 탐색 (Heuristic Search)
*   **목적**: 제약 조건을 "거의" 만족하는 해를 빠르게 찾고, 유망한 점(Promising Points)들을 식별한다.
*   **알고리즘**:
    1.  **탐욕적 초기화 (Greedy Initialization)**: 매 단계에서 제약 위반 비용(Cost)을 가장 많이 줄이는 점을 선택하여 초기 해를 구성한다.
    2.  **담금질 기법 (Simulated Annealing)**: 초기 해를 바탕으로 확률적 탐색을 수행하여 국소 최적해(Local Optima)를 탈출하고 더 나은 해를 찾는다.

### 5.2 Phase 2: 공간 축소 및 정밀 검증 (Pruning & Exact Verification)
*   **가설**: 최적 해는 휴리스틱이 찾아낸 유망한 점들의 집합(Core Set)과 그 주변 점들로 구성될 확률이 높다.
*   **과정**:
    1.  휴리스틱 결과에서 $x_i > 0$인 점들을 **Core Set**으로 선정한다.
    2.  나머지 점들 중 일부를 무작위로 **Exploration Set**으로 추가한다.
    3.  선택되지 않은 나머지 점들의 변수 $x_i$를 0으로 고정(Fixing)하여 ILP 문제의 크기를 대폭 줄인다.
    4.  축소된 ILP를 풀어 최종 해를 구한다.

---

## 6. 연구의 의의 및 기대 효과

1.  **계산 효율성 증대**: 기존의 전수 조사(Exhaustive Search) 방식에 비해 탐색 공간을 획기적으로 줄여, 기존에 풀지 못했던 대규모 파라미터($n=76$ 등)의 부호를 분류할 수 있다.
2.  **이론적 확장**: 기하학적 접근법을 통해 선형 부호뿐만 아니라 덧셈 부호(Additive Codes) 등 다른 대수적 구조로의 확장이 가능하다.
3.  **실용적 가치**: 최적의 무게 스펙트럼을 갖는 부호는 통신 시스템의 오류 정정 성능을 극대화하는 데 직접적으로 기여한다.

---

## 부록: 알고리즘 구현 요약 (Code Structure)

본 연구의 구현 코드는 다음과 같이 구성된다.

1.  **`generate_dataset.py`**:
    *   $PG(k-1, q)$의 모든 점을 생성하고 정규화(Canonical Form)한다.
    *   $q$가 소수일 때는 모듈러 역원을, 합성수일 때는 다항식 연산을 고려한다.

2.  **`main.py`**:
    *   **`GaloisFieldHelper`**: 유한체 연산 및 내적 계산.
    *   **`CodeClassifierExperiment`**: 실험 클래스.
        *   `solve_ilp()`: OR-Tools를 이용한 엄밀한 해 탐색.
        *   `solve_heuristic()`: Greedy + Simulated Annealing을 이용한 근사 해 탐색.
        *   `solve_hybrid()`: 휴리스틱으로 변수를 가지치기(Pruning)한 후 ILP 수행.

3.  **`experiment_parameters.txt`**:
    *   실험할 부호의 파라미터($n, k, q, W$)를 정의한다.
```

<!--
[PROMPT_SUGGESTION]작성된 마크다운 파일의 내용을 바탕으로, 연구 보고서의 '방법론' 섹션에 들어갈 텍스트를 좀 더 학술적인 어조로 다듬어 줘.[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Hybrid 알고리즘의 성능을 극대화하기 위해, Phase 2에서 Exploration Set을 랜덤하게 뽑는 대신 더 스마트하게 뽑는 방법(예: Reduced Cost 활용)을 제안해 줘.[/PROMPT_SUGGESTION]