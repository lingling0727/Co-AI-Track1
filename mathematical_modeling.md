# 부호 분류를 위한 하이브리드 알고리즘 수학적 모델링

본 문서는 선형 부호(Linear Code)의 분류 문제에서 발생하는 계산 복잡도를 해결하기 위해, **정수 선형 계획법(ILP)**과 **휴리스틱(Heuristic)**을 결합한 하이브리드 알고리즘의 수학적 모델링을 기술한다.

## 1. 문제 정의 (Problem Definition)

$[n, k]_q$ 선형 부호 $C$는 유한체 $F_q$ 위의 $k$차원 부분공간이다. 우리는 다음 조건을 만족하는 부호가 존재하는지 판별하고, 존재한다면 그 생성 행렬(Generator Matrix)을 구성하는 것을 목표로 한다.

*   **길이 (Length):** $n$
*   **차원 (Dimension):** $k$
*   **무게 스펙트럼 (Weight Spectrum):** 부호 $C$의 모든 비자명 부호어(non-zero codeword)의 무게 $w$는 집합 $W$에 속해야 한다. ($w \in W$)

이 문제는 기하학적으로 $PG(k-1, q)$ (Projective Space)에서 $n$개의 점(Point)을 중복을 허용하여 선택하는 문제로 환원된다.

## 2. 정수 선형 계획법 (ILP) 모델링

전체 투영 공간 $PG(k-1, q)$의 점의 개수를 $N$이라 하고, 각 점을 $P_0, P_1, \dots, P_{N-1}$이라 하자.
변수 $x_i$를 점 $P_i$가 부호의 생성 행렬에 포함되는 횟수(Multiplicity)라고 정의한다.

### 변수 (Variables)
$$ x_i \in \{0, 1, \dots, n\}, \quad \forall i \in \{0, \dots, N-1\} $$

### 제약 조건 (Constraints)

1.  **길이 제약 (Length Constraint):**
    선택된 점들의 총 개수는 $n$이어야 한다.
    $$ \sum_{i=0}^{N-1} x_i = n $$

2.  **무게 제약 (Weight Constraints):**
    투영 공간의 각 초평면(Hyperplane) $H_j$에 대하여, 해당 초평면에 포함된 점들의 개수를 $k_{H_j}$라 하면, 이에 대응하는 부호어의 무게는 $n - k_{H_j}$이다. 이 무게는 허용된 집합 $W$에 속해야 한다.
    $$ n - \sum_{P_i \in H_j} x_i \in W, \quad \forall H_j \subset PG(k-1, q) $$

    이를 선형화하기 위해 이진 변수 $b_{j, w}$를 도입한다. ($w \in W$)
    $$ \sum_{w \in W} b_{j, w} = 1 $$
    $$ \sum_{w \in W} w \cdot b_{j, w} = n - \sum_{P_i \in H_j} x_i $$

## 3. 하이브리드 알고리즘 (Hybrid Approach)

순수 ILP는 변수의 수($N$)가 커질수록 탐색 공간이 지수적으로 증가하여 해를 찾기 어렵다. 따라서 휴리스틱을 통해 탐색 공간을 사전에 축소(Pruning)하는 2단계 전략을 사용한다.

### Phase 1: 휴리스틱 탐색 및 공간 축소 (Heuristic Search & Pruning)
*   **목표:** 제약 조건을 "최대한" 만족하는 근사 해(Approximate Solution)를 빠르게 찾고, 유망한 점(Promising Points)들을 선별한다.
*   **알고리즘:** 탐욕적 초기화(Greedy Initialization) + 담금질 기법(Simulated Annealing).
*   **출력:** 근사 해 벡터 $\mathbf{x}^{heu}$.

### Phase 2: 축소된 공간에서의 ILP (ILP on Reduced Space)
*   **아이디어:** 휴리스틱이 선택하지 않은 점들은 최적 해에 포함될 확률이 낮다고 가정하고(가우시안 휴리스틱의 변형), 변수의 범위를 제한한다.
*   **변수 제한 (Variable Pruning):**
    휴리스틱 해에서 $x_i^{heu} > 0$인 점들은 **필수 후보군(Core Set)**으로 설정하고, 나머지 점들 중 일부만 **탐색 후보군(Exploration Set)**으로 추가한다. 나머지 점들에 대해서는 $x_i = 0$으로 고정한다.
*   **검증:** 축소된 변수 집합에 대해 ILP를 수행하여 엄밀한 해를 찾는다.