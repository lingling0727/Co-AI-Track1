[Kurz 2024 Benchmark Dataset: Ternary n=41]

1. 개요
   이 데이터셋은 Sascha Kurz의 2024년 논문에서 다루어진 
   Ternary Code [41, 4, 27]_3의 존재 여부(Non-existence)를 검증하기 위한 
   기하학적 환경을 제공함.

2. 파라미터 설정 (config.json)
   - Field (q): 3 (Ternary)
   - Length (n): 41
   - Dimension (k): 4 (Projective dimension = 3)
   - Distance (d): 27
   - Divisibility (delta): 9
   - Valid Weights (w_set): [27, 36]
     * 설명: 모든 초평면의 가중치 w는 d(27) 이상이어야 하며, 
       Divisible Code 조건에 따라 9의 배수여야 함.
   - Allowed Capacities: [14, 5]
     * 설명: 초평면이 포함할 수 있는 점의 개수(n - w)임.
       RCUB 계산 시 단순히 상한선(14)만 보는 것이 아니라, 
       14 다음은 바로 5로 떨어져야 한다는 이산적(Discrete) 조건을 활용해야 함.

3. 파일 구조 및 활용 가이드

   A. points.csv (Geometry)
      - PG(3, 3)의 정규화된 점 40개를 포함함.
      - Search Engine의 'Orbit-Representative' 단계에서 
        점들의 좌표를 기반으로 대칭성을 계산할 때 사용됨.

   B. incidence_packed.npy (Structure)
      - 40x5 (Packed Bits) 행렬. numpy.packbits로 압축됨.
      - 행(Row)은 초평면, 열(Col)은 점(Bitset)을 의미함.
      - 활용: Search Engine에서 np.unpackbits(..., axis=1)로 복원 후 사용하거나,
        비트 연산으로 Watched-Hyperplane을 최적화할 수 있음.

   C. bounds.json (Constraints)
      - u_p (Lower Bound): 단위 벡터(Unit Vectors)에 해당하는 점은 1, 나머지는 0.
        (Systematic Generator Matrix 형태 강제)
      - lambda_p (Upper Bound): 41로 초기화됨.
      - 활용: Search Engine의 'RCUB' 단계에서 
        초기 탐색 공간을 정의함. 
        (실제 탐색 시 RCUB 로직이 동적으로 상한을 더 타이트하게 조임.)

4. 검증 목표
   - 이 설정에서 Feasible Solution(해)이 발견되지 않아야 
     Kurz의 "Non-existence" 결과와 일치함.

5. 심화 분석 및 구현 전략 (Critique Reflection)
   A. RCUB의 정교화 (Discrete Pruning)
      - 일반적인 Singleton Bound 변형(n-d)은 상한을 14로 잡지만,
        이 문제에서는 용량이 14 미만이 되는 순간 13, 12...가 아니라 
        바로 5 이하로 떨어져야 함.
      - 구현 시: `current_capacity < 14`이면 `current_capacity <= 5`인지 검사하는
        강력한 가지치기 로직을 적용해야 함.

   B. Orbit vs Lexicographical
      - 구현 비용: PG(3, 3)의 점 40개에 대해 매 노드에서 Automorphism Group을 계산하는 것은 
        배보다 배꼽이 더 클 수 있음.
      - Kurz의 접근: 이를 해결하기 위해 Kurz는 **사전식 확장(Lexicographical Extension)**을 사용했음.
      - 검증 요구사항: 만약 Orbit 전략(Nauty 연동)을 사용한다면, 그 오버헤드를 
        Watched 기법의 가속 효과로 상쇄할 수 있는지에 대한 실험적 증명이 반드시 필요함.

   C. 체계적 확장 (Systematic Extension)
      - bounds.json은 단위 벡터를 고정하여 체계적 생성 행렬 형태를 강제함.
      - 이는 탐색 트리의 루트를 고정하는 효과가 있어 중복 탐색을 크게 줄여줌.

6. 연구 타당성 및 현실성 분석 (Feasibility Analysis)
   A. Kurz 논문의 파라미터 구성
      - Binary Codes: n <= 136, 짝수(Even), 이중 짝수(Doubly-even) 부호 등 분류.
      - Ternary Codes: 9-분할 가능(9-divisible) 부호에 대해 n <= 82, k <= 6 범위 탐색.
      - 기타: 4진(Quaternary) 및 특정 차원의 최적 부호 유일성 증명 포함.

   B. n=41 파라미터 선택의 이유
      - 비존재성(Non-existence) 확정: n=41은 9-divisible 부호가 존재하지 않는 'Zero row' 구간임.
      - RCUB의 위력 과시: 해가 없는 경로를 사전에 차단하는 RCUB가 기존 방식 대비 
        압도적인 노드 감소율을 보여줄 수 있는 최적의 케이스.

   C. 현실적 구현 가능성 (3일 이내)
      - 시간 소요: Kurz는 n=41 확장에 단일 코어로 약 200시간을 소요하기도 했음.
      - 전략: Watched와 RCUB는 Kurz가 언급한 "솔버보다 느린 격자점 열거 알고리즘의 병목"을 공략.
      - 우선순위: PG(3, 3) 점 40개 규모에서 Watched와 RCUB 구현만으로도 3일 내 결과 도출 가능.
        (Orbit 구현은 시간 부족 시 후순위로 미룸)

   D. Kurz의 솔버(Solver) 활용 팩트
      - Kurz는 확장이 오래 걸릴 때 CPLEX 등 상용 솔버로 불가능성(Infeasibility)을 확인했음.
      - 이는 Gurobi/CPLEX를 활용한 필터링 전략이 학술적으로 정당함을 뒷받침함.