[Kurz 2024 Benchmark Dataset: Binary n=153]

1. 개요
   이 데이터셋은 Sascha Kurz의 2024년 논문에서 다루어진 
   Binary Code [153, 7, 76]_2의 해(Solution)를 분류하기 위한 
   기하학적 환경을 제공함.

2. 파라미터 설정 (config.json)
   - Field (q): 2 (Binary)
   - Length (n): 153
   - Dimension (k): 7 (Projective dimension = 6)
   - Distance (d): 76
   - Divisibility (delta): 4
   - Valid Weights (w_set): [76, 80, 92, 96, 100]
     * 설명: 모든 초평면의 가중치 w는 d(27) 이상이어야 하며, 
       Divisible Code 조건에 따라 4의 배수여야 함.
       (논문 증명에 따라 84, 88 가중치는 존재하지 않으므로 제외함)
   - Allowed Capacities: [77, 73, 61, 57, 53]
     * 설명: 초평면이 포함할 수 있는 점의 개수(n - w)임.
       RCUB 계산 시 단순히 상한선만 보는 것이 아니라, 
       이산적(Discrete) 용량 조건을 활용하여 가지치기를 수행해야 함.

3. 파일 구조 및 활용 가이드

   A. points.csv (Geometry)
      - PG(6, 2)의 정규화된 점 127개를 포함함.
      - Search Engine의 'Orbit-Representative' 단계에서 
        점들의 좌표를 기반으로 대칭성을 계산할 때 사용됨.

   B. incidence_packed.npy (Structure)
      - 127x16 (Packed Bits) 행렬. numpy.packbits로 압축됨.
      - 행(Row)은 초평면, 열(Col)은 점(Bitset)을 의미함.
      - 활용: Search Engine에서 np.unpackbits(..., axis=1)로 복원 후 사용하거나,
        비트 연산으로 Watched-Hyperplane을 최적화할 수 있음.

   C. bounds.json (Constraints)
      - u_p (Lower Bound): 단위 벡터(Unit Vectors)에 해당하는 점은 1, 나머지는 0.
        (Systematic Generator Matrix 형태 강제)
      - lambda_p (Upper Bound): 2로 초기화됨. (각 점은 최대 2번 사용 가능)
      - 활용: Search Engine의 'RCUB' 단계에서 
        초기 탐색 공간을 정의함. 
        (실제 탐색 시 RCUB 로직이 동적으로 상한을 더 타이트하게 조임.)

4. 검증 목표
   - 이 설정에서 정확히 2개의 비동형(Non-isomorphic) 해를 찾아내야 함.
   - 이는 단순한 존재성 여부를 넘어 알고리즘의 정밀함을 검증하는 목표임.

5. 심화 분석 및 구현 전략 (Critique Reflection)
   A. RCUB의 정교화 (Discrete Pruning)
      - 허용된 용량 집합이 불연속적임 (예: 73 다음은 61).
      - 구현 시: 현재 용량이 특정 값 미만으로 떨어지면, 
        즉시 다음 허용 용량으로 상한을 낮추는 강력한 가지치기 로직을 적용해야 함.

   B. Orbit vs Lexicographical
      - PG(6, 2) 공간은 점이 127개로 대칭성이 매우 큼.
      - Nauty 연동 오버헤드를 고려하여, 초기 단계에서는 사전식 확장(Lexicographical Extension)을 
        우선 적용하고, 깊이가 깊어질수록 Orbit 전략을 혼용하는 것이 유리함.

   C. 체계적 확장 (Systematic Extension)
      - bounds.json은 단위 벡터를 고정하여 체계적 생성 행렬 형태를 강제함.
      - 이는 탐색 트리의 루트를 고정하는 효과가 있어 중복 탐색을 크게 줄여줌.

6. 연구 타당성 및 현실성 분석 (Feasibility Analysis)
   A. 선정 이유: 논문의 핵심 증명 케이스 (Case s=13)
      - Sascha Kurz(2024) Proposition 4에서 다루는 최적 부호 케이스임.
      - "차원이 3.5인 최적 가산 4원 부호"의 존재성을 증명하는 핵심 연결 고리임.

   B. 존재성 및 분류의 유의미성
      - [153, 7, 76]_2는 비동형(Non-isomorphic) 해가 정확히 2개 존재함.
      - 단순 "해 없음" 판정보다, 수조 개의 경우 중 2개의 정답을 찾아내는 것이 
        알고리즘 성능 증명에 훨씬 적합함.

   C. 기하학적 복잡도와 최적화 테스트
      - PG(6, 2)의 127개 점을 다루며, 탐색 공간이 2^1071에 달해 기존 방식으로는 불가능함.
      - 설계한 x_P 모델과 RCUB Pruning이 이 복잡한 구조에서 얼마나 빠르게 해를 찾는지 
        보여줌으로써 방식의 우월성을 입증할 수 있음.

   D. 요약 (링링의 한마디)
      - "본 프로젝트는 Sascha Kurz(2024) 논문의 Proposition 4를 기반으로 [153, 7, 76]_2 파라미터의 
        4-가분 이진 부호를 구현함. 이는 이론적으로 분류된 2개의 비동형 해를 전수 조사함으로써, 
        제안된 정수 선형 계획법(ILP) 모델과 대칭성 제거 알고리즘의 효율성을 검증하기에 가장 적합한 벤치마크임."