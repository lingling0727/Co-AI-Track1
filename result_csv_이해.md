# experiment_results.csv — 설명서

이 문서는 `experiment_results.csv`의 열(column) 의미와 분석 시 주의사항을 간단히 정리합니다.

파일 구조(열 순서)

- Timestamp: 결과가 기록된 시각 (YYYY-MM-DD HH:MM:SS)
- n: 부호의 전체 길이(총 좌표 수)
- k: 기하적 차원 관련 파라미터 (프로젝트 차원 k)
- q: 체 크기 (유한체의 q)
- Weights: 목표 무게 집합 (예: `{28, 32}`)
- Points: 데이터셋에 로드된 점 수 (projective space에서의 점 개수)
- Incidence_Size: incidence 행렬 크기 표기 (rows x cols)
- Allowed_Variables_Count: 하이브리드 실행 시 ILP에 남겨둔 변수(점) 수
- Seed: 실험(휴리스틱/데이터 생성)에 사용한 무작위 시드

- ILP_Status: ILP 실행 결과 상태 (예: `Feasible`, `Infeasible`, `Solver Not Found`, `Skipped`)
- ILP_Time: ILP가 끝날 때까지 걸린 시간(초)

- Heuristic_Status: 휴리스틱(담금질 등) 결과 상태 (`Success` = 제약 위반 0, `Fail` = 최종 위반 존재)
- Heuristic_Time: 휴리스틱 실행 시간(초)
- Heuristic_Cost: 휴리스틱의 최종 비용(위반 개수)
- Heuristic_MaxIter / Heuristic_Temp / Heuristic_CoolRate: 휴리스틱에 사용된 파라미터(반복수, 초기 온도, 냉각률)

- Hybrid_Status: 하이브리드(휴리스틱 → ILP) 최종 상태 (예: `Hybrid-Feasible`, `Hybrid-Infeasible`, `Optimal (Heuristic)`)
- Hybrid_Time: 하이브리드 전체 소요 시간(초)

해석 가이드(간단)

- 시간 단위는 초(seconds)입니다. ILP/Hybrid 시간이 지나치게 길면 `solve_ilp`에 시간 제한을 설정하세요.
- `Heuristic_Status == Success`이면 `Heuristic_Cost`는 0이며, 하이브리드 단계에서 ILP를 건너뛸 수 있습니다.
- `Allowed_Variables_Count`는 하이브리드에서 ILP에 전달된 변수 개수로, 작을수록 ILP가 더 빨리 풀릴 가능성이 높습니다.
- `Seed`를 기록하면 동일 파라미터로 실험을 재현할 수 있습니다(재현하려면 코드가 해당 시드를 사용하도록 해야 함).

주의/검증

- CSV 열/헤더가 변경되면 분석 파이프라인(스크립트)이 실패할 수 있으므로 헤더와 필드 매핑을 확인하세요.
- 데이터셋(`dataset/projective_space_k{k}_q{q}.txt`)의 정합성(점 수, 중복, 좌표 범위)은 `generate_dataset.py`의 검증 기능에서 자동으로 검사합니다. 검증 실패 시 `main.py` 실행 로그에 경고/에러가 출력됩니다.

권장 후속작업

- 결과를 자동으로 요약하는 스크립트(예: 성공률, 평균 시간, n 대비 스케일링)를 추가하면 분석이 편해집니다.
- ILP 시간 제한 및 휴리스틱 다중 시작을 실험 파라미터로 두어 재실험을 쉽게 하세요.

예시 행

```
2026-01-21 05:23:54,34,3,8,"{28, 32}",73,73x73,21,1402016181,Feasible,3.6482,Fail,0.9503,8,5000,100.0,0.995,Hybrid-Infeasible,0.3666
```

- 위 예시는 n=34, k=3, q=8 실험에서 데이터셋의 점 수가 73이며(Incidence 73x73), 하이브리드 단계에서 ILP에 21개 변수만 남겨둔 상태였음을 나타냅니다.

