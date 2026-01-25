# 실험 결과 데이터 설명서 (Experiment Results Description)

본 문서는 `experiment_results.csv` 파일에 기록된 각 컬럼의 의미와 해석 방법을 설명합니다. 이 데이터는 선형 부호 분류 알고리즘의 실행 결과, 성능 지표, 그리고 부호의 존재 여부를 포함하고 있습니다.

## 파일 개요
*   **파일명**: `experiment_results.csv`
*   **목적**: 선형 부호 $[n, k]_q$의 존재성 판별 실험 로그 및 알고리즘 성능 분석
*   **생성 주체**: `main.py` (실험 자동화 스크립트)

## 컬럼 상세 설명 (Column Details)

| 컬럼명 (Header) | 설명 (Description) | 비고 |
| :--- | :--- | :--- |
| **Timestamp** | 실험이 종료되고 결과가 기록된 날짜와 시간 | `YYYY-MM-DD HH:MM:SS` 형식 |
| **Length(n)** | 선형 부호의 길이 (Length) | 생성 행렬의 열(Column) 개수 |
| **Dimension(k)** | 선형 부호의 차원 (Dimension) | 생성 행렬의 행(Row) 개수 |
| **Field(q)** | 유한체의 크기 (Field Size) | 예: 2 (Binary), 3 (Ternary) 등 |
| **Target_Weights** | 허용된 비자명 부호어의 가중치 집합 | 예: `[3, 4]` (가중치가 3 또는 4인 부호어만 허용) |
| **Num_Points** | 사영 기하학 $PG(k-1, q)$의 전체 점 개수 | 탐색 공간의 크기를 결정하는 요소 |
| **Existence_Status** | 해당 파라미터의 부호 존재 여부 | `Feasible` (존재함), `Infeasible` (존재하지 않음) |
| **Search_Time(s)** | Phase 1 (탐색) 소요 시간 (초 단위) | ILP/Backtracking 알고리즘 실행 시간 |
| **Verify_Time(s)** | Phase 2 (검증) 소요 시간 (초 단위) | 해 검증 및 동형성(Isomorphism) 필터링 시간 |
| **Total_Solutions** | 솔버가 찾아낸 전체 해의 개수 | 중복(동형) 해를 포함한 수치 |
| **Unique_Solutions** | 동형성을 제거한 유일한 해의 개수 | 실제 서로 다른 부호의 개수 (Phase 2 결과) |
| **Nodes_Visited** | 탐색 트리에서 방문한 노드의 총 개수 | 알고리즘의 작업량을 나타냄 |
| **Pruned_Nodes** | 가지치기(Pruning)된 노드의 개수 | **Method 2 (RCUB)** 등에 의해 절약된 탐색 공간의 크기 |

## 결과 해석 가이드

### 1. 존재성 판별 (Existence)
*   **`Existence_Status`가 `Feasible`인 경우**: 주어진 $n, k, q$와 가중치 조건을 만족하는 선형 부호가 최소 1개 이상 존재합니다. `Unique_Solutions` 컬럼을 통해 서로 다른 구조의 부호가 몇 개인지 확인할 수 있습니다.
*   **`Existence_Status`가 `Infeasible`인 경우**: 해당 조건의 선형 부호는 수학적으로 존재하지 않습니다. 이는 전수 조사(Exhaustive Search)를 완료했음을 의미합니다.

### 2. 알고리즘 효율성 분석 (Efficiency)
*   **Pruning Ratio (가지치기 비율)**: `Pruned_Nodes` / (`Nodes_Visited` + `Pruned_Nodes`) 비율이 높을수록, 알고리즘이 불필요한 탐색을 효과적으로 줄였음을 의미합니다.
*   **Search Time**: `Nodes_Visited`가 비슷하더라도 `Search_Time`이 짧다면, **Method 1 (Watched-Hyperplane)**과 같은 최적화 기법이 노드 당 연산 비용을 잘 줄여주고 있다는 증거입니다.

### 3. 논문 비교 (Comparison)
*   논문의 실험 결과와 비교할 때는 `Nodes_Visited` (B&B-nodes) 수치를 주로 비교합니다. 본 알고리즘이 논문보다 적은 노드를 방문했다면 더 강력한 가지치기 조건(Method 2)이 적용된 것입니다.

---

**참고**: `experiment_results.csv` 파일의 헤더가 위 설명과 다르다면, 파일이 구버전일 수 있습니다. 파일을 삭제하거나 이름을 변경한 후 `main.py`를 다시 실행하면 올바른 헤더로 새로 생성됩니다.