
import numpy as np
import pandas as pd
import galois
import json
import os
import itertools
from typing import List, Dict, Any, Set, Tuple

class KurzDatasetGenerator:
    """
    Sascha Kurz의 2024년 논문에 제시된 프로포지션을 재현하기 위한 지능형 데이터셋을 생성함.

    이 클래스는 사영 기하의 점, 인접 행렬, 그리고 솔버의 속도 향상을 위한
    사전 계산 필터가 포함된 설정 파일 생성을 담당함.
    """

    def __init__(self, propositions: List[Dict[str, Any]], base_path: str):
        """
        프로포지션 파라미터와 기본 출력 경로로 생성기를 초기화함.

        Args:
            propositions: 각 프로포지션의 파라미터를 정의하는 딕셔너리 리스트.
            base_path: 데이터셋 폴더가 생성될 루트 디렉토리.
        """
        self.propositions = propositions
        self.base_path = base_path
        print(f"데이터셋 기본 경로: {os.path.abspath(self.base_path)}")
        # 필수 라이브러리 확인
        try:
            import galois
            import pandas
        except ImportError as e:
            print(f"오류: 필수 라이브러리가 없음. 설치 필요.")
            print("pip install galois pandas numpy 명령어로 의존성을 설치할 수 있음.")
            raise e


    def _generate_points(self, k: int, q: int, GF: galois.Field) -> np.ndarray:
        """
        사영 기하 PG(k-1, q)의 정규화된 모든 고유 점을 생성함.

        정규화는 각 벡터의 첫 번째 0이 아닌 성분을 1로 만듦.
        점들은 사전순으로 정렬됨.
        """
        print(f"  PG({k-1}, {q})에 대한 점 생성 중...")
        # V(k, q) 내의 모든 가능한 벡터 생성 (영벡터 포함)
        all_vectors = list(itertools.product(range(q), repeat=k))
        
        # 영벡터 제외
        vectors_gf = GF([v for v in all_vectors if any(v)])

        # 각 벡터 정규화
        normalized_vectors = []
        for vec in vectors_gf:
            first_nonzero_idx = np.nonzero(vec)[0][0]
            normalized_vec = vec / vec[first_nonzero_idx]
            normalized_vectors.append(normalized_vec)
        
        # 고유한 점들을 추출하고 정렬
        unique_points = np.unique(np.array(normalized_vectors), axis=0)
        
        # 사전순 정렬
        sorted_points = unique_points[np.lexsort(unique_points.T[::-1])]
        
        print(f"  {len(sorted_points)}개의 고유한 점을 생성함.")
        return sorted_points

    def _generate_incidence_matrix(self, points: np.ndarray, GF: galois.Field) -> Tuple[np.ndarray, List[List[int]]]:
        """
        압축된 점-초평면 인접 행렬 및 역인덱스를 생성함.
        초평면은 점과 동일한 집합으로 표현됨.
        """
        print("  인접 행렬 및 역인덱스 생성 중...")
        points_gf = GF(points)
        
        # 점과 초평면 벡터의 내적이 0이면, 그 점은 해당 초평면 위에 있음. (벡터화 연산)
        incidence_matrix = (points_gf @ points_gf.T) == 0
        
        # [개선] 방법 1(Watched-Hyperplane)을 위한 역인덱스 생성
        # point_to_hypers[p_idx]는 점 p_idx를 포함하는 모든 초평면의 인덱스 리스트임.
        point_to_hypers = [np.where(row)[0].tolist() for row in incidence_matrix]
        
        # 공간 절약을 위해 불리언 행렬을 이진 포맷(uint8)으로 압축
        packed_matrix = np.packbits(incidence_matrix, axis=1)
        
        print(f"  인접 행렬이 {incidence_matrix.shape} 형태로 생성되었고, {packed_matrix.shape} 형태로 압축됨.")
        return packed_matrix, point_to_hypers

    def _generate_config_json(self, params: Dict[str, Any], output_dir: str):
        """
        O(1) 조회가 가능한 용량(capacity) 조회 맵과 가분성(delta) 정보가 포함된 config.json 파일을 생성함.
        """
        print("  config.json 생성 중...")
        n = params['n']
        w_set = params['w_set']
        
        # S_H = n - w, 여기서 w는 허용된 가중치 집합의 원소
        allowed_capacities: Set[int] = {n - w for w in w_set}
        
        # O(1) 조회를 위한 불리언 맵 생성. 리스트의 인덱스가 용량을 나타냄.
        capacity_lookup = [i in allowed_capacities for i in range(n + 1)]
        
        config_data = {
            "q": params['q'],
            "k": params['k'],
            "n": params['n'],
            "d_min": params['d'],
            "w_set": list(w_set),
            "delta": params.get("delta", 1), # [개선] 가분성 정보 추가
            "allowed_capacities_s_h": sorted(list(allowed_capacities)),
            "capacity_lookup": capacity_lookup
        }
        
        filepath = os.path.join(output_dir, "config.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print(f"  설정 파일을 {filepath}에 저장함.")

    def _generate_bounds_json(self, points: np.ndarray, k: int, GF: galois.Field, output_dir: str):
        """
        생성 행렬 G = [I_k | A] 형태를 강제하기 위한 bounds.json 파일을 생성함.
        단위 벡터들을 찾아 하한값(u_p)을 1로 설정함.
        """
        print("  bounds.json 생성 중...")
        bounds_data = self._create_bounds_dict(points, k, GF)
        
        filepath = os.path.join(output_dir, "bounds.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(bounds_data, f, indent=4, ensure_ascii=False)
        print(f"  경계값 파일을 {filepath}에 저장함.")

    def _create_bounds_dict(self, points: np.ndarray, k: int, GF: galois.Field) -> Dict[str, Any]:
        """
        메모리 내에서 경계값 딕셔너리를 생성함.
        """
        num_points = len(points)
        lower_bounds = [0] * num_points
        
        unit_vectors = GF.Identity(k)
        unit_vector_indices = []
        
        for uv in unit_vectors:
            idx_tuple = np.where(np.all(points == uv, axis=1))
            if idx_tuple[0].size > 0:
                idx = idx_tuple[0][0]
                lower_bounds[idx] = 1
                unit_vector_indices.append(int(idx))
            else:
                print(f"  경고: 단위 벡터 {uv}를 점 집합에서 찾을 수 없음.")

        return {
            "comment": f"{k}개의 단위 벡터를 고정하여 체계적인 형태 G=[I_k|A]를 강제함.",
            "unit_vector_indices": sorted(unit_vector_indices),
            "lower_bounds": lower_bounds
        }

    def _generate_config_json(self, params: Dict[str, Any], output_dir: str):
        """
        O(1) 조회가 가능한 용량(capacity) 조회 맵과 가분성(delta) 정보가 포함된 config.json 파일을 생성함.
        """
        print("  config.json 생성 중...")
        config_data = self._create_config_dict(params)
        
        filepath = os.path.join(output_dir, "config.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print(f"  설정 파일을 {filepath}에 저장함.")

    def _create_config_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        메모리 내에서 설정 딕셔너리를 생성함.
        """
        n = params['n']
        w_set = params.get('w_set', set())
        
        # S_H = n - w, 여기서 w는 허용된 가중치 집합의 원소
        allowed_capacities: Set[int] = {n - w for w in w_set}
        
        # O(1) 조회를 위한 불리언 맵 생성. 리스트의 인덱스가 용량을 나타냄.
        capacity_lookup = [i in allowed_capacities for i in range(n + 1)]
        
        return {
            "q": params['q'],
            "k": params['k'],
            "n": params['n'],
            "d_min": params['d'],
            "w_set": list(w_set),
            "delta": params.get("delta", 1),
            "allowed_capacities_s_h": sorted(list(allowed_capacities)),
            "capacity_lookup": capacity_lookup
        }

    def generate_data_in_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 프로포지션에 대한 데이터셋을 파일 저장 없이 메모리에서 직접 생성함.
        """
        prop_name = params.get("name", "in-memory")
        print(f"\n--- {prop_name}에 대한 인메모리 데이터 생성 시작 ---")
        
        q, k = params['q'], params['k']
        
        GF = galois.GF(q)
        
        points = self._generate_points(k, q, GF)
        
        incidence_packed, point_to_hypers = self._generate_incidence_matrix(points, GF)
        
        config_data = self._create_config_dict(params)
        
        bounds_data = self._create_bounds_dict(points, k, GF)
        
        print(f"--- {prop_name} 인메모리 생성 완료 ---")
        
        return {
            "points": points.astype(np.int8),
            "incidence_packed": incidence_packed,
            "point_to_hypers": point_to_hypers,
            "config": config_data,
            "bounds": bounds_data,
            "params": params
        }

    def generate_dataset(self, params: Dict[str, Any]):
        """
        하나의 프로포지션에 대한 전체 데이터셋 생성을 총괄하는 함수임.
        """
        prop_name = params["name"]
        output_dir = os.path.join(self.base_path, prop_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- {prop_name}에 대한 데이터셋 생성 시작 ---")
        
        q, k = params['q'], params['k']
        
        # 1. 갈루아 필드 초기화
        GF = galois.GF(q)

        # 2. 점 생성 및 저장
        points = self._generate_points(k, q, GF)
        # CSV (사람이 읽기 좋게)
        points_df = pd.DataFrame(points, columns=[f'x{i}' for i in range(k)])
        points_csv_filepath = os.path.join(output_dir, "points.csv")
        points_df.to_csv(points_csv_filepath, index=False)
        print(f"  점들을 {points_csv_filepath}에 저장함.")
        # [개선] Numpy 바이너리 (빠른 로딩)
        points_npy_filepath = os.path.join(output_dir, "points.npy")
        np.save(points_npy_filepath, points.astype(np.int8))
        print(f"  점들을 {points_npy_filepath}에 저장함.")


        # 3. 인접 행렬 및 역인덱스 생성/저장
        incidence_packed, point_to_hypers = self._generate_incidence_matrix(points, GF)
        # 압축된 인접 행렬
        incidence_filepath = os.path.join(output_dir, "incidence_packed.npy")
        np.save(incidence_filepath, incidence_packed)
        print(f"  압축된 인접 행렬을 {incidence_filepath}에 저장함.")
        # [개선] 점-초평면 역인덱스
        p2h_filepath = os.path.join(output_dir, "point_to_hypers.json")
        with open(p2h_filepath, 'w', encoding='utf-8') as f:
            json.dump(point_to_hypers, f)
        print(f"  점-초평면 역인덱스를 {p2h_filepath}에 저장함.")

        # 4. config.json 생성
        self._generate_config_json(params, output_dir)
        
        # 5. bounds.json 생성
        self._generate_bounds_json(points, k, GF, output_dir)
        print(f"--- {prop_name} 완료 ---")

    def run(self):
        """
        설정된 모든 프로포지션에 대해 데이터셋 생성 프로세스를 실행함.
        """
        for params in self.propositions:
            self.generate_dataset(params)
        print("\n모든 데이터셋이 성공적으로 생성됨.")


if __name__ == "__main__":
    # Kurz (2024) 논문의 각 프로포지션에 대한 파라미터 정의
    PROPOSITIONS_DATA = [
        {
            "name": "prop1",
            "q": 4, "k": 5, "n": 66, "d": 48, "delta": 4,
            "w_set": {48, 52, 56, 60, 64}
        },
        {
            "name": "prop2",
            "q": 8, "k": 4, "n": 35, "d": 28, "delta": 4,
            "w_set": {28, 32}
        },
        {
            "name": "prop3",
            "q": 5, "k": 4, "n": 40, "d": 30, "delta": 5,
            "w_set": {30, 35, 40}
        },
        {
            "name": "prop4",
            "q": 2, "k": 7, "n": 153, "d": 76, "delta": 4,
            "w_set": {76, 80, 92, 96, 100} # Lemma 4 반영
        }
    ]
    
    # 스크립트는 'datasets' 폴더 내에 있으므로, 출력 폴더 'propX'는
    # 이 스크립트와 동일한 위치에 생성됨.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    generator = KurzDatasetGenerator(propositions=PROPOSITIONS_DATA, base_path=script_dir)
    generator.run()
