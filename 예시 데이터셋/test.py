import numpy as np
import pandas as pd
import json
import os

class DatasetValidator:
    def __init__(self, paths):
        self.paths = paths
        self.results = {}

    def check_file_exists(self):
        """파일 존재 여부 확인"""
        for key, path in self.paths.items():
            exists = os.path.exists(path)
            self.results[f"File_{key}_Exists"] = "PASS" if exists else "FAIL"

    def validate_dataset(self):
        try:
            # 1. 환경(Env) 검증 (제약 1, 2)
            with open(self.paths['config'], 'r') as f:
                conf = json.load(f)
            q, delta, w_set = conf['q'], conf['delta'], conf['w_set']
            
            # 제약 1: q가 소수 거듭제곱인가 (q=4 기준)
            self.results["C1_Q_Prime_Power"] = "PASS" if q in [2,3,4,5,7,8,9,11] else "FAIL"
            # 제약 2: w_set 가분성 체크
            self.results["C2_Divisibility"] = "PASS" if all(w % delta == 0 for w in w_set) else "FAIL"

            # 2. 구조(Geometry) 검증 (제약 3, 4, 11)
            pts_df = pd.read_csv(self.paths['points'])
            points = [list(map(int, p.split(','))) for p in pts_df['v_p']]
            incidence = np.load(self.paths['incidence_npy'])

            # 제약 3: 영벡터 포함 여부
            self.results["C3_No_Zero_Vector"] = "PASS" if all(any(v != 0 for v in pt) for pt in points) else "FAIL"
            # 제약 4: 정규화 여부 (첫 비영 성분이 1인가)
            is_canonical = True
            for pt in points:
                first_nonzero = next((x for x in pt if x != 0), None)
                if first_nonzero != 1: is_canonical = False; break
            self.results["C4_Normalization"] = "PASS" if is_canonical else "FAIL"
            # 제약 11: 인덱스 동기화 (점 개수와 행렬 열 개수 일치)
            self.results["C11_Index_Sync"] = "PASS" if len(points) == incidence.shape[1] else "FAIL"

            # 3. 기저(Base) 검증 (제약 5, 6)
            base_G = pd.read_csv(self.paths['base_matrix'], header=None).values
            # 제약 5: 모든 원소가 F_q 범위 내인가
            self.results["C5_Field_Range"] = "PASS" if (base_G < q).all() and (base_G >= 0).all() else "FAIL"
            # 제약 6: Full Rank 여부 (단순 랭크 체크)
            rank = np.linalg.matrix_rank(base_G.astype(float))
            self.results["C6_Base_Full_Rank"] = "PASS" if rank == base_G.shape[0] else "FAIL"

            # 4. 제한(Bounds) 검증 (제약 7, 8, 12)
            with open(self.paths['bounds'], 'r') as f:
                bounds = json.load(f)
            u_p, lambda_p = bounds['u_p'], bounds['lambda_p']
            
            # 제약 7, 8, 12: u_p <= lambda_p 및 비음의 정수
            is_valid_bounds = True
            for i in u_p:
                up_val, lp_val = u_p[i], lambda_p[i]
                if not (isinstance(up_val, int) and isinstance(lp_val, int)): is_valid_bounds = False; break
                if up_val < 0 or lp_val < 0 or up_val > lp_val: is_valid_bounds = False; break
            self.results["C7_8_12_Bounds_Logic"] = "PASS" if is_valid_bounds else "FAIL"

        except Exception as e:
            self.results["Error"] = str(e)

    def print_report(self):
        print("\n=== 데이터셋 무결성 테스트 보고서 ===")
        for test, result in self.results.items():
            print(f"{test.ljust(25)}: {result}")

# 사용 예시
paths = {
    'config': 'config.json',
    'points': 'points.csv',
    'incidence_npy': 'incidence.npy',
    'base_matrix': 'base_matrix.csv',
    'bounds': 'bounds.json'
}

validator = DatasetValidator(paths)
validator.check_file_exists()
validator.validate_dataset()
validator.print_report()