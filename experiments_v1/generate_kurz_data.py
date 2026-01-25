import numpy as np
import pandas as pd
import json
import os

class KurzDatasetGenerator:
    def __init__(self):
        # Kurz 2024 Proposition 4 Benchmark Parameters
        self.q = 2
        self.n = 153
        self.k = 7
        self.d = 76
        self.delta = 4  # Divisibility constant
        
        # Output paths
        self.output_dir = 'kurz_dataset_n153'
        os.makedirs(self.output_dir, exist_ok=True)

    def dot_fq(self, v1, v2):
        """Fq 내적 연산 (Standard integer dot product mod q)"""
        return np.dot(v1, v2) % self.q

    def generate_geometry(self):
        """PG(k-1, q)의 점과 초평면 생성 (Standardized)"""
        points = []
        # k차원 벡터 생성 (0 벡터 제외)
        for i in range(1, self.q**self.k):
            # 정수를 k자리 q진수 리스트로 변환
            vec = []
            temp = i
            for _ in range(self.k):
                vec.insert(0, temp % self.q)
                temp //= self.q
            
            # 정규화: 첫 번째 0이 아닌 성분이 1이 되도록 스케일링
            first_nonzero = next(x for x in vec if x != 0)
            if first_nonzero != 1:
                # q=2일 때는 항상 1이지만, 일반화를 위해 남겨둠 (역원 곱셈 필요)
                pass 
            
            if vec not in points:
                points.append(vec)
        
        # 사전순 정렬
        points.sort()
        return points

    def create_dataset(self):
        print(f"Generating dataset for q={self.q}, n={self.n}, k={self.k}...")
        
        # 1. Geometry 생성
        points = self.generate_geometry()
        num_points = len(points)
        print(f"Generated {num_points} points in PG({self.k-1}, {self.q}).")

        # 2. Incidence Matrix 생성 (Points vs Hyperplanes)
        # PG(k-1, q)에서는 점과 초평면의 개수가 같고 self-dual임.
        incidence = np.zeros((num_points, num_points), dtype=np.int8)
        
        for i in range(num_points): # Hyperplane defined by vector points[i]
            for j in range(num_points): # Point points[j]
                if self.dot_fq(points[i], points[j]) == 0:
                    incidence[i, j] = 1 # 포함됨
                else:
                    incidence[i, j] = 0 # 포함되지 않음
        
        # 3. 파일 저장
        
        # (1) config.json
        # w_set: {76, 80, 92, 96, 100} (84, 88 제외)
        # RCUB 정교화를 위해 허용 용량(allowed_capacities)도 계산: n - w
        # 논문 Proposition 4에 따라 특정 가중치만 유효함
        w_set = [76, 80, 92, 96, 100]
        allowed_capacities = sorted([self.n - w for w in w_set], reverse=True)
        
        config = {
            "q": self.q,
            "n": self.n,
            "k": self.k,
            "d": self.d,
            "delta": self.delta,
            "w_set": w_set,
            "allowed_capacities": allowed_capacities,
            "extension_info": {
                "mode": "systematic_basis",
                "description": "Search starts from fixed standard basis (Identity matrix) as per Constraint 5."
            },
            "description": "Kurz (2024) Proposition 4: Binary [153, 7, 76]_2 code classification (2 solutions)"
        }
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # (2) points.csv
        points_df = pd.DataFrame({
            'p_id': range(num_points),
            'v_p': [",".join(map(str, p)) for p in points]
        })
        points_df.to_csv(os.path.join(self.output_dir, 'points.csv'), index=False)

        # (3) incidence_packed.npy (Bitset optimization)
        # int8 대신 비트 패킹을 사용하여 저장 공간 절약
        incidence_packed = np.packbits(incidence, axis=1)
        np.save(os.path.join(self.output_dir, 'incidence_packed.npy'), incidence_packed)

        # (4) bounds.json
        # Systematic Generator Matrix Constraint: Unit vectors must have u_p >= 1
        # k=7 이므로 단위 벡터 생성
        unit_vectors = []
        for i in range(self.k):
            vec = [0] * self.k
            vec[i] = 1
            unit_vectors.append(vec)
            
        u_p = {str(i): 0 for i in range(num_points)}
        
        for i, p in enumerate(points):
            if p in unit_vectors:
                u_p[str(i)] = 1

        bounds = {
            "u_p": u_p,
            "lambda_p": {str(i): 2 for i in range(num_points)} # lambda_max = 2
        }
        with open(os.path.join(self.output_dir, 'bounds.json'), 'w') as f:
            json.dump(bounds, f, indent=4)

        print(f"Dataset created successfully in '{self.output_dir}/'")

if __name__ == "__main__":
    generator = KurzDatasetGenerator()
    generator.create_dataset()