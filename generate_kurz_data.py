import numpy as np
import pandas as pd
import json
import os

class KurzDatasetGenerator:
    def __init__(self):
        # Kurz 2024 Table 4 Benchmark Parameters
        self.q = 3
        self.n = 41
        self.k = 4
        self.d = 27
        self.delta = 9  # Divisibility constant
        
        # Output paths
        self.output_dir = 'kurz_dataset_n41'
        os.makedirs(self.output_dir, exist_ok=True)

    def dot_f3(self, v1, v2):
        """F3 내적 연산 (Standard integer dot product mod 3)"""
        return np.dot(v1, v2) % 3

    def generate_geometry(self):
        """PG(3, 3)의 점과 초평면 생성 (Standardized)"""
        points = []
        # 4차원 벡터 생성 (0,0,0,0 제외)
        for i in range(3**4):
            vec = [
                (i // 27) % 3,
                (i // 9) % 3,
                (i // 3) % 3,
                i % 3
            ]
            if sum(vec) == 0: continue
            
            # 정규화: 첫 번째 0이 아닌 성분이 1이 되도록 스케일링
            first_nonzero = next(x for x in vec if x != 0)
            if first_nonzero == 2:
                vec = [(2 * x) % 3 for x in vec]
            
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
        # PG(3,3)에서는 점과 초평면의 개수가 같고(40개), self-dual임.
        incidence = np.zeros((num_points, num_points), dtype=np.int8)
        
        for i in range(num_points): # Hyperplane defined by vector points[i]
            for j in range(num_points): # Point points[j]
                if self.dot_f3(points[i], points[j]) == 0:
                    incidence[i, j] = 1 # 포함됨
                else:
                    incidence[i, j] = 0 # 포함되지 않음
        
        # 3. 파일 저장
        
        # (1) config.json
        # w_set: 가능한 가중치 집합. w >= d 이고 w % delta == 0
        # n=41, d=27, delta=9 이므로 가능한 w는 27, 36 (45는 n 초과)
        # RCUB 정교화를 위해 허용 용량(allowed_capacities)도 계산: n - w
        w_set = sorted([w for w in range(self.d, self.n + 1) if w % self.delta == 0])
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
            "description": "Kurz (2024) Benchmark: Ternary [41, 4, 27]_3 code non-existence check"
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
        # int8 대신 비트 패킹을 사용하여 저장 공간 절약 (40x40 -> 40x5 bytes)
        incidence_packed = np.packbits(incidence, axis=1)
        np.save(os.path.join(self.output_dir, 'incidence_packed.npy'), incidence_packed)

        # (4) bounds.json
        # Systematic Generator Matrix Constraint: Unit vectors must have u_p >= 1
        unit_vectors = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
        u_p = {str(i): 0 for i in range(num_points)}
        
        for i, p in enumerate(points):
            if p in unit_vectors:
                u_p[str(i)] = 1

        bounds = {
            "u_p": u_p,
            "lambda_p": {str(i): self.n for i in range(num_points)}
        }
        with open(os.path.join(self.output_dir, 'bounds.json'), 'w') as f:
            json.dump(bounds, f, indent=4)

        print(f"Dataset created successfully in '{self.output_dir}/'")

if __name__ == "__main__":
    generator = KurzDatasetGenerator()
    generator.create_dataset()