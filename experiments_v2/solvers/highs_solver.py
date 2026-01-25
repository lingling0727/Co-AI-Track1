import numpy as np
import json
import os
import time
import sys

# highspy가 설치되어 있는지 확인
try:
    import highspy
except ImportError:
    print("Error: 'highspy' library is not installed.")
    print("Please install it using: pip install highspy")
    sys.exit(1)

class HighsMIPSolver:
    """
    Sascha Kurz (2024) 모델을 위한 HiGHS 기반 MIP 솔버.
    지능형 데이터셋을 활용하여 Method 1(Watched-Hyperplane)과 Method 2(RCUB)의 
    수학적 원리를 MIP 제약 조건으로 변환하여 구현함.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.load_data()

    def load_data(self):
        print(f"Loading dataset from {self.dataset_path}...")
        
        # 1. Config 로드
        with open(os.path.join(self.dataset_path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        self.n = self.config['n']
        self.k = self.config['k']
        self.q = self.config['q']
        self.allowed_capacities = sorted(list(self.config['allowed_capacities_s_h']))
        
        # 2. Incidence Matrix 로드 및 언패킹
        incidence_path = os.path.join(self.dataset_path, 'incidence_packed.npy')
        if os.path.exists(incidence_path):
            packed = np.load(incidence_path)
            # packbits는 uint8로 압축하므로, unpack 후 원본 크기에 맞게 자름
            # incidence_matrix는 (num_points, num_hypers) 형태이나, 
            # 사영 기하에서 점과 초평면의 개수는 같으므로 N x N 행렬임.
            unpacked = np.unpackbits(packed, axis=1)
            
            # points.csv의 행 개수를 통해 정확한 N을 확인
            # (unpackbits는 8의 배수로 패딩되므로 잘라내야 함)
            # 여기서는 config의 정보나 points.npy를 로드하여 확인 가능하지만,
            # 간단히 bounds.json의 길이를 이용함.
            with open(os.path.join(self.dataset_path, 'bounds.json'), 'r') as f:
                self.bounds = json.load(f)
                self.lower_bounds = np.array(self.bounds['lower_bounds'], dtype=np.int32)
            
            self.num_points = len(self.lower_bounds)
            self.num_hypers = self.num_points # Self-dual
            
            self.incidence_matrix = unpacked[:self.num_points, :self.num_hypers]
        else:
            raise FileNotFoundError(f"Incidence matrix not found at {incidence_path}")

    def solve(self):
        print(f"Building MIP model for [n={self.n}, k={self.k}, q={self.q}]...")
        h = highspy.Highs()
        
        # --- 1. 변수 설정 ---
        # x_P: 각 점의 중복도 (Integer, 0 <= x_P <= 2)
        # bounds.json의 lower_bounds 적용
        # 인덱스: 0 ~ num_points - 1
        x_indices = np.arange(self.num_points)
        
        # y_{H,c}: 초평면 H가 용량 c를 선택했는지 여부 (Binary)
        # 인덱스: num_points ~ ...
        num_caps = len(self.allowed_capacities)
        num_y_vars = self.num_hypers * num_caps
        y_start_idx = self.num_points
        
        # 변수 추가 (x)
        # highspy는 벡터 입력을 지원하지 않는 경우가 많아 루프로 추가하거나 addVars 사용
        # 여기서는 명시적으로 추가함.
        for i in range(self.num_points):
            lb = float(self.lower_bounds[i])
            ub = 2.0 # 문제 조건에 따라 설정 (일반적으로 1 또는 2)
            h.addVar(lb, ub)
            h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
            
        # 변수 추가 (y)
        for i in range(num_y_vars):
            h.addVar(0.0, 1.0)
            h.changeColIntegrality(y_start_idx + i, highspy.HighsVarType.kInteger)

        # --- 2. 제약 조건 설정 (Method 1 & 2) ---
        
        # Constraint 1: Total Length (sum(x) = n)
        # 1.0 * x_0 + ... + 1.0 * x_{N-1} = n
        col_indices = list(range(self.num_points))
        coeffs = [1.0] * self.num_points
        h.addRow(float(self.n), float(self.n), len(col_indices), np.array(col_indices, dtype=np.int32), np.array(coeffs, dtype=np.float64))
        
        # Constraint 2: Incidence & Capacity
        # For each hyperplane H: sum(x in H) - sum(c * y_{H,c}) = 0
        print("Adding hyperplane constraints (Method 1 & 2)...")
        
        # 희소 행렬 최적화를 위해 루프를 돌며 행을 추가
        # (Highs는 내부적으로 이를 CSR로 변환하여 처리함)
        for h_idx in range(self.num_hypers):
            # 2-1. x 변수 항 (인시던스 행렬 활용)
            # 해당 초평면에 포함된 점들의 인덱스 찾기
            points_in_hyper = np.where(self.incidence_matrix[h_idx])[0]
            
            row_indices = list(points_in_hyper)
            row_values = [1.0] * len(points_in_hyper)
            
            # 2-2. y 변수 항 (-c * y_{H,c})
            # y 변수들은 y_start_idx부터 순차적으로 배치됨
            # y_{h_idx, 0}, y_{h_idx, 1}, ...
            base_y_idx = y_start_idx + (h_idx * num_caps)
            
            for c_idx, cap in enumerate(self.allowed_capacities):
                row_indices.append(base_y_idx + c_idx)
                row_values.append(-float(cap))
            
            # Add Row: LHS = 0
            h.addRow(0.0, 0.0, len(row_indices), np.array(row_indices, dtype=np.int32), np.array(row_values, dtype=np.float64))
            
            # Constraint 3: Uniqueness (sum(y_{H,c}) = 1)
            # 해당 초평면에 대해 단 하나의 용량만 선택되어야 함
            y_indices = [base_y_idx + c_idx for c_idx in range(num_caps)]
            y_coeffs = [1.0] * num_caps
            h.addRow(1.0, 1.0, len(y_indices), np.array(y_indices, dtype=np.int32), np.array(y_coeffs, dtype=np.float64))

        # --- 3. 솔버 실행 ---
        print("Starting HiGHS solver...")
        h.setOptionValue("presolve", "on")
        h.setOptionValue("time_limit", 300.0) # 5분 제한
        
        start_time = time.time()
        status = h.run()
        end_time = time.time()
        
        # --- 상세 리포트 출력 ---
        info = h.getInfo()
        model_status = h.getModelStatus()
        
        print("\nSolving report")
        print(f"  Status            {model_status}")
        print(f"  Primal bound      {info.objective_function_value}")
        print(f"  Dual bound        {getattr(info, 'mip_dual_bound', 'N/A')}")
        print(f"  Gap               {getattr(info, 'mip_gap', 'N/A')}")
        print(f"  Nodes             {getattr(info, 'mip_node_count', 0)}")
        print(f"  LP iterations     {info.simplex_iteration_count}")
        print(f"Solve Time: {end_time - start_time:.4f}s")
        
        # --- 4. 결과 출력 ---
        if model_status == highspy.HighsModelStatus.kOptimal:
            print("\n--- Solution Found ---")
            # 솔루션 추출
            solution = h.getSolution()
            x_vals = np.array(solution.col_value[:self.num_points])
            # 정수 근사 (부동소수점 오차 보정)
            x_counts = np.round(x_vals).astype(np.int32)
            
            print(f"x_counts (non-zero):")
            non_zero_indices = np.where(x_counts > 0)[0]
            for idx in non_zero_indices:
                print(f"  Point {idx}: {x_counts[idx]}")
            
            print(f"Total points: {np.sum(x_counts)}")
            return x_counts
        else:
            print("No feasible solution found or time limit reached.")
            return None

if __name__ == "__main__":
    # 실행 예: python highs_solver.py prop4
    prop_name = sys.argv[1] if len(sys.argv) > 1 else "prop1"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "datasets", prop_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: {dataset_dir} 경로가 존재하지 않음.")
    else:
        solver = HighsMIPSolver(dataset_dir)
        solver.solve()