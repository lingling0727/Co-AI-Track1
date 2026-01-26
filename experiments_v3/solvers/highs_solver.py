# solvers/highs_solver.py
import numpy as np
import json
import os
import time
import sys

try:
    import highspy
except ImportError:
    print("에러: 'highspy' 라이브러리가 설치되지 않았음.", file=sys.stderr)
    print("pip install highspy 명령어로 설치해주길 바람.", file=sys.stderr)
    sys.exit(1)

class HighsMIPSolver:
    """
    Sascha Kurz (2024) 논문 모델을 HiGHS로 푸는 MIP 솔버임.
    기하 데이터를 받아서 선형 부호를 찾는 MIP 문제를 구성하고 해결함.
    """

    def __init__(self, data=None, dataset_path=None):
        if data is not None:
            self._load_from_data_obj(data)
        elif dataset_path is not None:
            self._load_from_path(dataset_path)
        else:
            raise ValueError("데이터 객체 'data'나 파일 경로 'dataset_path' 중 하나는 꼭 줘야함.")

    def _load_from_data_obj(self, data: dict):
        """딕셔너리 객체에서 설정이랑 데이터를 불러옴."""
        print("메모리 객체에서 데이터 로딩 중...")
        self.n = data['n']
        self.k = data['k']
        self.q = data['q']
        self.allowed_capacities = data['allowed_capacities']
        self.num_points = data['num_points']
        self.num_hypers = data['num_hypers']
        self.incidence_matrix = data['incidence_matrix']
        self.lower_bounds = data['lower_bounds']
        print("데이터 로딩 성공.")

    def _load_from_path(self, dataset_path: str):
        """예전 버전 호환을 위해 파일 경로에서 데이터를 불러옴."""
        print(f"파일 경로 {dataset_path}에서 데이터셋 로딩 중...")
        with open(os.path.join(dataset_path, 'config.json'), 'r') as f:
            config = json.load(f)
        self.n = config['n']
        self.k = config['k']
        self.q = config['q']
        self.allowed_capacities = sorted(list(config['allowed_capacities_s_h']))
        
        with open(os.path.join(dataset_path, 'bounds.json'), 'r') as f:
            bounds = json.load(f)
            self.lower_bounds = np.array(bounds['lower_bounds'], dtype=np.int32)
        
        self.num_points = len(self.lower_bounds)
        self.num_hypers = self.num_points

        packed = np.load(os.path.join(dataset_path, 'incidence_packed.npy'))
        unpacked = np.unpackbits(packed, axis=1)
        self.incidence_matrix = unpacked[:self.num_points, :self.num_hypers]

    def solve(self):
        """MIP 모델 만들고 풀어서, 결과를 딕셔너리로 반환함."""
        print(f"\n[n={self.n}, k={self.k}, q={self.q}]에 대한 MIP 모델 빌드 중...")
        h = highspy.Highs()

        # 연구용으로 상세 로그 출력하게 함
        h.setOptionValue("output_flag", True)
        h.setOptionValue("log_to_console", True)

        # 1. 변수 설정
        num_x_vars = self.num_points
        h.addVars(num_x_vars, self.lower_bounds.astype(np.float64), np.full(num_x_vars, float(self.q - 1)))
        for i in range(num_x_vars):
            h.changeColIntegrality(i, highspy.HighsVarType.kInteger)

        num_caps = len(self.allowed_capacities)
        num_y_vars = self.num_hypers * num_caps
        y_start_idx = num_x_vars
        h.addVars(num_y_vars, np.zeros(num_y_vars), np.ones(num_y_vars))
        for i in range(num_y_vars):
            h.changeColIntegrality(y_start_idx + i, highspy.HighsVarType.kInteger)

        # 2. 제약 조건 설정
        # 제약 1: 전체 길이는 n (sum(x) = n)
        h.addRow(float(self.n), float(self.n), num_x_vars, np.arange(num_x_vars), np.ones(num_x_vars))
        
        print("초평면 및 유일성 제약 추가 중...")
        for h_idx in range(self.num_hypers):
            # 제약 2: 인시던스 & 용량 (sum_{P in H} x_P - sum_{c} c*y_{H,c} = 0)
            points_in_hyper = np.where(self.incidence_matrix[h_idx, :])[0]
            y_indices_for_h = y_start_idx + (h_idx * num_caps) + np.arange(num_caps)
            
            row_indices = np.concatenate([points_in_hyper, y_indices_for_h])
            row_values = np.concatenate([np.ones(len(points_in_hyper)), -np.array(self.allowed_capacities, dtype=float)])
            
            h.addRow(0.0, 0.0, len(row_indices), row_indices, row_values)
            
            # 제약 3: 유일성 (sum_c y_{H,c} = 1)
            h.addRow(1.0, 1.0, num_caps, y_indices_for_h, np.ones(num_caps))

        # 3. 솔버 실행
        print("\nHiGHS 솔버 실행 시작...")
        h.setOptionValue("presolve", "on")
        # 연구용으로 오래 돌릴 수 있게 시간 제한은 없음.

        start_time = time.time()
        status = h.run()
        solve_time = time.time() - start_time
        
        info = h.getInfo()
        model_status = h.getModelStatus()
        
        solution_info = {
            "model_status": str(model_status),
            "solve_time": solve_time,
            "primal_bound": info.objective_function_value,
            "dual_bound": info.mip_dual_bound,
            "gap": info.mip_gap,
            "node_count": info.mip_node_count,
            "solution_found": model_status == highspy.HighsModelStatus.kOptimal,
            "x_counts": None,
        }
        
        if solution_info["solution_found"]:
            solution = h.getSolution()
            x_vals = np.array(solution.col_value[:self.num_points])
            solution_info["x_counts"] = np.round(x_vals).astype(np.int32)
        
        return solution_info
