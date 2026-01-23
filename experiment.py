import math

class LinearCodeClassifierParameters:
    def __init__(self,
                 code_properties: dict,       # 선형 코드의 수학적 제약 조건
                 efficiency_settings: dict,   # 격자점 열거 및 ILP 효율성 설정
                 resource_constraints: dict   # 실제 계산 시간 및 자원 제약
                 ):
        """
        선형 코드 분류 알고리즘의 파라미터를 초기화합니다.
        """
        self.code_properties = code_properties
        self.efficiency_settings = efficiency_settings
        self.resource_constraints = resource_constraints

        # 최종 파라미터 딕셔너리
        self.parameters = {}

        self._set_default_parameters()
        self._apply_code_properties()
        self._apply_efficiency_settings()
        self._apply_resource_constraints()

    def _set_default_parameters(self):
        """
        기본 파라미터 값을 설정합니다.
        """
        self.parameters = {
            # 격자점 열거 관련 기본 파라미터
            "lattice_max_depth": 15,            # 최대 탐색 깊이
            "lattice_pruning_strategy": "RCUB", # 가지치기 전략 (예: RCUB, B&B)
            "lattice_initial_bound_factor": 1.0, # 초기 상한 설정 계수

            # ILP 솔버 관련 기본 파라미터
            "ilp_solver_name": "gurobi",        # 사용할 ILP 솔버 (예: gurobi, cplex, glpk)
            "ilp_time_limit_sec": 300,          # 각 ILP 호출의 시간 제한 (초)
            "ilp_mip_gap": 0.001,               # ILP 최적성 갭 허용 오차 (0.1%)
            "ilp_feasibility_tolerance": 1e-6,  # ILP 제약 조건 위반 허용 오차

            # 일반 알고리즘 제어 파라미터
            "verbose_output": True,             # 상세 출력 여부
            "logging_level": "INFO",            # 로깅 레벨
            "max_candidate_extensions": 100000  # 확장 후보의 최대 수
        }

    def _apply_code_properties(self):
        """
        1. 선형 코드의 수학적 제약 조건을 기반으로 파라미터를 조정합니다.
        """
        n = self.code_properties.get("n", 66)
        k = self.code_properties.get("k", 5)
        q = self.code_properties.get("q", 4)
        weights = self.code_properties.get("weights", [])
        divisible_by_delta = self.code_properties.get("divisible_by_delta", False)
        min_weight_a = self.code_properties.get("min_weight_a", 1) # 논문의 'a' 파라미터
        max_weight_b = self.code_properties.get("max_weight_b", 100) # 논문의 'b' 파라미터
        delta_val = self.code_properties.get("delta", 1) # 논문의 'Delta' 파라미터

        # 코드 길이에 따라 격자 탐색 깊이 조정 (휴리스틱)
        self.parameters["lattice_max_depth"] = min(self.parameters["lattice_max_depth"], n // 2 + k)

        # 필드 크기 q에 따라 ILP의 변수/제약 조건 증가를 고려한 허용 오차 또는 시간 제한 조정
        if q > 2:
            self.parameters["ilp_mip_gap"] = max(self.parameters["ilp_mip_gap"], 0.005)
            # 수정: 수식 괄호 및 오타 수정 (* q/2) -> * (q / 2)
            self.parameters["ilp_time_limit_sec"] = min(self.parameters["ilp_time_limit_sec"] * (q / 2), 1800)

        # 가중치 제약 조건을 ILP 또는 격자 생성 시 반영
        self.parameters["code_weight_constraints"] = {
            "min_weight_a": min_weight_a,
            "max_weight_b": max_weight_b,
            "delta": delta_val,
            "weights_set": weights,
            "divisible_by_delta": divisible_by_delta
        }

        # 프로젝트성 코드 (M(P) in {0,1}) 여부에 따라 격자 변수 정의 방식 변경
        self.parameters["is_projective"] = self.code_properties.get("is_projective", True)

    def _apply_efficiency_settings(self):
        """
        2. 격자점 열거 및 ILP 솔버의 효율성 극대화를 위한 파라미터를 조정합니다.
        """
        pruning_factor = self.efficiency_settings.get("lattice_pruning_factor", 0.8)
        ilp_tolerance = self.efficiency_settings.get("ilp_solver_tolerance", 0.001)

        # 격자 가지치기 계수 적용
        self.parameters["lattice_pruning_threshold_factor"] = pruning_factor

        # ILP 솔버의 최적성 허용 오차 적용
        self.parameters["ilp_mip_gap"] = ilp_tolerance
        self.parameters["ilp_feasibility_tolerance"] = ilp_tolerance / 10 

        # 논문에서 언급된 'Phase 0'에 ILP를 사용하는지에 대한 플래그
        self.parameters["use_ilp_in_phase0"] = self.efficiency_settings.get("use_ilp_in_phase0", True)

        # 'Phase 0'에서 불필요한 확장 후보를 조기 제거하기 위한 추가 ILP 제약
        self.parameters["phase0_ilp_constraints"] = self.efficiency_settings.get("phase0_ilp_constraints", ["canonical_length_extension_check"])

    def _apply_resource_constraints(self):
        """
        3. 실제 계산 시간 및 자원의 제약을 기반으로 파라미터를 조정합니다.
        """
        max_time_sec = self.resource_constraints.get("max_computation_time_sec", 3600) 
        num_cores = self.resource_constraints.get("num_cores", 4) 

        # 총 계산 시간에 따라 ILP 개별 호출 시간 및 후보 수 제한 조정
        self.parameters["ilp_time_limit_sec"] = min(self.parameters["ilp_time_limit_sec"], max_time_sec / 10)

        # 병렬 처리 설정
        self.parameters["num_parallel_jobs"] = num_cores
        if num_cores > 1:
            self.parameters["lattice_parallelization_enabled"] = True
            self.parameters["ilp_solver_threads"] = num_cores 

    def get_parameters(self):
        """
        설정된 모든 파라미터를 반환합니다.
        """
        return self.parameters

# ==============================================================================
# 사용 예시
# ==============================================================================
if __name__ == "__main__":
    # 1. 선형 코드의 수학적 제약 조건 예시 (논문의 [66, 5, {48, 56}]4-code)
    my_code_properties = {
        "n": 66,
        "k": 5,
        "q": 4,
        "weights": [48, 56],
        "is_projective": True,
        "divisible_by_delta": True, # 논문의 예시에서는 pt-divisible (e.g., pt=8)
        "delta": 8, # 48과 56은 8로 나누어짐
        "min_weight_a": 6, # 48 / 8 = 6
        "max_weight_b": 7, # 56 / 8 = 7
    }

    # 2. 격자점 열거 및 ILP 솔버의 효율성 극대화를 위한 설정
    my_efficiency_settings = {
        "lattice_pruning_factor": 0.85, 
        "ilp_solver_tolerance": 0.0005, 
        "use_ilp_in_phase0": True,
        "phase0_ilp_constraints": ["canonical_length_extension_check", "gaps_in_weight_spectrum_check"]
    }

    # 3. 실제 계산 시간 및 자원의 제약
    my_resource_constraints = {
        "max_computation_time_sec": 7200, # 2시간
        "num_cores": 16,
        "max_memory_gb": 32
    }

    # 파라미터 객체 생성
    params_generator = LinearCodeClassifierParameters(
        my_code_properties,
        my_efficiency_settings,
        my_resource_constraints
    )

    # 설정된 파라미터 가져오기
    final_parameters = params_generator.get_parameters()

    print("--- 최종 설정된 파라미터 ---")
    for key, value in final_parameters.items():
        print(f"{key}: {value}")

    # 특정 파라미터 사용 예시
    print(f"\nILP 시간 제한: {final_parameters['ilp_time_limit_sec']} 초")
    print(f"격자 탐색 최대 깊이: {final_parameters['lattice_max_depth']}")
    print(f"Phase 0에서 ILP 사용 여부: {final_parameters['use_ilp_in_phase0']}")

    # 수정: 딕셔너리 접근 구문 및 오타 수정
    min_w = final_parameters['code_weight_constraints']['min_weight_a']
    delta = final_parameters['code_weight_constraints']['delta']
    print(f"코드의 최소 가중치 a*Delta: {min_w * delta}")
