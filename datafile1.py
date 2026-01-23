import math

class DynamicCodeParameterGenerator:
    def __init__(self,
                 code_definition: dict,
                 optimization_goals: dict,
                 system_resources: dict):
        """
        선형 코드 분류를 위한 파라미터를 동적으로 생성합니다.
        Args:
            code_definition (dict): 분류 대상 선형 코드의 수학적 특성
            optimization_goals (dict): 성능 목표 (정확도 vs 속도)
            system_resources (dict): 시스템 자원 제약
        """
        self.code_definition = code_definition
        self.optimization_goals = optimization_goals
        self.system_resources = system_resources
        self.generated_parameters = {}

        # 실행 순서대로 파라미터 생성 및 조정
        self._initialize_base_parameters()
        self._adjust_for_mathematical_constraints()
        self._optimize_for_efficiency()
        self._constrain_by_resources()

    def _initialize_base_parameters(self):
        """
        기본 파라미터 설정
        """
        self.generated_parameters = {
            "ilp_solver_timeout_per_call_sec": 600,   # 10분
            "ilp_relative_gap_tolerance": 0.005,      # 0.5%
            "lattice_enumeration_max_depth": 20,
            "lattice_pruning_aggressiveness": 0.6,    # 0.0 ~ 1.0
            "phase2_check_granularity": "medium",
            "enable_pre_filtering_phase0": False,
            "parallel_processing_threads": 1,
            "max_candidate_extensions_phase1": 50000
        }

    def _adjust_for_mathematical_constraints(self):
        """
        수학적 제약 조건에 따른 조정
        """
        n = self.code_definition.get("n_length", 50)
        k = self.code_definition.get("k_dimension", 5)
        q = self.code_definition.get("q_field_size", 2)
        delta = self.code_definition.get("delta_divisibility", 1)
        a_min_weight_factor = self.code_definition.get("min_weight_a", 1)
        b_max_weight_factor = self.code_definition.get("max_weight_b", 100)

        # [수정됨] 복잡도 추정 로직 추가 (n * k * log(q) 정도의 휴리스틱 사용)
        estimated_complexity = n * k * math.log2(q if q > 0 else 2)
        
        # 탐색 깊이 조정 (최소 10, 최대 50)
        self.generated_parameters["lattice_enumeration_max_depth"] = int(min(50, max(10, estimated_complexity // 10 + 5)))

        # Field Size(q)에 따른 ILP 난이도 조정
        if q > 2:
            # q가 클수록 시간 제한을 늘림 (제곱근 비례)
            self.generated_parameters["ilp_solver_timeout_per_call_sec"] *= (q / 2) ** 0.5
            # 허용 오차 완화 (최대 1%)
            self.generated_parameters["ilp_relative_gap_tolerance"] = max(self.generated_parameters["ilp_relative_gap_tolerance"], 0.01)

        # 가중치 범위 밀도 계산
        weight_range_factor = (b_max_weight_factor - a_min_weight_factor + 1) / (delta if delta > 0 else 1)
        
        # 가중치 범위가 좁으면 해를 찾기 어려우므로 탐색을 더 넓게 설정
        if weight_range_factor < 5:
            self.generated_parameters["lattice_pruning_aggressiveness"] = min(self.generated_parameters["lattice_pruning_aggressiveness"], 0.4)
            self.generated_parameters["max_candidate_extensions_phase1"] = max(self.generated_parameters["max_candidate_extensions_phase1"], 1000000)

    def _optimize_for_efficiency(self):
        """
        효율성 목표(속도 vs 정확도)에 따른 조정
        """
        precision_priority = self.optimization_goals.get("precision_priority", 0.5)
        speed_priority = self.optimization_goals.get("speed_priority", 0.5)
        enable_phase0_ilp = self.optimization_goals.get("enable_phase0_ilp_optimization", True)

        # 정확도 우선 (Precision Priority > 0.7)
        if precision_priority > 0.7:
            # 갭을 줄여 더 정확한 해 찾기
            self.generated_parameters["ilp_relative_gap_tolerance"] = min(self.generated_parameters["ilp_relative_gap_tolerance"], 0.001)
            
            # [수정됨] 오타 수정: 가지치기 강도 약화 (더 꼼꼼하게 탐색)
            # 기존 식: "lattice_pruning_aggressiveness" * 1 - ...
            reduction_factor = (precision_priority - 0.7)
            self.generated_parameters["lattice_pruning_aggressiveness"] = max(
                self.generated_parameters["lattice_pruning_aggressiveness"] * (1.0 - reduction_factor), 
                0.3
            )
            self.generated_parameters["phase2_check_granularity"] = "high"

        # 속도 우선 (Speed Priority > 0.7)
        elif speed_priority > 0.7:
            # 갭을 늘려 속도 향상
            self.generated_parameters["ilp_relative_gap_tolerance"] = max(self.generated_parameters["ilp_relative_gap_tolerance"], 0.02)
            
            # [수정됨] 오타 수정: 가지치기 강도 강화 (과감하게 버림)
            increase_factor = (speed_priority - 0.7)
            self.generated_parameters["lattice_pruning_aggressiveness"] = min(
                self.generated_parameters["lattice_pruning_aggressiveness"] * (1.0 + increase_factor), 
                0.9
            )
            self.generated_parameters["phase2_check_granularity"] = "low"
            # ILP 타임아웃 단축
            self.generated_parameters["ilp_solver_timeout_per_call_sec"] = min(self.generated_parameters["ilp_solver_timeout_per_call_sec"], 120)

        # Phase 0 설정
        self.generated_parameters["enable_pre_filtering_phase0"] = enable_phase0_ilp

    def _constrain_by_resources(self):
        """
        자원 제약에 따른 조정
        """
        total_time_limit_sec = self.system_resources.get("total_time_limit_sec", 3600)
        available_cpu_cores = self.system_resources.get("available_cpu_cores", 4)

        # 총 시간 제한의 1% 또는 최소 30초를 개별 ILP 호출에 할당
        self.generated_parameters["ilp_solver_timeout_per_call_sec"] = min(
            self.generated_parameters["ilp_solver_timeout_per_call_sec"],
            max(30, total_time_limit_sec // 100)
        )

        # 스레드 수 제한 (최대 8개)
        self.generated_parameters["parallel_processing_threads"] = min(available_cpu_cores, 8)
        
        # 코어가 많으면 후보 탐색 수를 늘림
        if available_cpu_cores > 1:
            # [수정됨] float 결과를 int로 변환
            multiplier = available_cpu_cores / 2
            self.generated_parameters["max_candidate_extensions_phase1"] = int(
                self.generated_parameters["max_candidate_extensions_phase1"] * multiplier
            )

    def get_parameters(self):
        return self.generated_parameters

# ==============================================================================
# 사용 예시
# ==============================================================================

if __name__ == "__main__":
    # 1. 예시 코드 정의
    my_code_spec = {
        "n_length": 120,
        "k_dimension": 6,
        "q_field_size": 2,
        "min_weight_a": 10,
        "max_weight_b": 20,
        "delta_divisibility": 2
    }

    # 2. 목표 설정 (정확도와 속도의 균형)
    my_optimization_goals = {
        "precision_priority": 0.6,
        "speed_priority": 0.4,
        "enable_phase0_ilp_optimization": True
    }

    # 3. 자원 설정 (3시간, 12코어, 64GB)
    my_system_resources = {
        "total_time_limit_sec": 10800, 
        "available_cpu_cores": 12,
        "memory_gb": 64
    }

    # 생성기 실행
    param_generator = DynamicCodeParameterGenerator(
        my_code_spec,
        my_optimization_goals,
        my_system_resources
    )
    generated_params = param_generator.get_parameters()

    print("--- [기본 균형] 동적으로 생성된 파라미터 ---")
    for param, value in generated_params.items():
        print(f"- {param}: {value}")

    # 4. 속도 우선 모드 테스트
    print("\n--- [속도 우선] 파라미터 변경 결과 ---")
    speed_priority_goals = {
        "precision_priority": 0.2, 
        "speed_priority": 0.9,  # 매우 높은 속도 우선순위
        "enable_phase0_ilp_optimization": True
    }
    speed_params = DynamicCodeParameterGenerator(my_code_spec, speed_priority_goals, my_system_resources).get_parameters()
    
    # 주요 변경점 출력
    print(f"- 가지치기 강도 (aggressiveness): {speed_params['lattice_pruning_aggressiveness']:.2f} (높을수록 많이 버림)")
    print(f"- 검증 상세도 (granularity): {speed_params['phase2_check_granularity']}")
    print(f"- ILP 제한 시간: {speed_params['ilp_solver_timeout_per_call_sec']}초")
