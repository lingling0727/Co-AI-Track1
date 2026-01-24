import math

class Prop2Parameters:
    def __init__(self):
        # ---------------------------------------------------------
        # 1. 기본 파라미터 정의 (Proposition 2) 
        # ---------------------------------------------------------
        self.q = 8              # Field Size (GF(8))
        self.target_n = 35      # 목표 길이 (Length)
        self.target_k = 4       # 목표 차원 (Dimension)
        self.target_weights = [28, 32]  # 허용되는 가중치 집합 (Weights)
        
        # ---------------------------------------------------------
        # 2. 증명 전략: 확장(Extension) 설정 
        # Proposition 2는 [34, 3] 코드를 [35, 4]로 확장하여 존재 여부를 확인
        # ---------------------------------------------------------
        self.start_n = 34       # 시작 코드(Seed Code)의 길이
        self.start_k = 3        # 시작 코드의 차원
        self.r = self.target_n - self.start_n  # 추가해야 할 길이 (r=1)
        
        # ---------------------------------------------------------
        # 3. 수학적 제약 조건 파라미터 (논문 Eq 3-7 관련)
        # ---------------------------------------------------------
        # Divisor Delta: 가중치 28, 32는 모두 4의 배수임 (q=8, p=2 -> p^2=4)
        self.delta = 4          
        
        # Projective Code 여부 (True이면 각 열은 중복될 수 없음, x_P <= 1)
        self.is_projective = True 
        
        # 가중치 범위 (Min/Max Weight)
        self.min_weight = min(self.target_weights) # 28
        self.max_weight = max(self.target_weights) # 32

    def print_summary(self):
        print(f"=== Proposition 2 Parameter Setup ===")
        print(f"Code Type: [{self.target_n}, {self.target_k}, {self.target_weights}]_8")
        print(f"Strategy: Extend from [{self.start_n}, {self.start_k}] to add r={self.r} columns")
        print(f"Divisibility (Delta): {self.delta}")
        print(f"Projective: {self.is_projective}")

# 파라미터 인스턴스 생성 및 확인
params = Prop2Parameters()
params.print_summary()
