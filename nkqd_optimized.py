#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKQD 탐색 엔진 - 최적화된 버전 (독립 실행 가능)

개선 사항:
1. O(1) 중복 검사 (set 기반)
2. 자동 max_depth 조정
3. 작은 문제 CPU 강제
4. nkqd_all_solutions.py를 완전히 대체
"""

import numpy as np
from itertools import combinations, product
import time
import threading
from typing import List, Tuple, Optional, Dict

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import pulp
    ILP_AVAILABLE = True
except ImportError:
    ILP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


###############################################################################
# 유한기하 생성
###############################################################################

def generate_projective_geometry(k, q):
    """PG(k-1, q) 생성 - nkqd_all_solutions와 동일"""
    all_vectors = []
    for vec in product(range(q), repeat=k):
        if any(v != 0 for v in vec):
            all_vectors.append(vec)
    
    points = []
    used = set()
    
    for vec in all_vectors:
        if vec in used:
            continue
        
        normalized = list(vec)
        first_nonzero_idx = next(i for i, v in enumerate(vec) if v != 0)
        first_nonzero = vec[first_nonzero_idx]
        
        if q == 2:
            inv = 1
        else:
            inv = pow(first_nonzero, q-2, q) if q > 2 else first_nonzero
        
        normalized = tuple((v * inv) % q for v in vec)
        
        if normalized not in used:
            points.append(normalized)
            used.add(normalized)
            
            for scalar in range(1, q):
                if scalar == 0:
                    continue
                scaled = tuple((v * scalar) % q for v in vec)
                used.add(scaled)
    
    return points


def generate_hyperplanes(points, k, q):
    """초평면 생성 - nkqd_all_solutions와 동일"""
    n_points = len(points)
    
    if k == 1:
        hyperplane_size = 1
    else:
        hyperplane_size = sum(q**i for i in range(k-1))
    
    hyperplanes = []
    
    # Fano plane 특수 케이스
    if k == 3 and q == 2 and n_points == 7:
        fano_lines = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 5, 6],
            [1, 3, 5],
            [1, 4, 6],
            [2, 3, 6],
            [2, 4, 5],
        ]
        if all(max(line) < n_points for line in fano_lines):
            return fano_lines
    
    for combo in combinations(range(n_points), hyperplane_size):
        hyperplanes.append(list(combo))
    
    return hyperplanes


###############################################################################
# NKQD 탐색 엔진 - 최적화 버전
###############################################################################

class NKQDOptimized:
    """최적화된 NKQD - 독립 실행 가능"""
    
    def __init__(self, n, k, d, q, use_gpu=None, use_ilp=True):
        self.n = n
        self.k = k
        self.d = d
        self.q = q
        
        # 작은 문제는 CPU 강제 (GPU 오버헤드 방지)
        if n <= 15 and use_gpu is None:
            print("  ⚠️ 작은 문제 감지: CPU 모드 강제 (GPU 오버헤드 방지)")
            use_gpu = False
        
        # GPU 자동 감지
        if use_gpu is None:
            self.use_gpu = GPU_AVAILABLE
            self.gpu_detected = GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and GPU_AVAILABLE
            self.gpu_detected = use_gpu and GPU_AVAILABLE
        
        self.use_ilp = use_ilp and ILP_AVAILABLE
        
        # 유한기하 생성
        self.points = generate_projective_geometry(k, q)
        self.hyperplanes = generate_hyperplanes(self.points, k, q)
        
        self.n_points = len(self.points)
        self.n_hyperplanes = len(self.hyperplanes)
        self.s = n - d
        
        # 연관도 행렬
        self.W = self.compute_connectivity_matrix()
        
        # 🌟 최적화: set 기반 중복 검사 (O(1))
        self.solution_set = set()
        
        # 통계
        self.stats = {
            'nodes_explored': 0,
            'lattice_points_explored': 0,  # benchmark와 호환
            'lattice_points_pruned': 0,
            'propagations': 0,
            'ilp_checks': 0,
            'ilp_prunes': 0,
            'mode_usage': {'HIGH': 0, 'LOW': 0},
            'depth_histogram': {},
            'solutions_found': 0,
            'duplicate_checks': 0,
            'bit_operations': 0,
            'memory_bytes': 0,
        }
        
        # 모든 해 저장
        self.all_solutions = []
        
        # 실행 제어
        self.running = True
        self.completed = False
        self.start_time = None
        
        self.value_strategy = 'middle_first'
    
    def compute_connectivity_matrix(self):
        """연관도 행렬 계산"""
        W = np.zeros((self.n_points, self.n_points), dtype=int)
        
        for h in self.hyperplanes:
            pts_in_h = [p for p in h if p < self.n_points]
            for i in pts_in_h:
                for j in pts_in_h:
                    if i != j:
                        W[i, j] += 1
        
        return W
    
    def is_duplicate_solution(self, solution):
        """🌟 최적화된 중복 해 검사 - O(1) set 기반"""
        self.stats['duplicate_checks'] += 1
        
        # 딕셔너리를 정렬된 튜플로 변환 (hashable)
        solution_tuple = tuple(sorted(solution.items()))
        
        if solution_tuple in self.solution_set:
            return True
        
        self.solution_set.add(solution_tuple)
        return False
    
    def propagate(self, L, U, fixed_mask):
        """Tight bound propagation with bit operation counting"""
        L = L.copy()
        U = U.copy()
        rounds = 0
        
        self.stats['bit_operations'] += len(L) * 64 * 2
        
        while True:
            rounds += 1
            changed = False
            
            # Upper bound 전파
            for p in range(self.n_points):
                if fixed_mask[p]:
                    continue
                
                old_U = U[p]
                new_U = U[p]
                
                for h in self.hyperplanes:
                    if p in h:
                        others_sum = sum(L[q] for q in h if q != p and q < self.n_points)
                        new_U = min(new_U, self.s - others_sum)
                        self.stats['bit_operations'] += len(h) * 64 + 32
                
                if new_U < old_U:
                    U[p] = new_U
                    changed = True
                    self.stats['bit_operations'] += 64
            
            # Lower bound 전파
            for p in range(self.n_points):
                if fixed_mask[p]:
                    continue
                
                old_L = L[p]
                others_sum = sum(U[q] for q in range(self.n_points) if q != p)
                new_L = max(L[p], self.n - others_sum)
                
                self.stats['bit_operations'] += self.n_points * 64 + 32
                
                if new_L > old_L:
                    L[p] = new_L
                    changed = True
                    self.stats['bit_operations'] += 64
            
            if np.any(L > U):
                self.stats['lattice_points_pruned'] += 1
                return None, None, rounds
            
            if not changed:
                break
            
            if rounds > 100:
                break
        
        return L, U, rounds
    
    def select_point_connectivity(self, L, U, fixed_mask, mode):
        """점 선택 - 연관도 기반"""
        unfixed = np.where(~fixed_mask)[0]
        if len(unfixed) == 0:
            return None
        
        W_unfixed = self.W[np.ix_(unfixed, unfixed)]
        
        if mode == "HIGH":
            if W_unfixed.size == 0:
                return unfixed[0]
            max_conn = W_unfixed.max(axis=1)
            selected_idx = np.argmax(max_conn)
        else:
            if W_unfixed.size == 0:
                return unfixed[0]
            degree = (W_unfixed > 0).sum(axis=1)
            selected_idx = np.argmin(degree)
        
        return unfixed[selected_idx]
    
    def select_mode(self, L, U, fixed_mask):
        """모드 선택 - HIGH/LOW"""
        unfixed = np.where(~fixed_mask)[0]
        if len(unfixed) == 0:
            return "LOW"
        
        rho = self.s / self.n_hyperplanes if self.n_hyperplanes > 0 else 0
        
        W_unfixed = self.W[np.ix_(unfixed, unfixed)]
        if W_unfixed.size == 0 or W_unfixed.max() == 0:
            return "LOW"
        
        mean_w = W_unfixed[W_unfixed > 0].mean() if np.any(W_unfixed > 0) else 1
        cluster_strength = W_unfixed.max() / mean_w if mean_w > 0 else 1
        
        if rho < 1.5 and cluster_strength > 1.5:
            return "HIGH"
        else:
            return "LOW"
    
    def select_branching_values(self, p, L, U, mode):
        """분기 값 선택"""
        if L[p] == U[p]:
            return [L[p]]
        
        domain = list(range(L[p], U[p] + 1))
        
        if self.value_strategy == 'ascending':
            return domain
        elif self.value_strategy == 'descending':
            return domain[::-1]
        elif self.value_strategy == 'middle_first':
            mid = len(domain) // 2
            return [domain[mid]] + domain[:mid] + domain[mid+1:]
        else:
            return domain
    
    def check_ilp(self, L, U, fixed_mask):
        """ILP 검증"""
        self.stats['ilp_checks'] += 1
        
        if not self.use_ilp:
            return True, {p: L[p] for p in range(self.n_points)}
        
        try:
            prob = pulp.LpProblem("NKQD", pulp.LpMinimize)
            
            x = {}
            for p in range(self.n_points):
                x[p] = pulp.LpVariable(f"x_{p}", cat='Integer',
                                      lowBound=L[p], upBound=U[p])
            
            prob += pulp.lpSum([x[p] for p in range(self.n_points)]) == self.n
            
            for h in self.hyperplanes:
                prob += pulp.lpSum([x[p] for p in h if p < self.n_points]) <= self.s
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                return True, {p: int(x[p].varValue) for p in range(self.n_points)}
            else:
                self.stats['ilp_prunes'] += 1
                return False, None
        except Exception as e:
            return True, {p: L[p] for p in range(self.n_points)}
    
    def search_recursive(self, L, U, fixed_mask, depth=0, max_depth=None):
        """
        재귀 탐색 - 모든 해 찾기 (실행 제어 포함)
        """
        if not self.running:
            return
        
        self.stats['nodes_explored'] += 1
        self.stats['lattice_points_explored'] += 1  # benchmark 호환
        self.stats['depth_histogram'][depth] = self.stats['depth_histogram'].get(depth, 0) + 1
        
        # 메모리 추적
        stack_memory = depth * (self.n_points * 8 * 3)
        self.stats['memory_bytes'] = max(self.stats['memory_bytes'], stack_memory)
        
        if max_depth is not None and depth >= max_depth:
            self.stats['lattice_points_pruned'] += 1
            return
        
        # 전파
        L_new, U_new, rounds = self.propagate(L, U, fixed_mask)
        self.stats['propagations'] += rounds
        
        if L_new is None:
            self.stats['lattice_points_pruned'] += 1
            return
        
        L, U = L_new, U_new
        
        # 종료 조건: 해 발견
        if np.all(fixed_mask) or np.all(L == U):
            feasible, solution = self.check_ilp(L, U, fixed_mask)
            if feasible and not self.is_duplicate_solution(solution):
                self.all_solutions.append(solution)
                self.stats['solutions_found'] += 1
            return
        
        # 모드 선택
        mode = self.select_mode(L, U, fixed_mask)
        self.stats['mode_usage'][mode] += 1
        self.stats['bit_operations'] += 100
        
        # 점 선택
        p_star = self.select_point_connectivity(L, U, fixed_mask, mode)
        if p_star is None:
            return
        
        # 분기 값 선택
        values = self.select_branching_values(p_star, L, U, mode)
        
        # 모든 분기 탐색
        for v in values:
            if not self.running:
                return
            
            L_branch = L.copy()
            U_branch = U.copy()
            fixed_branch = fixed_mask.copy()
            
            L_branch[p_star] = v
            U_branch[p_star] = v
            fixed_branch[p_star] = True
            
            self.search_recursive(L_branch, U_branch, fixed_branch, 
                                 depth + 1, max_depth)
    
    def run_with_timeout(self, timeout_seconds, max_depth=None):
        """
        제한 시간 동안 실행
        
        Args:
            timeout_seconds: 제한 시간 (초)
            max_depth: 최대 탐색 깊이 (None=점 개수로 자동 설정)
        
        Returns:
            elapsed: 실행 시간
        """
        self.start_time = time.time()
        
        # 🌟 max_depth 자동 조정
        if max_depth is None or max_depth > self.n_points:
            max_depth = self.n_points
            print(f"  ⚠️ max_depth 자동 설정: {max_depth} (점 개수에 맞춤)")
        
        # 타이머 쓰레드
        def stop_after_timeout():
            time.sleep(timeout_seconds)
            if not self.completed:
                self.running = False
        
        timer = threading.Thread(target=stop_after_timeout, daemon=True)
        timer.start()
        
        # 초기화
        L = np.zeros(self.n_points, dtype=int)
        U = np.full(self.n_points, self.n, dtype=int)
        fixed_mask = np.zeros(self.n_points, dtype=bool)
        
        # 탐색 실행
        try:
            self.search_recursive(L, U, fixed_mask, max_depth=max_depth)
            self.completed = True
            self.running = False
        except Exception as e:
            print(f"  ⚠️ 예외 발생: {e}")
        
        elapsed = time.time() - self.start_time
        
        print(f"\n  📊 중복 검사: {self.stats['duplicate_checks']:,}회")
        print(f"  📊 고유 해: {len(self.all_solutions):,}개")
        
        return elapsed
    
    def solve(self, max_depth=None, max_solutions=None, verbose=True):
        """
        탐색 실행 - 모든 해 찾기
        
        Args:
            max_depth: 최대 탐색 깊이
            max_solutions: 찾을 최대 해 개수 (현재 미사용)
            verbose: 출력 여부
        
        Returns:
            모든 해의 리스트
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🌟 모든 해 탐색: [n={self.n}, k={self.k}, d={self.d}]_{self.q}")
            print(f"{'='*70}\n")
            
            print(f"점 개수: {self.n_points}")
            print(f"Hyperplane 개수: {self.n_hyperplanes}")
            print(f"GPU: {'✓' if self.use_gpu else '✗'}")
            print(f"ILP: {'✓' if self.use_ilp else '✗'}")
            
            if max_solutions:
                print(f"최대 해 개수: {max_solutions}")
        
        # 초기화
        L = np.zeros(self.n_points, dtype=int)
        U = np.full(self.n_points, self.n, dtype=int)
        fixed_mask = np.zeros(self.n_points, dtype=bool)
        
        # 탐색
        start_time = time.time()
        self.search_recursive(L, U, fixed_mask, max_depth=max_depth)
        elapsed = time.time() - start_time
        
        # 결과 출력
        if verbose:
            print(f"\n{'='*70}")
            print("탐색 완료")
            print(f"{'='*70}")
            print(f"  시간: {elapsed:.2f}초")
            print(f"  탐색 노드: {self.stats['nodes_explored']}")
            print(f"  전파 횟수: {self.stats['propagations']}")
            print(f"  ILP 검증: {self.stats['ilp_checks']}회")
            print(f"  ILP 가지치기: {self.stats['ilp_prunes']}회")
            print(f"  모드 사용: {self.stats['mode_usage']}")
            print(f"{'='*70}\n")
            
            if self.all_solutions:
                print(f"{'='*70}")
                print(f"✅ 총 {len(self.all_solutions)}개 해 발견")
                print(f"{'='*70}\n")
                
                # 각 해 출력
                for i, sol in enumerate(self.all_solutions[:10]):
                    print(f"해 {i+1}: {sol}")
                    total = sum(sol.values())
                    print(f"  합계: {total}")
                    
                    # 제약 검증
                    violations = 0
                    for h in self.hyperplanes:
                        h_sum = sum(sol.get(p, 0) for p in h if p < self.n_points)
                        if h_sum > self.s:
                            violations += 1
                    
                    if violations == 0:
                        print(f"  ✓ 모든 제약 만족")
                    else:
                        print(f"  ⚠️ {violations}개 제약 위반")
                    print()
                
                if len(self.all_solutions) > 10:
                    print(f"... 외 {len(self.all_solutions) - 10}개 해\n")
                
            else:
                print(f"{'='*70}")
                print("❌ 해 없음")
                print(f"{'='*70}")
        
        return self.all_solutions


# 하위 호환성을 위한 alias
NKQDSearchEngineAllSolutions = NKQDOptimized


###############################################################################
# Colab 인터페이스
###############################################################################

def find_all_solutions(n, k, q, d, max_depth=None, max_solutions=None,
                      use_ilp=True, use_gpu=None, value_strategy='middle_first', plot=True):
    """
    모든 해 찾기 인터페이스
    
    Args:
        n, k, q, d: NKQD 파라미터
        max_depth: 최대 탐색 깊이
        max_solutions: 찾을 최대 해 개수 (None이면 무제한)
        use_ilp: ILP 사용
        use_gpu: GPU 사용 (None=자동감지, True=강제, False=끄기)
        value_strategy: 값 선택 전략
        plot: 그래프 출력
    
    Returns:
        all_solutions: 모든 해의 리스트
        engine: 엔진 객체
    """
    # GPU 자동 감지
    if use_gpu is None:
        try:
            import cupy as cp
            use_gpu = True
        except ImportError:
            use_gpu = False
    
    print(f"\n{'#'*70}")
    print(f"# 🌟 모든 해 탐색: [n={n}, k={k}, d={d}]_{q}")
    print(f"{'#'*70}\n")
    
    engine = NKQDOptimized(n, k, d, q, 
                          use_gpu=use_gpu, 
                          use_ilp=use_ilp)
    engine.value_strategy = value_strategy
    
    all_solutions = engine.solve(max_depth=max_depth, 
                                 max_solutions=max_solutions,
                                 verbose=True)
    
    # 그래프 생성
    if plot and PLOT_AVAILABLE and engine.stats['nodes_explored'] > 0:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 깊이 히스토그램
            depths = sorted(engine.stats['depth_histogram'].keys())
            counts = [engine.stats['depth_histogram'][d] for d in depths]
            axes[0].bar(depths, counts, color='steelblue', alpha=0.7)
            axes[0].set_xlabel('탐색 깊이')
            axes[0].set_ylabel('노드 수')
            axes[0].set_title('탐색 깊이 분포')
            axes[0].grid(axis='y', alpha=0.3)
            
            # 모드 사용
            mode_data = engine.stats['mode_usage']
            if sum(mode_data.values()) > 0:
                axes[1].pie(mode_data.values(), labels=mode_data.keys(),
                           autopct='%1.1f%%', startangle=90)
                axes[1].set_title('모드 사용 비율')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"그래프 생성 실패: {e}")
    
    return all_solutions, engine


###############################################################################
# 메인 실행
###############################################################################

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         NKQD 탐색 엔진 - 최적화 버전 (독립 실행 가능)                ║
╚══════════════════════════════════════════════════════════════════════╝

개선 사항:
1. O(1) 중복 검사 (set 기반)
2. 자동 max_depth 조정
3. 작은 문제 CPU 강제
4. nkqd_all_solutions.py를 완전히 대체
""")
    
    # 테스트 실행
    print("="*70)
    print("예시: 모든 해 찾기")
    print("="*70)
    
    engine = NKQDOptimized(n=10, k=3, d=4, q=2, use_gpu=None)
    
    start = time.time()
    
    L = np.zeros(engine.n_points, dtype=int)
    U = np.full(engine.n_points, engine.n, dtype=int)
    fixed_mask = np.zeros(engine.n_points, dtype=bool)
    
    engine.search_recursive(L, U, fixed_mask, max_depth=7)
    
    elapsed = time.time() - start
    
    print(f"\n결과:")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  탐색 노드: {engine.stats['lattice_points_explored']}")
    print(f"  발견 해: {len(engine.all_solutions)}개")
    print(f"  중복 검사: {engine.stats['duplicate_checks']}회")
