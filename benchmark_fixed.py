"""
NKQD 탐색 방법 비교 벤치마크 (최종 수정)

핵심 수정:
1. 방법 2와 NKQD가 정확히 같은 사영공간 사용
2. 두 방법 모두 max_depth 제한 없음 (완전 탐색)
3. 중복 검사 방식 동일 (set 기반)
4. 방법 2는 CPU, NKQD는 GPU 사용 (알고리즘 + 하드웨어 최적화 비교)
5. 두 방법 모두 정확한 해를 저장 (예: {0: 2, 1: 1, 2: 0, ...})
"""

import numpy as np
from itertools import product
import time
import threading
import psutil
import os

# NKQD import
from nkqd_optimized import NKQDOptimized

###############################################################################
# 사영공간 생성 (NKQD와 동일)
###############################################################################

def generate_projective_geometry_unified(k, q):
    """통일된 사영공간 생성 - NKQD와 동일"""
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


def generate_hyperplanes_unified(points, k, q):
    """통일된 초평면 생성 - NKQD와 동일"""
    n_points = len(points)
    
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
    
    # 일반적인 경우
    hyperplanes = []
    if k == 1:
        hyperplane_size = 1
    else:
        hyperplane_size = sum(q**i for i in range(k-1))
    
    from itertools import combinations
    for combo in combinations(range(n_points), hyperplane_size):
        hyperplanes.append(list(combo))
    
    return hyperplanes


###############################################################################
# 방법 2: 기본 Phase 0 (통일된 사영공간)
###############################################################################

class BasicPhase0Unified:
    """방법 2: NKQD와 동일한 사영공간 사용"""
    
    def __init__(self, n, k, q, d):
        self.n = n
        self.k = k
        self.q = q
        self.d = d
        
        # NKQD와 동일한 사영공간 생성
        self.points = generate_projective_geometry_unified(k, q)
        self.hyperplanes = generate_hyperplanes_unified(self.points, k, q)
        self.m = len(self.points)
        
        # set 기반 중복 검사 (NKQD와 동일)
        self.solution_set = set()
        
        # 통계
        self.stats = {
            'lattice_points_explored': 0,
            'lattice_points_pruned': 0,
            'solutions_found': 0,
            'bit_operations': 0,
            'memory_bytes': 0,
            'duplicate_checks': 0,
        }
        
        self.solutions = []
        self.running = True
        self.completed = False
        self.start_time = None
    
    def check_hyperplane_constraints(self, solution):
        """hyperplane 제약 검증"""
        self.stats['bit_operations'] += len(self.hyperplanes) * 10
        
        for h_points in self.hyperplanes:
            h_sum = sum(solution[p] for p in h_points)
            self.stats['bit_operations'] += len(h_points) * 2
            
            if h_sum > self.n - self.d:
                return False
        return True
    
    def is_duplicate_solution(self, solution):
        """set 기반 중복 검사 (NKQD와 동일)"""
        self.stats['duplicate_checks'] += 1
        
        solution_tuple = tuple(sorted(solution.items()))
        
        if solution_tuple in self.solution_set:
            return True
        
        self.solution_set.add(solution_tuple)
        return False
    
    def enumerate_solutions(self, remaining, pos, current):
        """재귀적 열거 (제한 없음)"""
        if not self.running:
            return
        
        self.stats['lattice_points_explored'] += 1
        self.stats['bit_operations'] += 5
        
        stack_memory = pos * self.m * 8
        self.stats['memory_bytes'] = max(self.stats['memory_bytes'], stack_memory)
        
        if pos == self.m:
            if remaining == 0:
                if self.check_hyperplane_constraints(current):
                    # 딕셔너리로 저장 (NKQD와 동일)
                    solution = {p: current[p] for p in range(self.m)}
                    if not self.is_duplicate_solution(solution):
                        self.solutions.append(solution)
                        self.stats['solutions_found'] += 1
                else:
                    self.stats['lattice_points_pruned'] += 1
            else:
                self.stats['lattice_points_pruned'] += 1
            return
        
        # 분기 (제한 없음)
        for val in range(remaining + 1):
            if not self.running:
                return
            current[pos] = val
            self.enumerate_solutions(remaining - val, pos + 1, current)
            self.stats['bit_operations'] += 3
    
    def run_with_timeout(self, timeout_seconds):
        """제한 시간 동안 실행"""
        self.start_time = time.time()
        
        def stop_after_timeout():
            time.sleep(timeout_seconds)
            if not self.completed:
                self.running = False
        
        timer = threading.Thread(target=stop_after_timeout, daemon=True)
        timer.start()
        
        current = [0] * self.m
        try:
            self.enumerate_solutions(self.n, 0, current)
            self.completed = True
            self.running = False
        except Exception as e:
            print(f"  ⚠️ 예외 발생: {e}")
        
        elapsed = time.time() - self.start_time
        
        print(f"\n  📊 중복 검사: {self.stats['duplicate_checks']:,}회")
        print(f"  📊 고유 해: {len(self.solutions):,}개")
        
        return elapsed


###############################################################################
# NKQD (제한 없음)
###############################################################################

class NKQDUnlimited(NKQDOptimized):
    """NKQD - max_depth 제한 제거"""
    
    def run_with_timeout(self, timeout_seconds, max_depth=None):
        """제한 시간 동안 실행 (max_depth 제한 없음)"""
        self.start_time = time.time()
        
        # max_depth 제한 제거
        print(f"  ⚠️ max_depth 제한 없음 (완전 탐색)")
        
        def stop_after_timeout():
            time.sleep(timeout_seconds)
            if not self.completed:
                self.running = False
        
        timer = threading.Thread(target=stop_after_timeout, daemon=True)
        timer.start()
        
        L = np.zeros(self.n_points, dtype=int)
        U = np.full(self.n_points, self.n, dtype=int)
        fixed_mask = np.zeros(self.n_points, dtype=bool)
        
        try:
            self.search_recursive(L, U, fixed_mask, max_depth=None)  # 제한 없음
            self.completed = True
            self.running = False
        except Exception as e:
            print(f"  ⚠️ 예외 발생: {e}")
        
        elapsed = time.time() - self.start_time
        
        print(f"\n  📊 중복 검사: {self.stats.get('duplicate_checks', 0):,}회")
        print(f"  📊 고유 해: {len(self.all_solutions):,}개")
        
        return elapsed


###############################################################################
# 유틸리티
###############################################################################

def format_number(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


def format_bytes(b):
    if b >= 1024**3:
        return f"{b/1024**3:.2f} GB"
    elif b >= 1024**2:
        return f"{b/1024**2:.2f} MB"
    elif b >= 1024:
        return f"{b/1024:.2f} KB"
    else:
        return f"{b} B"


###############################################################################
# 벤치마크
###############################################################################

def benchmark_comparison_fixed(n, k, q, d, duration_minutes=10):
    """
    수정된 벤치마크
    - 동일한 사영공간 사용
    - max_depth 제한 없음 (완전 탐색)
    - 방법 2: CPU (기본 열거)
    - NKQD: GPU (tight-bound propagation + 연관도 기반 분기)
    - 두 방법 모두 정확한 해를 저장 (예: {0: 2, 1: 1, 2: 0, ...})
    """
    print("="*70)
    print(f"🔬 벤치마크 비교: CPU vs GPU")
    print(f"="*70)
    print(f"파라미터: [n={n}, k={k}, d={d}]_{q}")
    print(f"실행 시간: 최대 {duration_minutes}분")
    print(f"⚠️ 방법 2 (CPU) vs NKQD (GPU)")
    print("="*70)
    
    duration_seconds = duration_minutes * 60
    interval_seconds = 60
    process = psutil.Process(os.getpid())
    
    print("\n📊 시스템 정보:")
    print(f"  CPU 코어: {psutil.cpu_count()}")
    print(f"  메모리: {format_bytes(psutil.virtual_memory().total)}")
    
    # 방법 2
    print(f"\n{'='*70}")
    print("🔵 방법 2: 기본 Phase 0 - CPU")
    print(f"{'='*70}")
    
    method2 = BasicPhase0Unified(n, k, q, d)
    
    print(f"  초기 설정:")
    print(f"    점 개수: {method2.m}")
    print(f"    초평면 개수: {len(method2.hyperplanes)}")
    
    def monitor_method2():
        for minute in range(1, duration_minutes + 1):
            time.sleep(interval_seconds)
            if not method2.running:
                break
            elapsed = time.time() - method2.start_time
            print(f"\n  ⏱️ {minute}분: 격자점 {format_number(method2.stats['lattice_points_explored'])}, "
                  f"해 {len(method2.solutions)}")
    
    monitor_thread2 = threading.Thread(target=monitor_method2, daemon=True)
    monitor_thread2.start()
    
    print(f"\n  🚀 시작...")
    elapsed2 = method2.run_with_timeout(duration_seconds)
    
    print(f"\n  {'✅ 정상 완료!' if method2.completed else '⏰ 시간 초과'} ({elapsed2:.1f}초)")
    print(f"    최종 해: {len(method2.solutions):,}개")
    
    # NKQD
    print(f"\n{'='*70}")
    print("🟢 NKQD All Solutions - GPU")
    print(f"{'='*70}")
    
    nkqd = NKQDUnlimited(n, k, d, q, use_gpu=True)  # GPU 사용
    
    print(f"  초기 설정:")
    print(f"    점 개수: {nkqd.n_points}")
    print(f"    초평면 개수: {nkqd.n_hyperplanes}")
    print(f"    GPU: {'✓' if nkqd.use_gpu else '✗'}")
    
    def monitor_nkqd():
        for minute in range(1, duration_minutes + 1):
            time.sleep(interval_seconds)
            if not nkqd.running:
                break
            elapsed = time.time() - nkqd.start_time
            print(f"\n  ⏱️ {minute}분: 격자점 {format_number(nkqd.stats['lattice_points_explored'])}, "
                  f"해 {len(nkqd.all_solutions)}")
    
    monitor_thread_nkqd = threading.Thread(target=monitor_nkqd, daemon=True)
    monitor_thread_nkqd.start()
    
    print(f"\n  🚀 시작...")
    elapsed_nkqd = nkqd.run_with_timeout(duration_seconds)
    
    print(f"\n  {'✅ 정상 완료!' if nkqd.completed else '⏰ 시간 초과'} ({elapsed_nkqd:.1f}초)")
    print(f"    최종 해: {len(nkqd.all_solutions):,}개")
    
    # 비교
    print(f"\n{'='*70}")
    print("📊 최종 비교")
    print(f"{'='*70}")
    
    print(f"\n발견한 해:")
    print(f"  방법 2 (CPU): {len(method2.solutions):,}개")
    print(f"  NKQD (GPU): {len(nkqd.all_solutions):,}개")
    
    # 해의 예시 출력 (처음 3개)
    if len(method2.solutions) > 0:
        print(f"\n  📋 방법 2 해 예시 (처음 3개):")
        for i, sol in enumerate(method2.solutions[:3]):
            print(f"    해 {i+1}: {sol}")
    
    if len(nkqd.all_solutions) > 0:
        print(f"\n  📋 NKQD 해 예시 (처음 3개):")
        for i, sol in enumerate(nkqd.all_solutions[:3]):
            print(f"    해 {i+1}: {sol}")
    
    if len(method2.solutions) == len(nkqd.all_solutions):
        print(f"\n  ✅ 개수 일치!")
        
        # 내용 비교
        if method2.completed and nkqd.completed:
            method2_set = set(tuple(sorted(sol.items())) for sol in method2.solutions)
            nkqd_set = set(tuple(sorted(sol.items())) for sol in nkqd.all_solutions)
            
            if method2_set == nkqd_set:
                print(f"  ✅ 내용 일치: 100% 정확도")
            else:
                print(f"  ⚠️ 내용 불일치!")
                # 차이점 분석
                only_method2 = method2_set - nkqd_set
                only_nkqd = nkqd_set - method2_set
                if only_method2:
                    print(f"    방법 2에만 있는 해: {len(only_method2)}개")
                if only_nkqd:
                    print(f"    NKQD에만 있는 해: {len(only_nkqd)}개")
    else:
        print(f"\n  ⚠️ 개수 불일치!")
        if len(method2.solutions) > len(nkqd.all_solutions):
            print(f"    방법 2가 {len(method2.solutions) - len(nkqd.all_solutions)}개 더 많음")
        else:
            print(f"    NKQD가 {len(nkqd.all_solutions) - len(method2.solutions)}개 더 많음")
    
    print(f"\n실행 시간:")
    print(f"  방법 2: {elapsed2:.1f}초")
    print(f"  NKQD: {elapsed_nkqd:.1f}초")
    if elapsed2 > 0 and elapsed_nkqd > 0:
        speedup = elapsed2 / elapsed_nkqd
        print(f"  속도: NKQD가 {speedup:.2f}x {'빠름' if speedup > 1 else '느림'}")
    
    print(f"\n격자점 탐색:")
    print(f"  방법 2: {format_number(method2.stats['lattice_points_explored'])}")
    print(f"  NKQD: {format_number(nkqd.stats['lattice_points_explored'])}")
    if method2.stats['lattice_points_explored'] > 0:
        reduction = (1 - nkqd.stats['lattice_points_explored'] / method2.stats['lattice_points_explored']) * 100
        print(f"  탐색 공간 축소: {reduction:.2f}%")
    
    print(f"\n{'='*70}")
    
    return {
        'method2': method2.stats,
        'nkqd': nkqd.stats,
        'method2_solutions': len(method2.solutions),
        'nkqd_solutions': len(nkqd.all_solutions),
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              NKQD 벤치마크 (CPU vs GPU)                              ║
╚══════════════════════════════════════════════════════════════════════╝

수정 사항:
- 동일한 사영공간 사용
- max_depth 제한 없음 (완전 탐색)
- 방법 2 (CPU): 기본 열거
- NKQD (GPU): tight-bound propagation + 연관도 기반 분기
- 두 방법 모두 정확한 해를 저장 (예: {0: 2, 1: 1, 2: 0, ...})

사용법:
    benchmark_comparison_fixed(n=10, k=3, q=2, d=4, duration_minutes=10)

""")
    
    result = benchmark_comparison_fixed(n=10, k=3, q=2, d=4, duration_minutes=2)
