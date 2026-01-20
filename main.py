import os
import time
import random
import math
import ast
from collections import Counter
import csv
import datetime

# generate_dataset 모듈에서 데이터 로드 함수 임포트
try:
    from generate_dataset import load_dataset
except ImportError:
    # generate_dataset.py가 같은 경로에 없을 경우를 대비한 더미 함수
    def load_dataset(n, k, q):
        print(f"[Warning] generate_dataset module not found. Using dummy data for {n},{k},{q}.")
        return [(1,0,0), (0,1,0), (0,0,1)], []

# OR-Tools 임포트 (ILP 솔버)
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("[Warning] 'ortools' library not found. ILP solver will be skipped.")

class GaloisFieldHelper:
    """
    유한체 연산(내적)을 돕는 헬퍼 클래스
    """
    def __init__(self, q):
        self.q = q
        self.is_prime = self._check_prime(q)
        # GF(4), GF(8) 등을 위한 곱셈 테이블 (예시: GF(4) = {0,1,a,a+1} -> {0,1,2,3})
        # GF(8) = {0..7}, p(x) = x^3 + x + 1
        # p(x)=x^2+x+1을 기약다항식으로 사용
        self.mul_table = {}
        if q == 4:
            # 0*x=0, 1*x=x
            # 2*2=3 (a*a = a+1)
            # 2*3=1 (a*(a+1) = a^2+a = 1)
            # 3*3=2 ((a+1)*(a+1) = a^2+1 = a)
            self.mul_table = {
                (0,0):0, (0,1):0, (0,2):0, (0,3):0,
                (1,0):0, (1,1):1, (1,2):2, (1,3):3,
                (2,0):0, (2,1):2, (2,2):3, (2,3):1,
                (3,0):0, (3,1):3, (3,2):1, (3,3):2
            }
        elif q == 8:
            # GF(8) 로그/지수 테이블 (Primitive Poly: x^3 + x + 1)
            self.gf8_exp = [1, 2, 4, 3, 6, 7, 5] * 2  # 인덱스 오버플로우 방지용 2배
            self.gf8_log = {1:0, 2:1, 4:2, 3:3, 6:4, 7:5, 5:6}

    def _check_prime(self, n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def _mul_gf8(self, a, b):
        if a == 0 or b == 0:
            return 0
        # a * b = g^(log a + log b)
        return self.gf8_exp[self.gf8_log[a] + self.gf8_log[b]]

    def dot_product(self, v1, v2):
        """두 벡터 v1, v2의 내적을 계산하여 0인지(직교하는지) 반환"""
        if self.is_prime:
            # 소수 체: 일반적인 모듈러 연산
            val = sum(a * b for a, b in zip(v1, v2)) % self.q
            return val == 0
        elif self.q == 4:
            # GF(4) 연산 (덧셈은 XOR, 곱셈은 테이블)
            val = 0
            for a, b in zip(v1, v2):
                term = self.mul_table.get((a, b), 0)
                val = val ^ term # GF(2^m) 덧셈은 XOR
            return val == 0
        elif self.q == 8:
            # GF(8) 연산
            val = 0
            for a, b in zip(v1, v2):
                val ^= self._mul_gf8(a, b) # 덧셈은 XOR
            return val == 0
        else:
            # 그 외 합성수는 구현 복잡도를 위해 생략 (실제 연구 시 라이브러리 필요)
            # 임시로 정수 내적 사용
            print(f"[Warning] Dot product for composite q={self.q} is using simple modulo. Results may be inaccurate.")
            return (sum(a * b for a, b in zip(v1, v2)) % self.q) == 0

class CodeClassifierExperiment:
    def __init__(self, n, k, q, weights):
        self.n = n
        self.k = k
        self.q = q
        self.target_weights = set(map(int, weights.split(',')))
        self.points, _ = load_dataset(n, k, q)
        # reproducibility seed for this experiment instance
        self.seed = random.randint(0, 2**31 - 1)
        random.seed(self.seed)
        self.gf = GaloisFieldHelper(q)
        
        # 기하학적 구조 미리 계산 (Incidence Matrix)
        # Hyperplanes는 Projective Space의 점들과 1:1 대응 (Dual Space)
        print(f"[*] Pre-calculating Incidence Matrix for PG({k-1}, {q})...")
        self.hyperplanes = self.points # Self-dual for simplicity in standard basis
        self.incidence = [] # incidence[h_idx][p_idx] = 1 if point p is on hyperplane h
        
        for h_vec in self.hyperplanes:
            row = []
            for p_vec in self.points:
                if self.gf.dot_product(h_vec, p_vec):
                    row.append(1)
                else:
                    row.append(0)
            self.incidence.append(row)
        print(f"[*] Incidence Matrix Built: {len(self.incidence)}x{len(self.incidence[0]) if self.incidence else 0}")
        # record convenient metadata
        self.points_count = len(self.points)
        self.incidence_shape = (len(self.incidence), len(self.incidence[0]) if self.incidence else 0)

    def solve_ilp(self, allowed_indices=None):
        """
        General ILP 접근법: OR-Tools 사용
        """
        if not ORTOOLS_AVAILABLE:
            return {"status": "Skipped (ortools not installed)", "time": 0}

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return {"status": "Solver Not Found", "time": 0}

        start_time = time.time()

        # allowed_indices가 주어지면 변수 공간을 축소
        if allowed_indices is None:
            indices = list(range(len(self.points)))
        else:
            indices = sorted(list(allowed_indices))

        # 변수: 각 선택된 점(point)이 부호에 몇 번 포함되는지 (x_i >= 0, Integer)
        # 변수 이름은 원래 점 인덱스를 사용하여 혼동을 줄임
        x = [solver.IntVar(0, self.n, f'x_{orig_i}') for orig_i in indices]

        # 제약 1: 전체 길이 = n
        solver.Add(solver.Sum(x) == self.n)

        # 제약 2: 무게 조건 (Weight Constraints)
        # 각 초평면 h에 대해, 무게 w = n - k_h 가 Target_Weights에 속해야 함.
        # k_h = sum(incidence[h][i] * x[i])
        possible_weights = list(self.target_weights)
        
        for h_idx in range(len(self.hyperplanes)):
            # k_h은 축소된 변수 공간(allowed indices)에 대해 계산
            k_h = solver.Sum(self.incidence[h_idx][orig_i] * x[var_idx] for var_idx, orig_i in enumerate(indices))
            
            # w_h = n - k_h
            # 이진 변수를 도입하여 w_h가 possible_weights 중 하나임을 표현
            b_vars = [solver.IntVar(0, 1, f'b_{h_idx}_{w}') for w in possible_weights]
            solver.Add(solver.Sum(b_vars) == 1) # 오직 하나의 무게만 선택
            
            # 선택된 무게와 실제 무게가 같아야 함
            weighted_sum = solver.Sum(b_vars[j] * possible_weights[j] for j in range(len(possible_weights)))
            solver.Add(weighted_sum == self.n - k_h)

        # 목적 함수: Feasibility 문제이므로 임의 설정
        solver.Minimize(0)

        print(f"[*] ILP Solving started (Vars: {solver.NumVariables()}, Constraints: {solver.NumConstraints()})...")
        status = solver.Solve()
        end_time = time.time()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            solution = [int(v.solution_value()) for v in x]
            # 변수 인덱스 -> 원래 점 인덱스로 매핑
            result_points = {orig_i: cnt for orig_i, cnt in zip(indices, solution) if cnt > 0}
            return {"status": "Feasible", "time": end_time - start_time, "solution": result_points, "allowed_count": len(indices)}
        else:
            status_str = "Infeasible" if status == pywraplp.Solver.INFEASIBLE else "Unsolved"
            return {"status": status_str, "time": end_time - start_time, "allowed_count": len(indices)}

    def _calculate_cost_for_heuristic(self, sol_indices):
        """
        주어진 해(점 인덱스 리스트)에 대한 비용(위반된 제약조건 수)을 계산합니다.
        """
        if not sol_indices:
            return len(self.hyperplanes)

        counts = Counter(sol_indices)
        violation_count = 0
        for h_idx in range(len(self.hyperplanes)):
            # k_h: 이 초평면 위에 있는 해의 점들의 개수
            k_h = sum(count for p_idx, count in counts.items() if self.incidence[h_idx][p_idx])
            weight = self.n - k_h
            if weight not in self.target_weights and weight != 0:
                violation_count += 1
        return violation_count

    def _generate_greedy_initial_solution(self):
        """
        탐욕적(Greedy) 구성 휴리스틱을 사용하여 초기 해를 생성합니다.
        매 단계에서 비용을 가장 많이 줄이는 점을 선택하여 추가합니다.
        """
        print("    [Heuristic] Generating initial solution with Greedy method...")
        num_points = len(self.points)
        solution_indices = []
        
        # k_h_values[h]는 현재 해에서 h번째 초평면 위에 있는 점의 수를 저장합니다.
        k_h_values = [0] * len(self.hyperplanes)
        
        for i in range(self.n):
            best_point_to_add = -1
            min_cost = float('inf')

            # 모든 점을 후보로 하여 최적의 점을 찾음
            for p_idx in range(num_points):
                # 이 점을 추가했을 경우의 비용을 증분적으로 계산
                cost = 0
                for h_idx in range(len(self.hyperplanes)):
                    # 만약 p_idx를 추가하면 k_h가 어떻게 변하는가?
                    new_k_h = k_h_values[h_idx] + self.incidence[h_idx][p_idx]
                    
                    # 최종 길이 n을 기준으로 무게 계산
                    weight = self.n - new_k_h
                    if weight not in self.target_weights and weight != 0:
                        cost += 1
                
                if cost < min_cost:
                    min_cost = cost
                    best_point_to_add = p_idx
            
            if best_point_to_add != -1:
                # 찾은 최적의 점을 해에 추가하고 k_h 값을 업데이트
                solution_indices.append(best_point_to_add)
                for h_idx in range(len(self.hyperplanes)):
                    k_h_values[h_idx] += self.incidence[h_idx][best_point_to_add]
            else:
                # 예외 상황 (일어나면 안 됨): 임의의 점 추가
                p_idx = random.randint(0, num_points - 1)
                solution_indices.append(p_idx)
                for h_idx in range(len(self.hyperplanes)):
                    k_h_values[h_idx] += self.incidence[h_idx][p_idx]

            if (i + 1) % 10 == 0 or i == self.n - 1:
                print(f"      [Greedy] ... built {i+1}/{self.n} points, best cost for this step: {min_cost}")
        
        print(f"    [Heuristic] Greedy initial solution generated.")
        return solution_indices

    def solve_heuristic(self, max_iter=10000, temp=100.0, cool_rate=0.995):
        """
        Heuristic 접근법: Simulated Annealing (담금질 기법)
        """
        start_time = time.time()
        
        # 초기 해 생성: 랜덤 방식에서 탐욕(Greedy) 방식으로 변경
        num_points = len(self.points)
        current_solution_indices = self._generate_greedy_initial_solution()
        
        current_cost = self._calculate_cost_for_heuristic(current_solution_indices)
        best_cost = current_cost
        best_solution = list(current_solution_indices)
        
        print(f"    [Heuristic] Initial cost from Greedy solution: {current_cost}")
        temperature = temp
        
        for i in range(max_iter):
            if best_cost == 0: break
            
            # 이웃 해 생성: 점 하나를 랜덤하게 다른 점으로 교체
            new_solution = list(current_solution_indices)
            idx_to_change = random.randint(0, self.n - 1)
            new_val = random.randint(0, num_points - 1)
            new_solution[idx_to_change] = new_val
            
            new_cost = self._calculate_cost_for_heuristic(new_solution)
            
            # Metropolis Criterion
            delta = new_cost - current_cost
            if delta < 0 or (temperature > 0 and random.random() < math.exp(-delta / temperature)):
                current_solution_indices = new_solution
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution_indices)
            
            temperature *= cool_rate
            
            if i % 1000 == 0:
                print(f"    [Heuristic] Iter {i}, Best Cost: {best_cost}, Temp: {temperature:.2f}")

        end_time = time.time()
        
        status = "Success" if best_cost == 0 else "Fail"
        return {
            "status": status,
            "time": end_time - start_time,
            "final_cost": best_cost,
            "solution": Counter(best_solution) if best_cost == 0 else None,
            # echo back parameters for logging
            "max_iter": max_iter,
            "temp": temp,
            "cool_rate": cool_rate,
            "seed": self.seed
        }

    def solve_hybrid(self):
        """
        Hybrid 접근법: Heuristic으로 탐색 공간을 축소(Pruning)한 후 ILP 수행
        """
        start_time = time.time()
        print("  > [Hybrid] Phase 1: Running Heuristic for Pruning...")
        
        # Phase 1: 짧게 휴리스틱 실행하여 유망한 점(Candidate Points) 식별
        # 반복 횟수를 줄여서 속도 확보 (예: 2000회)
        heu_res = self.solve_heuristic(max_iter=2000)
        
        if heu_res['status'] == "Success":
            print("  > [Hybrid] Heuristic found optimal solution directly!")
            return {"status": "Optimal (Heuristic)", "time": time.time() - start_time, "solution": heu_res['solution'], "allowed_count": len(self.points)}
        
        # Phase 2: Variable Pruning
        # 휴리스틱 결과에서 선택된 점들(Core Set) + 랜덤하게 일부 점(Exploration Set) 추가
        core_indices = set(heu_res['solution'].keys()) if heu_res['solution'] else set()
        
        # 전체 점의 30% 정도만 남기고 나머지는 Pruning (단, Core Set은 무조건 포함)
        num_total = len(self.points)
        num_keep = max(len(core_indices), int(num_total * 0.3))
        
        allowed_indices = set(core_indices)
        while len(allowed_indices) < num_keep:
            allowed_indices.add(random.randint(0, num_total - 1))
            
        print(f"  > [Hybrid] Phase 2: Running ILP on reduced space ({len(allowed_indices)}/{num_total} points)...")
        
        # 축소된 공간에서 ILP 실행
        ilp_res = self.solve_ilp(allowed_indices=allowed_indices)
        
        end_time = time.time()
        return {"status": f"Hybrid-{ilp_res['status']}", "time": end_time - start_time, "solution": ilp_res.get('solution'), "allowed_count": ilp_res.get('allowed_count')}


def main():
    # 실험 파라미터 파일 경로
    param_file = "experiment_parameters.txt"
    
    if not os.path.exists(param_file):
        print(f"'{param_file}' not found. Please create it or run generate_dataset.py first.")
        return

    print("=== Code Classification Experiment: ILP vs Heuristic ===")
    
    # CSV 결과 파일 설정: 매 실행 시 기존 파일을 덮어쓰고 헤더를 새로 작성
    results_csv = "experiment_results.csv"

    with open(results_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Timestamp", "n", "k", "q", "Weights",
            "Points", "Incidence_Size", "Allowed_Variables_Count", "Seed",
            "ILP_Status", "ILP_Time",
            "Heuristic_Status", "Heuristic_Time", "Heuristic_Cost",
            "Heuristic_MaxIter", "Heuristic_Temp", "Heuristic_CoolRate",
            "Hybrid_Status", "Hybrid_Time"
        ]
        # extrasaction='ignore' prevents errors when row dicts have extra keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        # 항상 헤더를 다시 작성
        writer.writeheader()

        with open(param_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # 쉼표가 무게 집합 {} 안에도 있으므로, 앞의 3개 쉼표까지만 분리
                parts = line.split(",", 3)
                n = int(parts[0].strip())
                k = int(parts[1].strip())
                q = int(parts[2].strip())
                weights = parts[3].strip().replace("{", "").replace("}", "")
                
                print(f"\n[Experiment] n={n}, k={k}, q={q}, Weights={{{weights}}}")
                
                try:
                    experiment = CodeClassifierExperiment(n, k, q, weights)
                except Exception as e:
                    print(f"  [Error] {e}")
                    continue
                
                # 1. Run ILP
                print("  > Running General ILP...")
                ilp_res = experiment.solve_ilp()
                print(f"  [ILP Result] Status: {ilp_res['status']}, Time: {ilp_res['time']:.4f}s")
                
                # 2. Run Heuristic
                print("  > Running Heuristic (Simulated Annealing)...")
                heu_res = experiment.solve_heuristic(max_iter=5000)
                print(f"  [Heuristic Result] Status: {heu_res['status']}, Cost: {heu_res.get('final_cost')}, Time: {heu_res['time']:.4f}s")

                # 3. Run Hybrid
                print("  > Running Hybrid Algorithm (Heuristic + ILP)...")
                hyb_res = experiment.solve_hybrid()
                print(f"  [Hybrid Result] Status: {hyb_res['status']}, Time: {hyb_res['time']:.4f}s")

                # 결과 CSV 저장
                writer.writerow({
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "n": n, "k": k, "q": q, "Weights": f"{{{weights}}}",
                    "Points": experiment.points_count,
                    "Incidence_Size": f"{experiment.incidence_shape[0]}x{experiment.incidence_shape[1]}",
                    "Allowed_Variables_Count": hyb_res.get('allowed_count'),
                    "Seed": experiment.seed,
                    "ILP_Status": ilp_res['status'],
                    "ILP_Time": round(ilp_res['time'], 4),
                    "Heuristic_Status": heu_res['status'],
                    "Heuristic_Time": round(heu_res['time'], 4),
                    "Heuristic_Cost": heu_res.get('final_cost'),
                    "Heuristic_MaxIter": heu_res.get('max_iter'),
                    "Heuristic_Temp": heu_res.get('temp'),
                    "Heuristic_CoolRate": heu_res.get('cool_rate'),
                    "Hybrid_Status": hyb_res['status'],
                    "Hybrid_Time": round(hyb_res['time'], 4)
                })
                csvfile.flush() # 데이터 유실 방지를 위해 즉시 기록

if __name__ == "__main__":
    main()