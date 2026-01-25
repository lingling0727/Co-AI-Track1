import json
import os
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from collections import Counter

class KurzSolver:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.config = None
        self.bounds = None
        self.incidence = None
        self.num_points = 0
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.dataset_dir}...")
        
        # 1. Load config
        with open(os.path.join(self.dataset_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # 2. Load bounds
        with open(os.path.join(self.dataset_dir, 'bounds.json'), 'r') as f:
            self.bounds = json.load(f)
            
        # 3. Load points to get exact count
        points_df = pd.read_csv(os.path.join(self.dataset_dir, 'points.csv'))
        self.num_points = len(points_df)
        
        # 4. Load and unpack incidence matrix (Optimization)
        packed_path = os.path.join(self.dataset_dir, 'incidence_packed.npy')
        packed_incidence = np.load(packed_path)
        
        # Unpack bits: (num_hyperplanes, packed_cols) -> (num_hyperplanes, num_points_padded)
        # numpy.packbits는 8비트 단위로 패킹하므로, 원본보다 열이 더 많을 수 있음
        unpacked = np.unpackbits(packed_incidence, axis=1)
        
        # Slice to actual number of points (remove padding)
        self.incidence = unpacked[:, :self.num_points]
        
        print(f"  - n: {self.config['n']}")
        print(f"  - Geometry: {self.num_points} points, {self.incidence.shape[0]} hyperplanes")
        print(f"  - Allowed Capacities (S_H): {self.config['allowed_capacities']}")

    def solve(self, solution_limit=None, output_file='solutions.json'):
        print("\nBuilding CP-SAT Model...")
        model = cp_model.CpModel()
        
        # 1. Define Variables x_P in [0, 2]
        x = []
        for i in range(self.num_points):
            # Apply bounds from bounds.json
            # u_p is lower bound (e.g., 1 for unit vectors)
            lb = self.bounds['u_p'].get(str(i), 0)
            # lambda_p is upper bound (2)
            ub = self.bounds['lambda_p'].get(str(i), 2)
            
            x.append(model.NewIntVar(lb, ub, f'x_{i}'))
            
        # 2. Total Sum Constraint: sum(x_P) = n = 153
        model.Add(sum(x) == self.config['n'])
        
        # 3. Hyperplane Constraints (RCUB Pruning Strategy)
        # S_H = sum(x_P for P in H)
        # S_H must be in allowed_capacities [77, 73, 61, 57, 53]
        # This acts as a strong pruning mechanism by restricting the domain of S_H
        allowed_caps = self.config['allowed_capacities']
        domain = cp_model.Domain.FromValues(allowed_caps)
        
        s_h_vars = []
        for h_idx in range(self.incidence.shape[0]):
            # Get indices of points in this hyperplane
            # incidence is 0/1 (uint8), so we find where it is 1
            p_indices = np.where(self.incidence[h_idx] == 1)[0]
            
            # Define S_H variable with restricted domain (Discrete Pruning)
            s_h = model.NewIntVarFromDomain(domain, f'S_H_{h_idx}')
            
            # Link S_H to sum of x_P
            model.Add(s_h == sum(x[p] for p in p_indices))
            s_h_vars.append(s_h)
            
        # 4. Solve
        print("Solving...")
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.log_search_progress = True  # 진행 상황 모니터링 활성화
        
        solution_printer = SolutionCollector(x, s_h_vars, solution_limit)
        status = solver.Solve(model, solution_printer)
        
        print(f"\nStatus: {solver.StatusName(status)}")
        print(f"Total solutions found: {solution_printer.solution_count}")
        
        # Save solutions to file
        if solution_printer.solutions:
            output_path = os.path.join(self.dataset_dir, output_file)
            print(f"Saving {len(solution_printer.solutions)} solutions to {output_path}...")
            with open(output_path, 'w') as f:
                json.dump(solution_printer.solutions, f, indent=2)
        
        # 5. Analyze and Classify Solutions
        if solution_printer.solution_count > 0:
            self.classify_solutions(solution_printer.solutions)

    def classify_solutions(self, solutions):
        print("\n=== Solution Classification (Coarse Invariant) ===")
        # Group by (Multiplicity Multiset, Hyperplane Sum Profile)
        groups = {}
        
        for idx, sol in enumerate(solutions):
            x_vals = sol['x']
            s_h_vals = sol['s_h']
            
            # Invariant 1: Multiset of multiplicities (e.g., counts of 0s, 1s, 2s)
            mult_counts = tuple(sorted(Counter(x_vals).items()))
            
            # Invariant 2: Sorted profile of hyperplane sums
            sum_profile = tuple(sorted(s_h_vals))
            
            signature = (mult_counts, sum_profile)
            
            if signature not in groups:
                groups[signature] = []
            groups[signature].append(idx)
            
        print(f"Number of non-isomorphic classes found: {len(groups)}")
        for i, (sig, indices) in enumerate(groups.items()):
            mult_counts, sum_profile = sig
            print(f"\nClass {i+1}: {len(indices)} solutions")
            print(f"  - Multiplicity Counts (Value, Count): {mult_counts}")
            print(f"  - Hyperplane Sums (Min, Max): {min(sum_profile)}, {max(sum_profile)}")

class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, s_h_vars, limit=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__x_vars = x_vars
        self.__s_h_vars = s_h_vars
        self.__limit = limit
        self.solution_count = 0
        self.solutions = []

    def on_solution_callback(self):
        self.solution_count += 1
        x_vals = [self.Value(v) for v in self.__x_vars]
        s_h_vals = [self.Value(v) for v in self.__s_h_vars]
        self.solutions.append({'x': x_vals, 's_h': s_h_vals})
        print(f"Solution {self.solution_count} found...", end='\r')
        
        if self.__limit and self.solution_count >= self.__limit:
            print(f"\nSolution limit ({self.__limit}) reached. Stopping search.")
            self.StopSearch()

if __name__ == "__main__":
    # Path to the dataset generated by generate_kurz_data.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'kurz_dataset_n153')
    
    if os.path.exists(dataset_path):
        solver = KurzSolver(dataset_path)
        # 너무 많은 해가 나올 경우를 대비해 1000개까지만 찾고 저장하도록 설정
        solver.solve(solution_limit=1000, output_file='solutions_n153.json')
    else:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run 'generate_kurz_data.py' first.")