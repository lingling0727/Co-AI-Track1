import json
import os
from collections import Counter

class SolutionAnalyzer:
    def __init__(self, dataset_dir, solutions_file='solutions_n153.json'):
        self.dataset_dir = dataset_dir
        self.solutions_path = os.path.join(dataset_dir, solutions_file)
        self.solutions = []

    def load_solutions(self):
        if not os.path.exists(self.solutions_path):
            print(f"Error: Solutions file not found at {self.solutions_path}")
            print("Please run 'kurz_solver.py' first to generate solutions.")
            return False
        
        with open(self.solutions_path, 'r') as f:
            self.solutions = json.load(f)
        print(f"Loaded {len(self.solutions)} solutions from {self.solutions_path}")
        return True

    def analyze(self):
        if not self.load_solutions():
            return

        print("\n=== Solution Classification (Invariant Analysis) ===")
        groups = {}
        
        for idx, sol in enumerate(self.solutions):
            x_vals = sol['x']
            s_h_vals = sol['s_h']
            
            # Invariant 1: Multiplicity Profile (counts of 0s, 1s, 2s)
            # 점들의 중복도 분포 (예: 2가 몇 개인지)
            mult_counts = tuple(sorted(Counter(x_vals).items()))
            
            # Invariant 2: Hyperplane Sum Profile (Spectrum)
            # 초평면 가중치 분포 (코드의 Weight Distribution과 직결됨)
            # w(H) = n - S_H 이므로 S_H 분포가 같으면 가중치 분포도 같음
            sum_profile = tuple(sorted(s_h_vals))
            
            # Combined Signature (이 값이 다르면 확실히 비동형임)
            signature = (mult_counts, sum_profile)
            
            if signature not in groups:
                groups[signature] = []
            groups[signature].append(idx)
            
        print(f"Total Unique Classes Found: {len(groups)}")
        
        for i, (sig, indices) in enumerate(groups.items()):
            mult_counts, sum_profile = sig
            print(f"\n[Class {i+1}] Count: {len(indices)}")
            print(f"  - Multiplicity Profile (Value, Count): {mult_counts}")
            print(f"  - Hyperplane Sums (Min, Max): {min(sum_profile)}, {max(sum_profile)}")
            print(f"  - Representative Solution Index: {indices[0]}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'kurz_dataset_n153')
    
    analyzer = SolutionAnalyzer(dataset_path)
    analyzer.analyze()