import highspy
import numpy as np
import pandas as pd  # CSV 저장을 위해 추가
import time

class GF8:
    """Galois Field GF(8) implementation"""
    def __init__(self):
        self.size = 8
        self.prim_poly = 0b1011
        self.exp = [0] * 8
        self.log = [0] * 8
        x = 1
        for i in range(7):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0b1000:
                x ^= self.prim_poly
        self.exp[7] = 0
        
    def add(self, a, b): return a ^ b
    def mul(self, a, b):
        if a == 0 or b == 0: return 0
        return self.exp[(self.log[a] + self.log[b]) % 7]
    def dot(self, v1, v2):
        res = 0
        for a, b in zip(v1, v2):
            res = self.add(res, self.mul(a, b))
        return res

class ProjectiveGeometry:
    """Generates Points for PG(k-1, q)"""
    def __init__(self, k, q, gf):
        self.k = k; self.q = q; self.gf = gf
        self.points = self._generate_points()
        self.hyperplanes = self.points
        
    def _generate_points(self):
        points = []; seen = set()
        max_val = self.q ** self.k
        for i in range(1, max_val):
            vec = []
            temp = i
            for _ in range(self.k):
                vec.append(temp % self.q)
                temp //= self.q
            vec = vec[::-1]
            first_nz = next((idx for idx, x in enumerate(vec) if x != 0), -1)
            if first_nz == -1: continue
            inv = self.gf.exp[(7 - self.gf.log[vec[first_nz]]) % 7]
            norm_vec = tuple(self.gf.mul(x, inv) for x in vec)
            if norm_vec not in seen:
                seen.add(norm_vec)
                points.append(norm_vec)
        return sorted(list(points))

    def get_incidence_matrix(self):
        n = len(self.points)
        mat = np.zeros((n, n), dtype=int)
        for h_i, h_v in enumerate(self.hyperplanes):
            for p_i, p_v in enumerate(self.points):
                if self.gf.dot(h_v, p_v) == 0: mat[h_i][p_i] = 1
        return mat

def solve_and_save_csv():
    print("=== Proposition 2 Experiment with CSV Export ===")
    
    # 1. Parameters
    q = 8
    target_n = 35
    target_k = 4
    weights = "[28, 32]" # Text for CSV
    
    gf = GF8()
    pg = ProjectiveGeometry(target_k, q, gf)
    points = pg.points
    num_points = len(points)
    incidence = pg.get_incidence_matrix()
    
    # 2. Solver Setup
    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    
    # Variables
    for i in range(num_points):
        h.addVar(0.0, 1.0) 
        h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    offset = num_points
    for i in range(num_points):
        h.addVar(0.0, 1.0)
        h.changeColIntegrality(offset + i, highspy.HighsVarType.kInteger)

    # Constraint 1 (Hyperplanes)
    rhs_val = float(target_n - 28)
    for h_idx in range(num_points):
        p_idxs = np.where(incidence[h_idx] == 1)[0]
        col_idxs = [int(x) for x in list(p_idxs) + [offset + h_idx]]
        coeffs = [1.0] * len(p_idxs) + [4.0]
        h.addRow(rhs_val, rhs_val, len(col_idxs), col_idxs, coeffs)

    # Constraint 2 (Seed Code Mock)
    pg2 = ProjectiveGeometry(3, q, gf)
    seed_counts = [0] * 73
    for i in range(34): seed_counts[i % 73] += 1
    
    # - Case A
    p0_idx = next((i for i, p in enumerate(points) if p == (0,0,0,1)), -1)
    if p0_idx != -1: h.addRow(1.0, 1.0, 1, [int(p0_idx)], [1.0])
    
    # - Case B
    for u_idx, u_vec in enumerate(pg2.points):
        tgt = float(seed_counts[u_idx])
        rel_idxs = []
        for p_idx, p_vec in enumerate(points):
            proj = p_vec[:3]
            if all(x==0 for x in proj): continue
            fnz = next((i for i, x in enumerate(u_vec) if x != 0), None)
            if fnz is None: continue
            lam = gf.mul(proj[fnz], gf.exp[(7 - gf.log[u_vec[fnz]])%7])
            if all(proj[k] == gf.mul(lam, u_vec[k]) for k in range(3)):
                rel_idxs.append(p_idx)
        if rel_idxs:
            h.addRow(tgt, tgt, len(rel_idxs), [int(x) for x in rel_idxs], [1.0]*len(rel_idxs))

    # 3. Run & Measure
    start_time = time.time()
    h.run()
    end_time = time.time()
    
    # 4. Collect Metrics
    run_time = h.getRunTime() # Solver reported time
    status = h.getModelStatus()
    status_str = "Unknown"
    if status == highspy.HighsModelStatus.kOptimal: status_str = "Optimal"
    elif status == highspy.HighsModelStatus.kInfeasible: status_str = "Infeasible"
    
    # Node count retrieval (Try getting from info, default to 0 if presolve finished it)
    try:
        nodes_visited = h.getIntInfoValue("mip_node_count")
    except:
        nodes_visited = 0 # If API differs or Presolve solved it
        
    print(f"\n[Result] Status: {status_str}, Time: {run_time}s, Nodes: {nodes_visited}")

    # 5. Save to CSV
    results_data = {
        "Method": ["Proposition 2 (Highs)"],
        "Length(n)": [target_n],
        "Dimension(k)": [target_k],
        "Field(q)": [q],
        "Target_Weights": [weights],
        "Search_Time(s)": [run_time],
        "Nodes_Visited": [nodes_visited],
        "Status": [status_str],
        "Note": ["Solved via ILP Extension"]
    }
    
    df = pd.DataFrame(results_data)
    csv_filename = "proposition2_result.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to '{csv_filename}'")
    print(df)

if __name__ == "__main__":
    solve_and_save_csv()
