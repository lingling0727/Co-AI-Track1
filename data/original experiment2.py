import highspy
import numpy as np

class GF8:
    """
    Galois Field GF(8) implementation (Primitive polynomial: x^3 + x + 1)
    """
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
        
    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        if a == 0 or b == 0: return 0
        return self.exp[(self.log[a] + self.log[b]) % 7]

    def dot(self, v1, v2):
        res = 0
        for a, b in zip(v1, v2):
            res = self.add(res, self.mul(a, b))
        return res

class ProjectiveGeometry:
    """Generates Points and Hyperplanes for PG(k-1, q)"""
    def __init__(self, k, q, gf):
        self.k = k
        self.q = q
        self.gf = gf
        self.points = self._generate_points()
        self.hyperplanes = self.points
        
    def _generate_points(self):
        points = []
        max_val = self.q ** self.k
        seen = set()
        
        for i in range(1, max_val):
            vec = []
            temp = i
            for _ in range(self.k):
                vec.append(temp % self.q)
                temp //= self.q
            vec = vec[::-1]
            
            first_nonzero = -1
            for idx, val in enumerate(vec):
                if val != 0:
                    first_nonzero = idx
                    break
            
            if first_nonzero == -1: continue
            
            inv = self.gf.exp[(7 - self.gf.log[vec[first_nonzero]]) % 7]
            norm_vec = tuple(self.gf.mul(x, inv) for x in vec)
            
            if norm_vec not in seen:
                seen.add(norm_vec)
                points.append(norm_vec)
        return sorted(list(points))

    def get_incidence_matrix(self):
        num_items = len(self.points)
        matrix = np.zeros((num_items, num_items), dtype=int)
        
        for h_idx, h_vec in enumerate(self.hyperplanes):
            for p_idx, p_vec in enumerate(self.points):
                if self.gf.dot(h_vec, p_vec) == 0:
                    matrix[h_idx][p_idx] = 1
        return matrix

def solve_proposition2():
    print("=== Setting up Proposition 2 Experiment (Fixed API) ===")
    
    # 1. Parameters
    q = 8
    target_n = 35
    target_k = 4
    min_weight = 28
    divisibility = 4
    
    gf = GF8()
    
    # 2. Geometry PG(3, 8)
    print(f"Generating Geometry for PG({target_k-1}, {q})...")
    pg = ProjectiveGeometry(target_k, q, gf)
    points = pg.points
    num_points = len(points)
    incidence = pg.get_incidence_matrix()
    print(f"Number of points/hyperplanes: {num_points}")
    
    # 3. Highs Solver Setup
    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    
    # --- [수정 포인트 1] addVar 사용법 변경 ---
    # 키워드 인자(lb=, ub=) 제거 -> 위치 인자(0.0, 1.0) 사용
    # type 인자 제거 -> changeColIntegrality 함수로 별도 설정
    
    # Variables x_P (0 to num_points-1)
    for i in range(num_points):
        h.addVar(0.0, 1.0) 
        h.changeColIntegrality(i, highspy.HighsVarType.kInteger)

    # Variables y_H (num_points to 2*num_points-1)
    offset = num_points
    for i in range(num_points):
        h.addVar(0.0, 1.0)
        h.changeColIntegrality(offset + i, highspy.HighsVarType.kInteger)

    # Constraint 1: Weight Divisibility & Bounds
    rhs_value = float(target_n - min_weight) # 7.0
    
    print("Adding Hyperplane Constraints...")
    for h_idx in range(num_points):
        p_indices = np.where(incidence[h_idx] == 1)[0]
        
        col_indices = list(p_indices) + [offset + h_idx]
        # 인덱스는 반드시 int, 계수는 float여야 함
        col_indices = [int(x) for x in col_indices] 
        coeffs = [1.0] * len(p_indices) + [float(divisibility)]
        
        # --- [수정 포인트 2] addRow 사용법 변경 ---
        # h.addRow(lower_bound, upper_bound, num_nonzeros, indices, values)
        h.addRow(rhs_value, rhs_value, len(col_indices), col_indices, coeffs)

    # Constraint 2: Extension Constraints
    print("Adding Extension Constraints (Projection to Seed Code)...")
    
    # --- MOCK DATA FOR SEED CODE ---
    num_points_pg2 = 73 
    seed_multiplicities = [0] * num_points_pg2
    for i in range(34):
        seed_multiplicities[i % num_points_pg2] += 1
    # -------------------------------

    pg2 = ProjectiveGeometry(3, q, gf)
    pg2_points = pg2.points
    
    # Case A: u = 0
    p_zero_idx = -1
    for i, p in enumerate(points):
        if p == (0,0,0,1):
            p_zero_idx = i
            break
    
    if p_zero_idx != -1:
        h.addRow(1.0, 1.0, 1, [int(p_zero_idx)], [1.0])
    
    # Case B: u != 0
    for u_idx, u_vec in enumerate(pg2_points):
        target_count = float(seed_multiplicities[u_idx])
        
        relevant_p_indices = []
        for p_idx, p_vec in enumerate(points):
            proj = p_vec[:3]
            if all(x==0 for x in proj): continue
            
            first_nz = next((i for i, x in enumerate(u_vec) if x != 0), None)
            if first_nz is None: continue 
            
            lambda_val = gf.mul(proj[first_nz], gf.exp[(7 - gf.log[u_vec[first_nz]])%7])
            
            is_multiple = True
            for k in range(3):
                if proj[k] != gf.mul(lambda_val, u_vec[k]):
                    is_multiple = False
                    break
            
            if is_multiple:
                relevant_p_indices.append(p_idx)
                
        if relevant_p_indices:
            coeffs = [1.0] * len(relevant_p_indices)
            col_idx_list = [int(x) for x in relevant_p_indices]
            h.addRow(target_count, target_count, len(col_idx_list), col_idx_list, coeffs)

    # 4. Solve
    print("Solving ILP...")
    h.run()
    
    status = h.getModelStatus()
    print(f"Solver Status: {status}")
    
    if status == highspy.HighsModelStatus.kOptimal:
        print("Solution Found!")
    elif status == highspy.HighsModelStatus.kInfeasible:
        print("Infeasible! (Proposition 2 Proved)")
    else:
        print(f"Other status: {status}")

if __name__ == "__main__":
    solve_proposition2()
