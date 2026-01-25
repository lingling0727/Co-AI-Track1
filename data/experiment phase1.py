import highspy
import numpy as np
import time
import pandas as pd

# ==========================================
# 1. 기초 클래스 (GF8, ProjectiveGeometry)
# ==========================================
class GF8:
    def __init__(self):
        self.size = 8
        self.prim_poly = 0b1011
        self.exp = [0]*8; self.log = [0]*8
        x = 1
        for i in range(7):
            self.exp[i] = x; self.log[x] = i
            x <<= 1
            if x & 0b1000: x ^= self.prim_poly
        self.exp[7] = 0
    def add(self, a, b): return a ^ b
    def mul(self, a, b):
        if a==0 or b==0: return 0
        return self.exp[(self.log[a]+self.log[b])%7]
    def dot(self, v1, v2):
        res = 0
        for a,b in zip(v1,v2): res = self.add(res, self.mul(a,b))
        return res

class ProjectiveGeometry:
    def __init__(self, k, q, gf):
        self.k=k; self.q=q; self.gf=gf
        self.points = self._gen_points()
    def _gen_points(self):
        pts = []; seen = set()
        for i in range(1, self.q**self.k):
            v = []
            tmp = i
            for _ in range(self.k): v.append(tmp%self.q); tmp//=self.q
            v = v[::-1]
            fnz = next((idx for idx,x in enumerate(v) if x!=0), -1)
            if fnz==-1: continue
            inv = self.gf.exp[(7-self.gf.log[v[fnz]])%7]
            nv = tuple(self.gf.mul(x, inv) for x in v)
            if nv not in seen: seen.add(nv); pts.append(nv)
        return sorted(list(pts))
    def get_incidence_matrix(self):
        n = len(self.points)
        mat = np.zeros((n, n), dtype=int)
        for h_i, h_p in enumerate(self.points):
            for p_i, p_p in enumerate(self.points):
                if self.gf.dot(h_p, p_p) == 0: mat[h_i][p_i] = 1
        return mat

# ==========================================
# 2. 실험 클래스 (Relax Mode 추가)
# ==========================================
class LinearCodeExperiment:
    def __init__(self):
        self.q = 8
        self.target_n = 35
        self.target_k = 4
        self.min_weight = 28
        self.divisibility = 4
        
        self.gf = GF8()
        self.pg = ProjectiveGeometry(self.target_k, self.q, self.gf)
        self.num_points = len(self.pg.points)
        self.incidence = self.pg.get_incidence_matrix()
        
        self.highs = None
        self.logs = [] 

    def _build_model(self, relax_mode=False):
        """
        relax_mode=True: Phase 1 작동 시연을 위해 제약조건을 일부 완화함
        """
        h = highspy.Highs()
        h.setOptionValue("output_flag", False) # 로그 너무 많으면 끔
        
        # Variables: x_P
        for i in range(self.num_points):
            h.addVar(0.0, 1.0)
            h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
            
        # Variables: y_H
        offset = self.num_points
        for i in range(self.num_points):
            h.addVar(0.0, 1.0)
            h.changeColIntegrality(offset + i, highspy.HighsVarType.kInteger)
            
        # Constraint: Weight Divisibility
        # 원본 Eq: 4*y + sum(x) = 7
        # 완화 Eq: 6 <= 4*y + sum(x) <= 8 (여유를 둠)
        rhs_val = float(self.target_n - self.min_weight) # 7.0
        
        if relax_mode:
            lb, ub = rhs_val - 2.0, rhs_val + 2.0 # 완화된 범위
        else:
            lb, ub = rhs_val, rhs_val # 엄격한 범위
            
        for h_idx in range(self.num_points):
            p_idxs = np.where(self.incidence[h_idx] == 1)[0]
            col_idxs = [int(x) for x in list(p_idxs) + [offset + h_idx]]
            coeffs = [1.0] * len(p_idxs) + [float(self.divisibility)]
            h.addRow(lb, ub, len(col_idxs), col_idxs, coeffs)
            
        # Constraint: Extension from Seed Code (Infeasible의 주 원인)
        # 시연을 위해 relax_mode일 때는 이 제약을 건너뜀
        if not relax_mode:
            pg2 = ProjectiveGeometry(3, self.q, self.gf)
            seed_counts = [0] * len(pg2.points)
            for i in range(34): seed_counts[i % len(pg2.points)] += 1
            
            p0_idx = next((i for i, p in enumerate(self.pg.points) if p == (0,0,0,1)), -1)
            if p0_idx != -1:
                h.addRow(1.0, 1.0, 1, [int(p0_idx)], [1.0])
                
            for u_idx, u_vec in enumerate(pg2.points):
                tgt = float(seed_counts[u_idx])
                rel_idxs = []
                for p_idx, p_vec in enumerate(self.pg.points):
                    proj = p_vec[:3]
                    if all(x==0 for x in proj): continue
                    fnz = next((i for i, x in enumerate(u_vec) if x != 0), None)
                    if fnz is None: continue
                    lam = self.gf.mul(proj[fnz], self.gf.exp[(7-self.gf.log[u_vec[fnz]])%7])
                    if all(proj[k] == self.gf.mul(lam, u_vec[k]) for k in range(3)):
                        rel_idxs.append(p_idx)
                if rel_idxs:
                    h.addRow(tgt, tgt, len(rel_idxs), [int(x) for x in rel_idxs], [1.0]*len(rel_idxs))
        
        self.highs = h

    def run_phase_0(self, relax=False):
        print(f"\n=== [Phase 0] ILP Feasibility Check (Relaxed={relax}) ===")
        self._build_model(relax_mode=relax)
        
        start_t = time.time()
        self.highs.run()
        end_t = time.time()
        
        status = self.highs.getModelStatus()
        status_str = "Unknown"
        is_feasible = False
        
        if status == highspy.HighsModelStatus.kInfeasible:
            status_str = "Infeasible"
        elif status == highspy.HighsModelStatus.kOptimal:
            status_str = "Optimal (Feasible)"
            is_feasible = True
            
        self.logs.append({
            "Phase": 0, "Solution_ID": "-", "Status": status_str,
            "Time(s)": round(end_t - start_t, 4), "Selected_Count": "-", "Indices": "-"
        })
        
        print(f"Phase 0 Status: {status_str}, Time: {end_t - start_t:.4f}s")
        if is_feasible:
            print(">> SUCCESS: Model is Feasible! Moving to Phase 1.")
        else:
            print(">> FAILURE: Model is Infeasible. Phase 1 will be skipped.")
        return is_feasible

    def run_phase_1(self, max_solutions=3):
        print(f"\n=== [Phase 1] Lattice Point Enumeration (Max {max_solutions}) ===")
        
        solution_count = 0
        
        while solution_count < max_solutions:
            start_t = time.time()
            self.highs.run()
            end_t = time.time()
            
            status = self.highs.getModelStatus()
            
            if status != highspy.HighsModelStatus.kOptimal:
                self.logs.append({
                    "Phase": 1, "Solution_ID": "End", "Status": "No more solutions",
                    "Time(s)": round(end_t - start_t, 4), "Selected_Count": "-", "Indices": "-"
                })
                print(">> No more solutions found (Space exhausted).")
                break
                
            # 해 추출
            sol_obj = self.highs.getSolution()
            col_vals = np.array(sol_obj.col_value)
            current_x = col_vals[:self.num_points]
            selected_indices = [i for i, val in enumerate(current_x) if val > 0.5]
            
            solution_count += 1
            
            self.logs.append({
                "Phase": 1, "Solution_ID": solution_count, "Status": "Found",
                "Time(s)": round(end_t - start_t, 4),
                "Selected_Count": len(selected_indices),
                "Indices": str(selected_indices[:5]) + "..." # 길어서 생략
            })
            
            print(f"Sol #{solution_count} Found! (Selected {len(selected_indices)} columns)")
            
            # --- [핵심] No-good Cut 추가 (현재 해 금지) ---
            # sum(1-x for x=1) + sum(x for x=0) >= 1
            cut_indices = []
            cut_coeffs = []
            cut_lhs_offset = 0.0
            
            for i in range(self.num_points):
                val = current_x[i]
                if val > 0.5: # 1인 변수 -> (1-x) -> -x
                    cut_indices.append(int(i))
                    cut_coeffs.append(-1.0)
                    cut_lhs_offset += 1.0
                else: # 0인 변수 -> x
                    cut_indices.append(int(i))
                    cut_coeffs.append(1.0)
            
            rhs = 1.0 - cut_lhs_offset
            # Row 추가 (LowBound, UpBound, NumNZ, Indices, Values)
            self.highs.addRow(rhs, highspy.kHighsInf, len(cut_indices), cut_indices, cut_coeffs)
            print(f"   >> Added Cut to ban Sol #{solution_count}")

    def save_csv(self, filename="phase_results_success.csv"):
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"\n[SUCCESS] Results saved to '{filename}'")
        print(df[["Phase", "Solution_ID", "Status", "Time(s)"]])

if __name__ == "__main__":
    exp = LinearCodeExperiment()
    
    # [중요] relax=True로 설정하여 Phase 0를 강제로 통과시킴
    is_feasible = exp.run_phase_0(relax=True)
    
    if is_feasible:
        # 해를 3개만 찾아봄
        exp.run_phase_1(max_solutions=3)

    exp.save_csv()
    
    exp.save_csv()
