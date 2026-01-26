import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry import get_incidence_matrix, get_projection_map

try:
    import highspy
except ImportError as e:
    # Defer the import error until build_and_solve is called, so that importing
    # this module doesn't crash tooling/tests that don't solve.
    highspy = None


class CodeExtender:
    def __init__(self, n: int, k: int, q: int, target_weights):
        self.n = int(n)
        self.k = int(k)
        self.q = int(q)
        self.target_weights = sorted(list(target_weights))

    def build_and_solve(self, points, hyperplanes, base_code_counts=None, points_km1=None):
        """
        Builds and solves the ILP model using HiGHS (highspy).
        Returns: (solutions, nodes_visited, pruned_nodes)
          - solutions: list[dict[int,int]] mapping point index -> multiplicity
          - nodes_visited: HiGHS MIP node count if available, else 0
          - pruned_nodes: not available here; return 0
        """
        if highspy is None:
            raise ImportError(
                "'highspy' (HiGHS Python bindings) is not installed. Please run 'pip install highspy'."
            )

        num_points = len(points)
        allowed_k = [self.n - w for w in self.target_weights]
        if not allowed_k:
            return [], 0, 0

        # Incidence: rows=hyperplanes, cols=points, entries in {0,1}
        incidence = get_incidence_matrix(points, hyperplanes, self.q)
        # Ensure plain Python ints for indexing
        incidence = np.asarray(incidence, dtype=int)

        inf = highspy.kHighsInf

        h = highspy.Highs()
        # Determinism / clean output
        try:
            h.setOptionValue("output_flag", False)
        except Exception:
            pass
        try:
            h.setOptionValue("threads", 1)
        except Exception:
            pass
        try:
            # Force exact MIP (no relative gap)
            h.setOptionValue("mip_rel_gap", 0.0)
        except Exception:
            pass
        try:
            # Optional seed, if supported by the installed HiGHS build
            h.setOptionValue("random_seed", 0)
        except Exception:
            pass

        # ----- Variables -----
        # x_i: integer multiplicities
        x_idx = []
        for _ in range(num_points):
            h.addVar(0.0, float(self.n))
            x_idx.append(h.getNumCol() - 1)

        # Set x integrality
        for col in x_idx:
            h.changeColIntegrality(col, highspy.HighsVarType.kInteger)

        # Systematic assumption: unit vectors must appear (x_i >= 1)
        if base_code_counts is None:
            for i, p in enumerate(points):
                if p.count(1) == 1 and p.count(0) == self.k - 1:
                    # set lower bound to 1
                    h.changeColBounds(x_idx[i], 1.0, float(self.n))

        # z_{h,k}: binary selector for each hyperplane and allowed_k
        # We'll store z indices in a dict keyed by (h_idx, k_val)
        z_idx: Dict[Tuple[int, int], int] = {}
        for h_i in range(len(hyperplanes)):
            for k_val in allowed_k:
                h.addVar(0.0, 1.0)
                col = h.getNumCol() - 1
                h.changeColIntegrality(col, highspy.HighsVarType.kInteger)  # binary via bounds + integer
                z_idx[(h_i, k_val)] = col

        # ----- Constraints -----
        # (1) Length: sum x_i = n
        h.addRow(float(self.n), float(self.n), num_points, x_idx, [1.0] * num_points)

        # (2) Hyperplane disjunction:
        #   expr = sum_{p in H} x_p
        #   expr == sum_k k * z_{h,k}
        #   sum_k z_{h,k} == 1
        for h_i in range(len(hyperplanes)):
            # sum_k z == 1
            z_cols = [z_idx[(h_i, k_val)] for k_val in allowed_k]
            h.addRow(1.0, 1.0, len(z_cols), z_cols, [1.0] * len(z_cols))

            # expr - sum_k k*z == 0
            cols = []
            vals = []

            # x part
            row = incidence[h_i]
            # Add only nonzeros for sparsity
            for p_i in np.nonzero(row)[0]:
                cols.append(x_idx[int(p_i)])
                vals.append(1.0)

            # z part
            for k_val in allowed_k:
                cols.append(z_idx[(h_i, k_val)])
                vals.append(-float(k_val))

            h.addRow(0.0, 0.0, len(cols), cols, vals)

        # (3) Extension constraints (if applicable)
        if base_code_counts is not None and points_km1 is not None:
            mapping, _ext_idx = get_projection_map(self.k, self.q, points, points_km1)
            for p_km1_idx, p_k_indices in mapping.items():
                target_count = int(base_code_counts.get(p_km1_idx, 0))
                cols = [x_idx[i] for i in p_k_indices]
                vals = [1.0] * len(cols)
                h.addRow(float(target_count), float(target_count), len(cols), cols, vals)

        # Objective: 0 (feasibility). HiGHS minimizes by default; costs are 0 already.

        # ----- Enumerate all feasible solutions via integer no-good cuts -----
        solutions: List[Dict[int, int]] = []
        seen = set()

        M = float(self.n)  # big-M for no-good constraints

        def sol_key(sol: Dict[int, int]):
            return tuple(sorted(sol.items()))

        while True:
            h.run()
            status = h.getModelStatus()

            if status != highspy.HighsModelStatus.kOptimal:
                break

            sol = h.getSolution()
            col_vals = sol.col_value  # array-like

            # Extract x solution
            x_vals = [int(round(col_vals[col])) for col in x_idx]
            sol_dict = {i: v for i, v in enumerate(x_vals) if v > 0}
            key = sol_key(sol_dict)
            if key in seen:
                # Defensive: if HiGHS returns same solution again, stop
                break
            seen.add(key)
            solutions.append(sol_dict)

            # Add integer no-good cut to exclude EXACT same integer vector x
            # Introduce binary u_i for each x_i:
            #   x_i - x_i* <= M u_i
            #   x_i* - x_i <= M u_i
            # And sum u_i >= 1
            u_cols = []
            for i, x_star in enumerate(x_vals):
                h.addVar(0.0, 1.0)
                u = h.getNumCol() - 1
                h.changeColIntegrality(u, highspy.HighsVarType.kInteger)  # binary via bounds + integer
                u_cols.append(u)

                # x_i - M u_i <= x_star
                h.addRow(-inf, float(x_star), 2, [x_idx[i], u], [1.0, -M])

                # -x_i - M u_i <= -x_star   (equiv: x_star - x_i <= M u_i)
                h.addRow(-inf, float(-x_star), 2, [x_idx[i], u], [-1.0, -M])

            # sum u_i >= 1
            h.addRow(1.0, inf, len(u_cols), u_cols, [1.0] * len(u_cols))

        # Sort solutions deterministically to reduce downstream tie effects
        solutions = sorted(solutions, key=sol_key)

        # Nodes visited (best-effort)
        nodes = 0
        try:
            info = h.getInfo()
            # Different builds expose different field names; try common ones
            nodes = int(getattr(info, "mip_node_count", 0) or getattr(info, "mip_nodes", 0) or 0)
        except Exception:
            nodes = 0

        return solutions, nodes, 0
