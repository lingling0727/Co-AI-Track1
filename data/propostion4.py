import numpy as np
import itertools

def generate_projective_geometry(k, q):
    """
    Generate points and hyperplanes for PG(k-1, q).
    Returns:
        points: List of column vectors representing points in PG(k-1, q).
        hyperplanes: List of dual vectors representing hyperplanes.
        incidence_matrix: Matrix A where A[i][j] = 1 if point j is in hyperplane i.
    """
    # Generate all non-zero vectors of length k
    raw_vectors = list(itertools.product(range(q), repeat=k))
    raw_vectors.remove((0,)*k) # Remove zero vector
    
    # Normalize vectors to get projective points (first non-zero element is 1)
    points = []
    seen = set()
    for v in raw_vectors:
        # Find first non-zero to normalize
        first_nz = next((x for x in v if x != 0), None)
        factor = pow(first_nz, -1, q) # Inverse in GF(q)
        normalized = tuple((x * factor) % q for x in v)
        if normalized not in seen:
            seen.add(normalized)
            points.append(normalized)
            
    # In PG(k-1, q), hyperplanes are isomorphic to points (dual space)
    hyperplanes = points 
    num_elements = len(points)
    
    # Create Incidence Matrix A
    # A[i, j] = 1 if dot(H_i, P_j) == 0 (mod q)
    incidence_matrix = np.zeros((num_elements, num_elements), dtype=int)
    
    for i, h in enumerate(hyperplanes):
        for j, p in enumerate(points):
            dot_prod = sum(h_val * p_val for h_val, p_val in zip(h, p)) % q
            if dot_prod == 0:
                incidence_matrix[i, j] = 1
                
    return points, incidence_matrix

class Proposition4_Parameters:
    def __init__(self):
        # 1. Basic Code Parameters from Proposition 4
        self.n = 153      # Length
        self.k = 7        # Dimension
        self.q = 2        # Field size (Binary)
        self.d_min = 76   # Minimum Distance
        
        # 2. Weight Constraints (from Lemma 4 & Prop 4)
        # Non-zero weights must be in this set
        self.allowed_weights = {76, 80, 92, 96, 100}
        
        # 3. Point Multiplicity Constraints (from Proof of Prop 4)
        # "Noting that C has maximum point multiplicity 2"
        self.max_point_multiplicity = 2 
        self.min_point_multiplicity = 0
        
        # 4. ILP Formulation Data
        # Generate the geometry for PG(6, 2)
        print(f"Generating geometry for PG({self.k-1}, {self.q})...")
        self.points, self.A = generate_projective_geometry(self.k, self.q)
        self.num_vars = len(self.points) # 127 variables for PG(6,2)
        
        # Calculate allowed intersection sizes (Hyperplane capacities)
        # Relation: Weight(c) = n - |Intersection with H|
        # Therefore: |Intersection| = n - Weight
        self.allowed_intersections = {self.n - w for w in self.allowed_weights}
        # Include n (weight 0) effectively, though usually handled separately
        
        print(f"Setup Complete.")
        print(f" - Variables (Points): {self.num_vars}")
        print(f" - Constraints (Hyperplanes): {self.A.shape[0]}")
        print(f" - Allowed Hyperplane Point Counts (RHS): {sorted(list(self.allowed_intersections))}")

    def get_ilp_model_description(self):
        """Returns a text description of the ILP formulation."""
        desc = []
        desc.append("Minimize/Find: Vector x (length 127)")
        desc.append(f"Subject to:")
        desc.append(f"1. A * x = k_H (where k_H is a vector of intersection sizes)")
        desc.append(f"2. For each element k_h in k_H: k_h in {self.allowed_intersections}")
        desc.append(f"3. sum(x) = {self.n} (Total length)")
        desc.append(f"4. {self.min_point_multiplicity} <= x_i <= {self.max_point_multiplicity} (Integer constraints)")
        return "\n".join(desc)

    def verify_known_solutions(self):
        """
        Data for the two solutions found in the paper for verification.
        """
        sol1 = {
            'weights': {76: 107, 80: 15, 92: 5},
            'aut_group_order': 16128
        }
        sol2 = {
            'weights': {76: 108, 80: 14, 92: 4, 96: 1},
            'aut_group_order': 32256
        }
        return [sol1, sol2]

# --- Execution ---
prop4 = Proposition4_Parameters()

print("\n--- ILP Formulation for Proposition 4 ---")
print(prop4.get_ilp_model_description())

print("\n--- Target Solutions (Ground Truth) ---")
solutions = prop4.verify_known_solutions()
for i, sol in enumerate(solutions, 1):
    print(f"Solution {i}: Distribution {sol['weights']}, |Aut|={sol['aut_group_order']}")
