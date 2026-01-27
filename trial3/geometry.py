import itertools
import numpy as np
import os

class GaloisField:
    def __init__(self, q):
        self.q = q
        self.is_prime = self._is_prime(q)
        self.mul_table = {}
        self.add_table = {}
        self.inv_table = {}
        if not self.is_prime:
            self._init_tables()
        self.primitive = self._find_primitive_element()

    def _is_prime(self, n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def _init_tables(self):
        if self.q == 4:
            self.mul_table = {
                (0,0):0, (0,1):0, (0,2):0, (0,3):0,
                (1,0):0, (1,1):1, (1,2):2, (1,3):3,
                (2,0):0, (2,1):2, (2,2):3, (2,3):1,
                (3,0):0, (3,1):3, (3,2):1, (3,3):2
            }
            self.inv_table = {1:1, 2:3, 3:2}
        elif self.q == 8:
            exp = [1, 2, 4, 3, 6, 7, 5, 1, 2, 4, 3, 6, 7, 5]
            log = {1:0, 2:1, 4:2, 3:3, 6:4, 7:5, 5:6}
            for a in range(8):
                for b in range(8):
                    if a == 0 or b == 0: self.mul_table[(a,b)] = 0
                    else: self.mul_table[(a,b)] = exp[log[a] + log[b]]
            self.inv_table = {a: exp[(7 - log[a]) % 7] for a in range(1, 8)}
        else:
            if self.q == 9:
                # GF(9) ~= F3[x] / (x^2 + 1), elements are 3*h + l
                for a in range(9):
                    for b in range(9):
                        h1, l1 = divmod(a, 3)
                        h2, l2 = divmod(b, 3)
                        
                        # Addition
                        self.add_table[(a,b)] = 3*((h1 + h2) % 3) + ((l1 + l2) % 3)
                        
                        # Multiplication: (h1*x+l1)*(h2*x+l2) with x^2 = -1 = 2
                        h_prod = (h1*l2 + l1*h2) % 3
                        l_prod = (l1*l2 + 2*h1*h2) % 3
                        self.mul_table[(a,b)] = 3*h_prod + l_prod
                
                for a in range(1, 9):
                    for b in range(1, 9):
                        if self.mul_table.get((a,b)) == 1:
                            self.inv_table[a] = b
                            break
            else:
                raise NotImplementedError(f"GF({self.q}) is not supported yet.")

    def _find_primitive_element(self):
        if self.q == 2: return 1
        for a in range(2, self.q):
            curr = a
            order = 1
            while order < self.q - 1:
                curr = self.mul(curr, a)
                if curr == 1: break
                order += 1
            if order == self.q - 1:
                return a
        return 1

    def add(self, a, b):
        if self.is_prime: return (a + b) % self.q
        if self.q == 9: return self.add_table.get((a,b))
        return a ^ b

    def sub(self, a, b):
        if self.is_prime: return (a - b) % self.q
        if self.q == 9: return self.add(a, self.mul(b, 2)) # a - b = a + 2b in char 3
        return a ^ b

    def mul(self, a, b):
        if self.is_prime: return (a * b) % self.q
        return self.mul_table.get((a,b), self.mul_table.get((b,a), 0))

    def inv(self, a):
        if a == 0: raise ValueError("Division by zero")
        if self.is_prime: return pow(a, self.q - 2, self.q)
        return self.inv_table[a]

    def dot(self, v1, v2):
        res = 0
        for a, b in zip(v1, v2):
            res = self.add(res, self.mul(a, b))
        return res

_gf_cache = {}
def get_gf(q):
    if q not in _gf_cache: _gf_cache[q] = GaloisField(q)
    return _gf_cache[q]

def generate_projective_points(k, q):
    gf = get_gf(q)
    points = set()
    for v in itertools.product(range(q), repeat=k):
        if all(x == 0 for x in v): continue
        first_nonzero = next((x for x in v if x != 0), None)
        inv = gf.inv(first_nonzero)
        normalized = tuple(gf.mul(x, inv) for x in v)
        points.add(normalized)
    return sorted(list(points))

def generate_hyperplanes(k, q):
    return generate_projective_points(k, q)

def is_point_in_hyperplane(point, hyperplane, q):
    gf = get_gf(q)
    return gf.dot(point, hyperplane) == 0

def get_incidence_matrix(points, hyperplanes, q):
    matrix = []
    for h in hyperplanes:
        row = [1 if is_point_in_hyperplane(p, h, q) else 0 for p in points]
        matrix.append(row)
    return np.array(matrix)

def mat_vec_mul(matrix, vec, gf):
    k = len(vec)
    result = []
    for i in range(k):
        val = 0
        for j in range(k):
            term = gf.mul(matrix[i][j], vec[j])
            val = gf.add(val, term)
        result.append(val)
    return tuple(result)

def is_independent(vectors, q):
    if not vectors: return True
    gf = get_gf(q)
    k = len(vectors[0])
    temp_rows = [list(v) for v in vectors]
    pivot_row = 0
    for col in range(k):
        if pivot_row >= len(temp_rows): break
        if temp_rows[pivot_row][col] == 0:
            for r in range(pivot_row + 1, len(temp_rows)):
                if temp_rows[r][col] != 0:
                    temp_rows[pivot_row], temp_rows[r] = temp_rows[r], temp_rows[pivot_row]
                    break
            else: continue
        inv = gf.inv(temp_rows[pivot_row][col])
        for r in range(pivot_row + 1, len(temp_rows)):
            if temp_rows[r][col] != 0:
                factor = gf.mul(temp_rows[r][col], inv)
                for c in range(col, k):
                    term = gf.mul(factor, temp_rows[pivot_row][c])
                    temp_rows[r][c] = gf.sub(temp_rows[r][c], term)
        pivot_row += 1
    return pivot_row == len(vectors)

def generate_gl_generators(k, q):
    gf = get_gf(q)
    generators = []
    
    # 1. Scalar multiplication of the first row by a primitive element
    if q > 2:
        prim = gf.primitive
        mat = [[0]*k for _ in range(k)]
        for i in range(k): mat[i][i] = 1
        mat[0][0] = prim
        generators.append(mat)
        
    if k >= 2:
        # 2. Elementary matrix E_{1,2}(1)
        mat = [[0]*k for _ in range(k)]
        for i in range(k): mat[i][i] = 1
        mat[0][1] = 1
        generators.append(mat)
        
        # 3. Cyclic permutation
        mat = [[0]*k for _ in range(k)]
        for i in range(k-1):
            mat[i][i+1] = 1
        mat[k-1][0] = 1
        generators.append(mat)
        
    return generators

def _generate_full_gl(k, q):
    matrices = []
    all_vectors = list(itertools.product(range(q), repeat=k))
    nonzero_vectors = [v for v in all_vectors if any(x != 0 for x in v)]
    gf = get_gf(q)
    def backtrack(current_rows):
        if len(current_rows) == k:
            matrices.append([list(r) for r in current_rows])
            return
        for v in nonzero_vectors:
            if is_independent(current_rows + [v], q):
                backtrack(current_rows + [v])
    backtrack([])
    return matrices

def generate_linear_group(k, q, limit=5000):
    gl_order = 1
    for i in range(k): gl_order *= (q**k - q**i)
    
    if gl_order <= limit:
        return _generate_full_gl(k, q)
    
    # Fallback or other groups logic can remain here if needed
    return _generate_diagonal_group(k, q)

def _generate_diagonal_group(k, q):
    matrices = []
    nonzero_elements = list(range(1, q))
    for diag in itertools.product(nonzero_elements, repeat=k):
        matrix = [[0]*k for _ in range(k)]
        for i in range(k): matrix[i][i] = diag[i]
        matrices.append(matrix)
    return matrices

def _generate_scalar_group(k, q):
    matrices = []
    for s in range(1, q):
        matrix = [[s if i == j else 0 for j in range(k)] for i in range(k)]
        matrices.append(matrix)
    return matrices

def get_orbits(points, generators, q):
    gf = get_gf(q)
    visited = set()
    representatives = []
    for p in points:
        if p in visited: continue
        representatives.append(p)
        
        stack = [p]
        visited.add(p)
        
        while stack:
            curr = stack.pop()
            for gen in generators:
                next_vec = mat_vec_mul(gen, curr, gf)
                if all(x == 0 for x in next_vec): continue
                
                first_nonzero = next((x for x in next_vec if x != 0), None)
                inv = gf.inv(first_nonzero)
                next_p = tuple(gf.mul(x, inv) for x in next_vec)
                
                if next_p not in visited: 
                    visited.add(next_p)
                    stack.append(next_p)
    return representatives