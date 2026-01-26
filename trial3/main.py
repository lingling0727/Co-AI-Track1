import time
import sys
import os
import csv
import datetime

try:
    from geometry import generate_projective_points, generate_hyperplanes
    from ilp_model import CodeExtender
    from checker import verify_solution, filter_isomorphic_solutions
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    sys.exit(1)

def save_geometry_data(k, q, points):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(directory): os.makedirs(directory)
    filename = os.path.join(directory, f"geometry_k{k}_q{q}.txt")
    with open(filename, "w") as f:
        f.write(f"# PG({k-1}, {q}) - {len(points)} points\n")
        for idx, p in enumerate(points): f.write(f"{idx}: {p}\n")

def save_experiment_results(n, k, q, weights, num_points, phase0_time, phase0_5_time, phase1_5_prep_time, search_time, phase2_time, total_sols, unique_sols, nodes_visited, pruned_nodes):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv")
    headers = [
        "Timestamp", "n", "k", "q", "Weights", "Points",
        "Phase0_Time", "Phase0.5_Time", "Phase1.5_Prep_Time", "Phase1_Search_Time", "Phase2_Time", "Status",
        "Total_Sols", "Unique_Sols", "Nodes", "Pruned"
    ]
    
    file_exists = os.path.isfile(filename)
    should_write_header = False

    if file_exists:
        try:
            with open(filename, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    existing_headers = next(reader)
                    if existing_headers != headers:
                        should_write_header = True
                        file_exists = False
                except StopIteration:
                    should_write_header = True
        except Exception:
            should_write_header = True
            file_exists = False
    else:
        should_write_header = True

    mode = 'a' if file_exists else 'w'
    with open(filename, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if should_write_header: writer.writerow(headers)
        status = "Feasible" if total_sols > 0 else "Infeasible"
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n, k, q, str(sorted(list(weights))), num_points,
            f"{phase0_time:.4f}", f"{phase0_5_time:.4f}", f"{phase1_5_prep_time:.4f}", f"{search_time:.4f}", f"{phase2_time:.4f}", status,
            total_sols, unique_sols, nodes_visited, pruned_nodes
        ])
    print(f"  > [Logged] Results saved to '{filename}'")

def run_classification(n, k, q, weights_str):
    try:
        target_weights = set(map(int, weights_str.split(',')))
    except ValueError:
        print("Error: Invalid weights format.")
        return

    print("="*60)
    print(f"[*] Trial 3: Enhanced Classification (Phase 0.5 & 1.5)")
    print(f"[*] Parameters: [n={n}, k={k}, q={q}], Weights={target_weights}")
    print("="*60)

    # [1] Geometry
    print("\n[1] Generating Geometry...")
    start_geom = time.time()
    points = generate_projective_points(k, q)
    hyperplanes = points
    print(f"  > Generated {len(points)} points in {time.time() - start_geom:.4f}s.")
    save_geometry_data(k, q, points)

    # [2] Phase 0, 0.5, 1, 1.5
    print("\n[2] Phase 0~1.5: Enumeration with Enhanced Pruning...")
    extender = CodeExtender(n, k, q, target_weights)
    solutions, nodes, pruned, p0_time, p0_5_time, p1_5_prep_time, p1_search_time = extender.build_and_solve(points, hyperplanes)
    
    print(f"  > Enumeration finished.")
    print(f"  > Phase 0 (ILP Feasibility): {p0_time:.4f}s")
    print(f"  > Phase 0.5 (Theoretical Bounds): {p0_5_time:.4f}s")
    print(f"  > Phase 1.5 (Symmetry Prep): {p1_5_prep_time:.4f}s")
    print(f"  > Phase 1 (Backtrack Search): {p1_search_time:.4f}s")
    print(f"  > Candidates found: {len(solutions)}")

    if not solutions:
        save_experiment_results(n, k, q, target_weights, len(points), p0_time, p0_5_time, p1_5_prep_time, p1_search_time, 0.0, 0, 0, nodes, pruned)
        return

    # [3] Phase 2: Verification
    print("\n[3] Phase 2: Verification & Filtering...")
    start_check = time.time()
    verified = [sol for sol in solutions if verify_solution(sol, n, target_weights, points, hyperplanes, q)]
    unique = filter_isomorphic_solutions(verified, n, points, hyperplanes, q)
    phase2_time = time.time() - start_check
    
    print(f"  > Phase 2 finished in {phase2_time:.4f}s.")
    print(f"  > Unique Solutions: {len(unique)}")

    print("\n[*] Final Solutions:")
    for i, sol in enumerate(unique[:5]):
        print(f"  #{i+1}: {sol}")
    if len(unique) > 5: print(f"  ... {len(unique)-5} more.")

    save_experiment_results(n, k, q, target_weights, len(points), p0_time, p0_5_time, p1_5_prep_time, p1_search_time, phase2_time, len(solutions), len(unique), nodes, pruned)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        run_classification(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        print("Usage: python main.py <n> <k> <q> <weights>")
        print("Running default test: n=7, k=3, q=2, weights=3,4")
        run_classification(7, 3, 2, "3,4")
