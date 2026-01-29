import time
import sys
import os
import csv
import datetime

import threading
import psutil

try:
    from geometry import generate_projective_points, generate_hyperplanes
    from ilp_model import CodeExtender
    from checker import verify_solution, filter_isomorphic_solutions
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    sys.exit(1)

def get_memory_usage_mb():
    """현재 프로세스의 메모리 사용량을 MB 단위로 반환합니다."""
    process = psutil.Process(os.getpid())
    # rss (Resident Set Size)는 실제 사용 중인 물리 메모리 양입니다.
    return process.memory_info().rss / (1024 * 1024)

def monitor_and_log_memory(log_filename, time_limit_sec, interval_sec):
    """
    별도 스레드에서 실행될 함수입니다.
    주기적으로 메모리 사용량을 모니터링하고 기록합니다.
    제한 시간이 지나면 프로그램을 강제 종료합니다.
    """
    start_time = time.time()
    
    headers = ["Timestamp", "Elapsed_Time_sec", "Memory_Usage_MB"]
    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    except IOError as e:
        print(f"  > [Monitor Error] Log file could not be written: {e}")
        return

    print(f"  > [Monitor] Started. Logging every {interval_sec}s for {time_limit_sec}s.")

    # 설정된 간격으로 제한 시간 동안 기록
    num_intervals = time_limit_sec // interval_sec # 예: 3600s / 60s = 60번 기록
    for i in range(num_intervals):
        time.sleep(interval_sec)
        
        elapsed_time = time.time() - start_time
        mem_usage = get_memory_usage_mb()
        
        try:
            with open(log_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{elapsed_time:.2f}",
                    f"{mem_usage:.2f}"
                ])
            print(f"  > [Monitor] Logged memory: {mem_usage:.2f} MB at {elapsed_time:.0f}s")
        except IOError as e:
            print(f"  > [Monitor Error] Failed to log memory: {e}")

    print(f"  > [Monitor] Time limit of ~{time_limit_sec}s reached. Terminating process.")
    os._exit(0) # 정리 작업 없이 강제 종료

def save_geometry_data(k, q, points):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(directory): os.makedirs(directory)
    filename = os.path.join(directory, f"geometry_k{k}_q{q}.txt")
    with open(filename, "w") as f:
        f.write(f"# PG({k-1}, {q}) - {len(points)} points\n")
        for idx, p in enumerate(points): f.write(f"{idx}: {p}\n")

def save_experiment_results(n, k, q, weights, num_points, phase0_time, phase0_5_time, phase1_5_prep_time, search_time, phase2_time, total_sols, unique_sols, nodes_visited, pruned_nodes, status):
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
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n, k, q, str(sorted(list(weights))), num_points,
            f"{phase0_time:.4f}", f"{phase0_5_time:.4f}", f"{phase1_5_prep_time:.4f}", f"{search_time:.4f}", f"{phase2_time:.4f}", status,
            total_sols, unique_sols, nodes_visited, pruned_nodes
        ])
    print(f"  > [Logged] Results saved to '{filename}'")

def run_classification(n, k, q, weights_str, memory_monitoring=False):
    # --- Memory Monitoring Setup ---
    if memory_monitoring:
        time_limit_sec = 60 * 60  # 60분
        interval_sec = 1 * 60   # 1분
        
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        clean_weights_str = weights_str.replace(",", "-")
        log_filename = os.path.join(log_dir, f"mem_trial3_{n}_{k}_{q}_{clean_weights_str}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv")

        print(f"\n[*] Starting memory monitoring (limit: {time_limit_sec}s, interval: {interval_sec}s)")
        print(f"  > Log file: {log_filename}")
        
        monitor_thread = threading.Thread(
            target=monitor_and_log_memory,
            args=(log_filename, time_limit_sec, interval_sec),
            daemon=True
        )
        monitor_thread.start()

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
    solutions, nodes, pruned, p0_time, p0_5_time, p1_5_prep_time, p1_search_time, solve_status = extender.build_and_solve(points, hyperplanes)
    
    print(f"  > Enumeration finished.")
    print(f"  > Phase 0 (ILP Feasibility): {p0_time:.4f}s")
    print(f"  > Phase 0.5 (Theoretical Bounds): {p0_5_time:.4f}s")
    print(f"  > Phase 1.5 (Symmetry Prep): {p1_5_prep_time:.4f}s")
    print(f"  > Phase 1 (Backtrack Search): {p1_search_time:.4f}s")
    print(f"  > Candidates found: {len(solutions)}")
    print(f"  > Solver Status: {solve_status}")

    if not solutions:
        save_experiment_results(n, k, q, target_weights, len(points), p0_time, p0_5_time, p1_5_prep_time, p1_search_time, 0.0, 0, 0, nodes, pruned, solve_status)
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

    save_experiment_results(n, k, q, target_weights, len(points), p0_time, p0_5_time, p1_5_prep_time, p1_search_time, phase2_time, len(solutions), len(unique), nodes, pruned, solve_status)

if __name__ == "__main__":
    args = sys.argv[1:]
    memory_monitoring = False
    if "--monitor" in args:
        memory_monitoring = True
        args.remove("--monitor")
        print("[*] Memory monitoring enabled.")

    if len(args) == 4:
        run_classification(int(args[0]), int(args[1]), int(args[2]), args[3], memory_monitoring=memory_monitoring)
    else:
        print("Usage: python trial3/main.py [--monitor] <n> <k> <q> <weights>")
        print("Example: python trial3/main.py --monitor 7 3 2 \"3,4\"")
        print("Running default test: n=7, k=3, q=2, weights=3,4")
        run_classification(7, 3, 2, "3,4", memory_monitoring=memory_monitoring)
