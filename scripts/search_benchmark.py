#!/usr/bin/env python3
"""
Test script for comparing convergence and runtime of local search operators.

Compares: 2-opt, 3-opt, Or-opt via unified lso function
"""

import numpy as np
import time
import os
from tsp.search import (
    lso, LSO_2OPT, LSO_3OPT, LSO_OROPT,
    find_2opt_move, find_3opt_move, find_oropt_move,
    precompute_candidates
)
from tsp.representation import tour_cost, is_valid_tour
from tsp.greedy import greedy_cycle

tour_files = [
    f"tours/tour{n}.csv"
    for n in [50, 250, 500, 750, 1000]
]

def load_distance_matrix(filepath):
    """Load a distance matrix from a CSV file."""
    return np.loadtxt(filepath, delimiter=',')


def create_greedy_tour(distance_matrix, temp=1.0):
    """Create a tour using greedy cycle heuristic, retrying until successful."""
    while True:
        tour = greedy_cycle(distance_matrix, temp)
        if tour is not None:
            return tour


def benchmark_operator(method, tour, distance_matrix, candidates, 
                       name, max_iter=1000):
    """Benchmark a single local search operator."""
    tour_copy = tour.copy()
    
    cost_before = tour_cost(tour_copy, distance_matrix)
    
    start_time = time.perf_counter()
    total_gain = lso(tour_copy, distance_matrix, candidates, 
                     method=method, max_iter=max_iter)
    elapsed = time.perf_counter() - start_time
    
    cost_after = tour_cost(tour_copy, distance_matrix)
    valid = is_valid_tour(tour_copy)
    
    # Verify gain consistency
    actual_change = cost_after - cost_before
    gain_matches = abs(total_gain - actual_change) < 1e-6
    
    return {
        'name': name,
        'cost_before': cost_before,
        'cost_after': cost_after,
        'improvement': cost_before - cost_after,
        'improvement_pct': 100 * (cost_before - cost_after) / cost_before,
        'reported_gain': total_gain,
        'gain_matches': gain_matches,
        'valid': valid,
        'time_ms': elapsed * 1000,
    }


def run_benchmark(tour_file, num_trials=5, seed=None):
    """Run benchmarks for all operators on a tour instance."""
    if seed is not None:
        np.random.seed(seed)
    
    distance_matrix = load_distance_matrix(tour_file)
    n = distance_matrix.shape[0]
    
    print(f"\n{'='*70}")
    print(f"Benchmark: {os.path.basename(tour_file)} (n={n}), trials={num_trials}")
    print(f"{'='*70}")
    
    results = {
        '2-opt': [],
        '3-opt': [],
        'Or-opt': [],
    }
    
    # Precompute candidates once per instance
    candidates = precompute_candidates(distance_matrix)
    
    for trial in range(num_trials):
        # Create initial tour using greedy cycle
        tour = create_greedy_tour(distance_matrix, temp=1.0)
        
        # Benchmark each operator
        results['2-opt'].append(
            benchmark_operator(LSO_2OPT, tour, distance_matrix, candidates, '2-opt')
        )
        results['3-opt'].append(
            benchmark_operator(LSO_3OPT, tour, distance_matrix, candidates, '3-opt')
        )
        results['Or-opt'].append(
            benchmark_operator(LSO_OROPT, tour, distance_matrix, candidates, 'Or-opt')
        )
    
    return results


def print_summary(results):
    """Print summary statistics for benchmark results."""
    print(f"\n{'Operator':<15} {'Avg Improv %':>12} {'Std %':>8} {'Avg Time (ms)':>14} {'Valid':>6} {'Gain OK':>8}")
    print("-" * 70)
    
    for name, trials in results.items():
        avg_improv = np.mean([t['improvement_pct'] for t in trials])
        std_improv = np.std([t['improvement_pct'] for t in trials])
        avg_time = np.mean([t['time_ms'] for t in trials])
        all_valid = all(t['valid'] for t in trials)
        all_gain_ok = all(t['gain_matches'] for t in trials)
        
        print(f"{name:<15} {avg_improv:>12.2f} {std_improv:>8.2f} {avg_time:>14.2f} "
              f"{'✓' if all_valid else '✗':>6} {'✓' if all_gain_ok else '✗':>8}")


def convergence_analysis(tour_file, seed=42):
    """Analyze convergence by tracking cost after each iteration."""
    np.random.seed(seed)
    
    distance_matrix = load_distance_matrix(tour_file)
    n = distance_matrix.shape[0]
    
    print(f"\n{'='*70}")
    print(f"Convergence Analysis: {os.path.basename(tour_file)} (n={n})")
    print(f"{'='*70}")
    
    candidates = precompute_candidates(distance_matrix)
    
    operators = [
        ('2-opt', LSO_2OPT, find_2opt_move),
        ('3-opt', LSO_3OPT, find_3opt_move),
        ('Or-opt 5', LSO_OROPT, lambda t,d,c: find_oropt_move(t,d,c,5)),
        ('Or-opt 20', LSO_OROPT, lambda t,d,c: find_oropt_move(t,d,c,20)),
        ('Or-opt 100', LSO_OROPT, lambda t,d,c: find_oropt_move(t,d,c,100)),
    ]
    
    print(f"\n{'Operator':<12} {'Initial':>10} {'Final':>10} {'Iters':>6} {'Time (ms)':>10}")
    print("-" * 55)
    
    for name, method, find_move in operators:
        tour = create_greedy_tour(distance_matrix, temp=1.0)
        initial_cost = tour_cost(tour, distance_matrix)
        
        # Run with iteration tracking
        total_iters = 0
        start_time = time.perf_counter()
        
        for _ in range(10_000):
            gain = find_move(tour, distance_matrix, candidates)
            
            if gain >= 0:
                break
            total_iters += 1
        
        elapsed = time.perf_counter() - start_time
        final_cost = tour_cost(tour, distance_matrix)
        
        print(f"{name:<12} {initial_cost:>10.2f} {final_cost:>10.2f} "
              f"{total_iters:>6} {elapsed*1000:>10.2f}")


def scaling_analysis(num_trials=3, seed=42):
    """Analyze how runtime scales with problem size using available tour files."""
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print("Scaling Analysis: Runtime vs Problem Size")
    print(f"{'='*70}")
    
    operators = [
        ('2-opt', LSO_2OPT),
        ('3-opt', LSO_3OPT),
        ('Or-opt', LSO_OROPT),
    ]
    
    # Get sizes from tour files
    sizes = []
    for f in tour_files:
        dm = load_distance_matrix(f)
        sizes.append(dm.shape[0])
    
    # Header
    print(f"\n{'Operator':<12}", end='')
    for n in sizes:
        print(f"{'n='+str(n):>12}", end='')
    print()
    print("-" * (12 + 12 * len(sizes)))
    
    for name, method in operators:
        print(f"{name:<12}", end='')
        
        for tour_file in tour_files:
            distance_matrix = load_distance_matrix(tour_file)
            candidates = precompute_candidates(distance_matrix)
            
            times = []
            for _ in range(num_trials):
                tour = create_greedy_tour(distance_matrix, temp=1.0)
                
                start = time.perf_counter()
                lso(tour, distance_matrix, candidates, method=method, max_iter=500)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            print(f"{avg_time:>10.1f}ms", end='')
        print()


def main():
    print("Local Search Operators - Benchmark Suite")
    print("=" * 70)
    
    if not tour_files:
        print("Error: No tour files found in tours/ directory")
        return
    
    print(f"\nFound {len(tour_files)} tour files:")
    for f in tour_files:
        dm = load_distance_matrix(f)
        print(f"  - {os.path.basename(f)} (n={dm.shape[0]})")
    
    # Warm up JIT compilation using smallest tour
    print("\nWarming up JIT compilation...")
    dm = load_distance_matrix(tour_files[0])
    cand = precompute_candidates(dm)
    tour = create_greedy_tour(dm, temp=1.0)
    
    lso(tour.copy(), dm, cand, method=LSO_2OPT, max_iter=10)
    lso(tour.copy(), dm, cand, method=LSO_3OPT, max_iter=10)
    lso(tour.copy(), dm, cand, method=LSO_OROPT, max_iter=10)
    print("JIT compilation complete.")
    
    # Run benchmarks on each tour file
    for tour_file in tour_files:
        results = run_benchmark(tour_file, num_trials=5, seed=42)
        print_summary(results)
    
    # Convergence analysis on each tour
    for tour_file in tour_files:
        convergence_analysis(tour_file, seed=42)
    
    # Scaling analysis
    scaling_analysis(num_trials=3, seed=42)
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
