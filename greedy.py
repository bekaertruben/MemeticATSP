import numpy as np
from numba import njit

"""
GREEDY HEURISTIC:

This implements a greedy heuristic to construct an initial tour for the TSP.
The heuristic builds a tour by starting from a random city and repeatedly
adding the nearest unvisited city until all cities are included.
The temperature parameter allows for stochasticity in the selection process,
enabling exploration of different tours.
"""


@njit(cache=True)
def softmax(x, temp=1.0):
    max_x = np.max(x)
    exps = np.exp((x - max_x) / temp)
    return exps / np.sum(exps)


@njit(cache=True)
def weighted_choice(n, probs):
    """
    Performs a weighted random choice, compatible with Numba.
    Assumes that probs sum to 1.
    """
    rand = np.random.random()
    weight_sum = 0.
    for i in range(n):
        weight_sum += probs[i]
        if rand <= weight_sum:
            return i
    return n - 1


@njit(cache=True)
def greedy_cycle(distance_matrix, temp=0):
    """
    Initializes a single tour using a stochastic nearest neighbor heuristic.
    
    Returns a tour as a (2, N) array where:
    - tour[0, i] is the right neighbor (next city) of city i
    - tour[1, i] is the down neighbor (previous city) of city i
    """
    N = distance_matrix.shape[0]
    
    # Initialize tour with (2, N) representation
    tour = np.zeros((2, N), dtype=np.int_) + N
    available = np.ones(N, dtype=np.bool_)

    start = np.random.randint(N)
    available[start] = False

    current_city = start
    for _ in range(N - 1):
        candidates = np.where(available)[0]
        distances = distance_matrix[current_city, candidates]
        
        # Filter out infinite distances
        valid_mask = distances != np.inf
        if not np.any(valid_mask):
            return None
            
        valid_candidates = candidates[valid_mask]
        valid_distances = distances[valid_mask]

        if temp == 0:
            idx = np.argmin(valid_distances)
        elif temp == np.inf:
            idx = np.random.randint(len(valid_candidates))
        else:
            # Normalize distances by their mean to make temp scale-invariant
            mean_distance = np.mean(valid_distances)
            scaled_distances = valid_distances if mean_distance == 0 else valid_distances / mean_distance
            
            probs = np.exp(-scaled_distances / temp)
            probs /= np.sum(probs)
            idx = weighted_choice(len(valid_candidates), probs)
        
        next_city = valid_candidates[idx]

        # Set adjacencies: current_city -> next_city (right) and next_city <- current_city (down)
        tour[0, current_city] = next_city
        tour[1, next_city] = current_city
        available[next_city] = False
        current_city = next_city
    
    if distance_matrix[current_city, start] == np.inf:
        return None
    
    # Close the cycle: last city -> start and start <- last city
    tour[0, current_city] = start
    tour[1, start] = current_city
    return tour