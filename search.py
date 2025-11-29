import numpy as np
from numba import njit

"""
LOCAL SEARCH OPERATORS:

This module implements local search operators for TSP tours.
These operators modify tours in-place and return the total gain.
"""


@njit(cache=True)
def lso_3opt(tour, distance_matrix, candidates, max_iter=100):
    """
    Iterated local search using 3-opt moves in-place.
    
    Repeatedly applies improving 3-opt moves until no improvement is found.
    Returns the total gain (negative value means improvement).
    """
    total_gain = 0.0
    
    for _ in range(max_iter):
        gain = find_3opt_move(tour, distance_matrix, candidates)
        if gain < 0:
            total_gain += gain
        else:
            break  # No improvement found
    
    return total_gain


@njit(cache=True)
def find_3opt_move(tour, distance_matrix, candidates):
    """
    Find and apply an improving 3-opt move for a tour in (2, N) edge-map representation.
    
    Modifies tour in-place if an improving move is found.
    Returns the gain (negative if improved, 0 otherwise).
    """
    n = tour.shape[1]
    succ = tour[0]
    pred = tour[1]

    # Start from a random node to avoid bias
    start_node = np.random.randint(n)

    for i in range(n):
        a = (start_node + i) % n
        a_prime = succ[a]
        
        for j in range(candidates.shape[1]):
            b_prime = candidates[a, j]
            if b_prime == a or b_prime == a_prime:
                continue
            
            b = pred[b_prime]
            if a_prime == b or b_prime == a or succ[b_prime] == a:
                continue

            # Find best third cut
            best_c, best_score = -1, -np.inf
            c = succ[b_prime]
            while c != a:
                c_prime = succ[c]
                score = distance_matrix[c, c_prime] - distance_matrix[b, c_prime] - distance_matrix[c, a_prime]
                if score > best_score:
                    best_score, best_c = score, c
                c = c_prime

            if best_c == -1:
                continue

            # Compute change and apply if improving
            c = best_c
            c_prime = succ[c]
            change = (distance_matrix[a, b_prime] + distance_matrix[b, c_prime] + distance_matrix[c, a_prime]
                    - distance_matrix[a, a_prime] - distance_matrix[b, b_prime] - distance_matrix[c, c_prime])
            
            if change < 0:
                # Apply the 3-opt move in-place
                # New edges: a -> b', b -> c', c -> a'
                succ[a] = b_prime
                pred[b_prime] = a

                succ[b] = c_prime
                pred[c_prime] = b

                succ[c] = a_prime
                pred[a_prime] = c
                
                return change
    
    return 0.0


@njit(cache=True)
def precompute_candidates(distance_matrix, num_candidates=5, num_nn=5):
    """
    Precompute candidate edges using Sinkhorn assignment and Nearest Neighbors.
    """
    avg_dist = np.mean(distance_matrix)
    temp = avg_dist / 50.0
    S = sinkhorn_assignment(distance_matrix, temp)
    
    n = distance_matrix.shape[0]
    total_candidates = num_candidates + num_nn
    candidates = np.zeros((n, total_candidates), dtype=np.int32)
    
    for i in range(n):
        # Sort descending
        sinkhorn_indices = np.argsort(S[i, :])[::-1]
        # Sort ascending (nearest neighbors)
        nn_indices = np.argsort(distance_matrix[i, :])
        
        count = 0
        
        # Add Sinkhorn candidates
        for idx in sinkhorn_indices:
            if idx == i:
                continue
            if count >= num_candidates:
                break
            candidates[i, count] = idx
            count += 1
            
        # Add Nearest Neighbor candidates
        for idx in nn_indices:
            if idx == i:
                continue
            if count >= total_candidates:
                break
            
            # Check for duplicates
            is_duplicate = False
            for k in range(count):
                if candidates[i, k] == idx:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                candidates[i, count] = idx
                count += 1
        
    return candidates


@njit(cache=True)
def sinkhorn_assignment(D, temperature, max_iter=1000, threshold=1e-9):
    """
    Solve the soft assignment problem using Sinkhorn-Knopp algorithm.
    """
    K = np.exp(-D / temperature) # Gibbs Kernel
    
    u = np.ones(D.shape[0])
    v = np.ones(D.shape[1])
    
    for _ in range(max_iter):
        u_prev = u.copy()
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
        
        if np.allclose(u, u_prev, atol=threshold):
            break
            
    x_matrix = np.diag(u) @ K @ np.diag(v)
    
    return x_matrix