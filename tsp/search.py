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
def lso_2opt(tour, distance_matrix, candidates, max_iter=100):
    """
    Iterated local search using 2-opt moves in-place.
    
    For ATSP, uses crossing-edge elimination without segment reversal.
    Repeatedly applies improving 2-opt moves until no improvement is found.
    Returns the total gain (negative value means improvement).
    """
    total_gain = 0.0
    
    for _ in range(max_iter):
        gain = find_2opt_move(tour, distance_matrix, candidates)
        if gain < 0:
            total_gain += gain
        else:
            break  # No improvement found
    
    return total_gain


@njit(cache=True)
def find_2opt_move(tour, distance_matrix, candidates):
    """
    Find and apply an improving 2-opt move for ATSP.
    
    Replaces edges (a -> a') and (b -> b') with (a -> b) and (a' -> b'),
    reversing the segment from a' to b. For ATSP, the cost of traversing
    the reversed segment in the opposite direction must be calculated.
    
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
        
        # Traverse from a' and accumulate reversal cost incrementally
        # reversal_delta = cost of reversed edges - cost of original edges
        reversal_delta = 0.0
        prev_node = a_prime
        curr_node = succ[a_prime]
        
        while curr_node != a:
            # Update reversal cost: edge prev_node -> curr_node becomes curr_node -> prev_node
            reversal_delta += distance_matrix[curr_node, prev_node] - distance_matrix[prev_node, curr_node]
            
            b = curr_node
            b_prime = succ[b]
            
            # Check if b is a candidate for a
            is_candidate = False
            for j in range(candidates.shape[1]):
                if candidates[a, j] == b:
                    is_candidate = True
                    break
            
            if is_candidate and b_prime != a:
                # Compute total change
                change = (distance_matrix[a, b] + distance_matrix[a_prime, b_prime]
                        - distance_matrix[a, a_prime] - distance_matrix[b, b_prime]
                        + reversal_delta)
                
                if change < 0:
                    # Reverse the path from a' to b by swapping succ/pred
                    node = a_prime
                    while node != b_prime:
                        next_node = succ[node]
                        succ[node], pred[node] = pred[node], succ[node]
                        node = next_node
                    
                    # Reconnect: a -> b, a' -> b'
                    succ[a] = b
                    pred[b] = a
                    succ[a_prime] = b_prime
                    pred[b_prime] = a_prime
                    
                    return change
            
            prev_node = curr_node
            curr_node = succ[curr_node]
    
    return 0.0


@njit(cache=True)
def lso_oropt(tour, distance_matrix, candidates, max_iter=100, max_segment_size=3):
    """
    Iterated local search using Or-opt moves in-place.
    
    Or-opt relocates segments of 1, 2, or 3 consecutive cities to a new position.
    This preserves traversal direction, making it ATSP-compatible.
    Repeatedly applies improving Or-opt moves until no improvement is found.
    Returns the total gain (negative value means improvement).
    """
    total_gain = 0.0
    
    for _ in range(max_iter):
        gain = find_oropt_move(tour, distance_matrix, candidates, max_segment_size)
        if gain < 0:
            total_gain += gain
        else:
            break  # No improvement found
    
    return total_gain


@njit(cache=True)
def find_oropt_move(tour, distance_matrix, candidates, max_segment_size=3):
    """
    Find and apply an improving Or-opt move for a tour.
    
    Relocates a segment of 1 to max_segment_size consecutive cities to a new position.
    The segment is removed from its current location and inserted after a target node.
    Direction is preserved, making this suitable for asymmetric TSP.
    
    Modifies tour in-place if an improving move is found.
    Returns the gain (negative if improved, 0 otherwise).
    """
    n = tour.shape[1]
    succ = tour[0]
    pred = tour[1]

    # Start from a random node to avoid bias
    start_node = np.random.randint(n)

    # Pre-allocate array for segment nodes (max 3 nodes)
    seg_nodes = np.zeros(max_segment_size, dtype=np.int64)

    for i in range(n):
        seg_start = (start_node + i) % n
        seg_pred = pred[seg_start]  # Node before segment
        
        # Try different segment sizes (1, 2, 3)
        seg_end = seg_start
        seg_nodes[0] = seg_start
        
        for seg_size in range(1, max_segment_size + 1):
            seg_succ = succ[seg_end]  # Node after segment
            
            # Skip if segment wraps around to start
            if seg_succ == seg_start:
                break
            
            # Cost of removing segment from current position
            # Remove edges: seg_pred -> seg_start and seg_end -> seg_succ
            # Add edge: seg_pred -> seg_succ
            removal_cost = (distance_matrix[seg_pred, seg_succ]
                          - distance_matrix[seg_pred, seg_start]
                          - distance_matrix[seg_end, seg_succ])
            
            # Try inserting segment after each candidate node
            for j in range(candidates.shape[1]):
                target = candidates[seg_start, j]
                
                # Skip if target is adjacent to segment
                if target == seg_pred or target == seg_succ:
                    continue
                
                # Check if target is in the segment using pre-collected nodes
                in_segment = False
                for k in range(seg_size):
                    if seg_nodes[k] == target:
                        in_segment = True
                        break
                if in_segment:
                    continue
                
                target_succ = succ[target]
                
                # Skip if target_succ is the segment start (would create invalid tour)
                if target_succ == seg_start:
                    continue
                
                # Cost of inserting segment after target
                # Remove edge: target -> target_succ
                # Add edges: target -> seg_start and seg_end -> target_succ
                insertion_cost = (distance_matrix[target, seg_start]
                                + distance_matrix[seg_end, target_succ]
                                - distance_matrix[target, target_succ])
                
                change = removal_cost + insertion_cost
                
                if change < 0:
                    # Apply the Or-opt move in-place
                    
                    # Step 1: Remove segment from current position
                    # Connect seg_pred directly to seg_succ
                    succ[seg_pred] = seg_succ
                    pred[seg_succ] = seg_pred
                    
                    # Step 2: Insert segment after target
                    # target -> seg_start -> ... -> seg_end -> target_succ
                    succ[target] = seg_start
                    pred[seg_start] = target
                    succ[seg_end] = target_succ
                    pred[target_succ] = seg_end
                    
                    return change
            
            # Move to next segment size - update seg_end and record node
            seg_end = succ[seg_end]
            if seg_end == seg_start:
                break
            if seg_size < max_segment_size:
                seg_nodes[seg_size] = seg_end
    
    return 0.0


@njit(cache=True)
def precompute_candidates(distance_matrix, num_candidates=10, num_nn=5):
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
