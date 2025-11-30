import numpy as np
from numba import njit
from numba.typed import List


"""
REPRESENTATION:

Tours of length N are represented as numpy arrays of shape (2, N)
where the first row represents the right edge adjacencies
and the second row represents the down edge adjacencies.
"""


@njit(cache=True)
def is_valid_tour(tour):
    """
    Check if a tour is valid by verifying:
    1. Starting from city 0, we return to city 0 after exactly N steps (not before)
    2. The inverse (down) adjacencies are consistent with the forward (right) adjacencies
    
    Returns True if the tour is valid, False otherwise.
    """
    N = tour.shape[1]
    
    # Check forward traversal: start at city 0, should return after exactly N steps
    current = 0
    for i in range(N):
        next_city = tour[0, current]
        
        # Check bounds
        if next_city < 0 or next_city >= N:
            return False
        
        # Check inverse consistency: if current -> next_city, then next_city's down should be current
        if tour[1, next_city] != current:
            return False
        
        # If we return to city 0 before N steps, invalid
        if next_city == 0 and i < N - 1:
            return False
        
        current = next_city
    
    # After N steps, we should be back at city 0
    if current != 0:
        return False
    
    return True


@njit(cache=True)
def tour_cost(tour, distance_matrix):
    """
    Calculate the total cost of a tour using the distance matrix.
    """
    s = 0.0
    for i, j in enumerate(tour[0]):
        s += distance_matrix[i, j]
    return s


@njit(cache=True)
def to_city_order(tour):
    """
    Convert the adjacency representation to a city order.
    
    Returns a 1D array of city indices in the order they are visited,
    starting from city 0.
    """
    N = tour.shape[1]
    order = np.empty(N, dtype=np.int_)
    
    current = 0
    for i in range(N):
        order[i] = current
        current = tour[0, current]
    
    return order


@njit(cache=True)
def hamming_distance(tour1, tour2):
    """
    Compute the Hamming distance between two tours in adjacency representation.
    
    Returns the number of differing edges.
    """
    diff = 0
    N = tour1.shape[1]
    for i in range(N):
        if tour1[0, i] != tour2[0, i]:
            diff += 1
    return diff


@njit(cache=True)
def extract_subtours(tour):
    """
    Extract all subtours from the permutation.
    These are:
    1. Cycles: closed loops where you can start at a node and return to it
    2. Chains: sequences that start and end without looping back

    If the predecessor array is inconsistent with the successor array,
    there might be weird behavior, such as multiple chains including the same node.
    
    Returns:
        subtours: List of numpy arrays, each containing [start_node, end_node, length, is_chain]
        subtour_ids: Array mapping each node to its subtour index
    """
    succ = tour[0]
    N = succ.shape[0]
    visited = np.zeros(N, dtype=np.bool_)
    subtour_ids = np.full(N, -1, dtype=np.int64)
    subtours = List()
    subtour_idx = 0
    
    for i in range(N):
        if visited[i]:
            continue
        
        length = 0
        is_chain = False

        # Trace this cycle/chain
        current = i
        for _ in range(N + 1):
            if visited[current]:
                raise ValueError("Inconsistent predecessor and successor arrays detected.")

            visited[current] = True
            subtour_ids[current] = subtour_idx
            length += 1

            next_node = succ[current]
            if next_node < 0 or next_node >= N:
                is_chain = True
                end = current
                break  # End of chain (no valid successor)
            current = next_node
            if current == i:
                end = current
                break  # Completed the cycle
        
        # Trace the chain backwards to mark all nodes
        start = i
        if is_chain:
            while True:
                prev_node = tour[1, start]
                if prev_node < 0 or prev_node >= N:
                    break
                start = prev_node
        
        subtours.append(np.array([start, end, length, is_chain], dtype=np.int64))
        subtour_idx += 1
    
    return subtours, subtour_ids


@njit(cache=True)
def merge_subtours(tour, s1_start, s2_start, distance_matrix):
    """
    Merge two directed subtours using directed 2-exchange.
    
    Finds the best way to break one edge from each subtour and reconnect
    them to form a single subtour, respecting edge directions.
    
    For subtours SA and SB with edges:
        u -> u' in SA
        v -> v' in SB
    
    We replace with:
        u -> v'
        v -> u'
    
    This merges the two subtours into one while preserving direction.
    
    Handles incomplete subtours (chains) where some nodes may not have
    valid successors.
    """
    if s1_start == s2_start:
        return tour
    
    succ = tour[0]
    pred = tour[1]
    N = succ.shape[0]

    # Collect nodes in each subtour
    nodes1 = List()
    current = s1_start
    while True:
        nodes1.append(current)
        current = succ[current]
        if current == s1_start:
            break
    
    nodes2 = List()
    current = s2_start
    while True:
        nodes2.append(current)
        current = succ[current]
        if current == s2_start:
            break
    
    if len(nodes1) == 0 or len(nodes2) == 0:
        return tour
    
    # Find best directed merge
    best_delta = np.inf
    best_u, best_v = -1, -1
    for u in nodes1:
        u_prime = succ[u]
        for v in nodes2:
            v_prime = succ[v]
            
            current_cost = distance_matrix[u, u_prime] + distance_matrix[v, v_prime]
            new_cost = distance_matrix[u, v_prime] + distance_matrix[v, u_prime]
            delta = new_cost - current_cost
            
            if delta < best_delta:
                best_delta = delta
                best_u = u
                best_v = v

    # For edges u->u' in S1 and v->v' in S2
    # Replace with u->v' and v->u'
    u_prime = succ[best_u]
    v_prime = succ[best_v]
    
    # Apply the merge: u->v' and v->u'
    succ[best_u] = v_prime
    pred[v_prime] = best_u
    succ[best_v] = u_prime
    pred[u_prime] = best_v


@njit(cache=True) 
def repair_tour(tour, distance_matrix):
    """
    Repair a possibly fragmented tour into a single Hamiltonian cycle.
    
    First ensures all nodes have valid successors (connecting chains),
    then iteratively merges subtours until only one remains.
    """
    succ = tour[0]
    pred = tour[1]
    N = len(succ)
    
    # Step 1: Link up chains to ensure all nodes have valid successors
    subtours, _ = extract_subtours(tour)
    for st in subtours:
        is_chain = st[3]
        if is_chain:
            start_node = st[0]
            end_node = st[1]
            
            # Close the cycle by connecting end_node to start_node
            succ[end_node] = start_node
            pred[start_node] = end_node

    # Step 2: Now merge subtours (all nodes should have valid successors)
    for _ in range(N):
        subtours, _ = extract_subtours(tour)
        if len(subtours) <= 1:
            break
        
        s1 = subtours[0][0]
        s2 = subtours[1][0]
        tour = merge_subtours(tour, s1, s2, distance_matrix)
        succ = tour[0]
        pred = tour[1]
    
    tour[0] = succ
    tour[1] = pred
    return tour