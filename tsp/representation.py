import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from numba.typed import List
from collections import namedtuple


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
def invert_permutation(p):
    """
    Invert a permutation represented as a 1D array.
    
    Args:
        p: 1D array of length N representing a permutation of {0, 1, ..., N-1}
    Returns:
        inv_p: 1D array of length N where inv_p[p[i]] = i
    """
    N = p.shape[0]
    inv_p = np.empty(N, dtype=np.int_)
    for i in range(N):
        inv_p[p[i]] = i
    return inv_p


# Subtour representation: (length, is_chain, start_node, end_node)
SubtourType = types.UniTuple(types.int64, 4)


@jitclass([
    ('tour', types.int64[:, :]),
    ('distance_matrix', types.float64[:, :]),
    ('subtours', types.ListType(SubtourType)),
    ('subtour_ids', types.int64[:])
])
class Subtours:
    """
    Class to manage subtours within a tour.
    The subtours types are:
    1. Cycles: closed loops where you can start at a node and return to it
    2. Chains: sequences that start and end without looping back

    Attributes:
        tour: The tour in adjacency representation (2, N)
        distance_matrix: The distance matrix used for cost calculations
        subtours: List of subtours, each as
                    (length, is_chain, start_node, end_node)
        subtour_ids: Array mapping each node to its subtour index
    """
    def __init__(self, tour, distance_matrix):
        self.tour = tour
        self.distance_matrix = distance_matrix
        self.subtours = List.empty_list(SubtourType)
        self.subtour_ids = np.full(tour.shape[1], -1, dtype=np.int64)
    
    def initialize(self):
        _initialize(self.tour, self.subtours, self.subtour_ids)
    
    def merge_subtours(self, s1, s2):
        _merge_subtours(self.tour, self.distance_matrix, self.subtours, self.subtour_ids, s1, s2)
    
    def close_chain(self, chain_idx):
        _close_chain(self.tour, self.subtours, chain_idx)
    
    def repair(self):
        _repair(self.tour, self.distance_matrix, self.subtours, self.subtour_ids)


@njit(cache=True)
def _initialize(tour, subtours, subtour_ids):
    """
    Extract all subtours from the permutation.
    
    If the tour's predecessor array is inconsistent with the successor array,
    weird behavior may arrise, such as multiple chains including the same node.
    """
    N = tour.shape[1]
    visited = np.zeros(N, dtype=np.bool_)
    subtours.clear()
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
            
            next_node = tour[0, current]
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
        
        subtours.append((length, is_chain, start, end))
        subtour_idx += 1


@njit(cache=True)
def _merge_subtours(tour, distance_matrix, subtours, subtour_ids, s1, s2):
    """
    Merge two directed subtours (cycles) using directed 2-exchange.
    s1 and s2 are indices for cycles in subtours.
    
    Finds the best way to break one edge from each subtour and reconnect
    them to form a single subtour, respecting edge directions.
    
    For subtours SA and SB with edges:
        u -> u' in SA
        v -> v' in SB
    
    We replace with:
        u -> v'
        v -> u'
    
    Modifies the tour in-place, removes both subtours from the list,
    and adds a new merged subtour.
    """
    length1, is_chain1, start1, end1 = subtours[s1]
    length2, is_chain2, start2, end2 = subtours[s2]
    
    assert not is_chain1 and not is_chain2, "Both subtours must be cycles to merge."
    
    best_delta = np.inf
    best_edges = None
    
    # Iterate over all edges in both subtours
    current1 = start1
    for _ in range(length1):
        next1 = tour[0, current1]
        current2 = start2
        for _ in range(length2):
            next2 = tour[0, current2]
            
            # Calculate cost difference
            delta = (distance_matrix[current1, next2] +
                     distance_matrix[current2, next1] -
                     distance_matrix[current1, next1] -
                     distance_matrix[current2, next2])
            
            if delta < best_delta:
                best_delta = delta
                best_edges = (current1, next1, current2, next2)
            
            current2 = next2
        current1 = next1
    
    u, u_prime, v, v_prime = best_edges
    # Perform the 2-exchange
    tour[0, u] = v_prime
    tour[1, v_prime] = u
    tour[0, v] = u_prime
    tour[1, u_prime] = v
    
    # Update subtours id array
    min_s, max_s = min(s1, s2), max(s1, s2)
    mask_new_subtour = (subtour_ids == s1) | (subtour_ids == s2)
    subtour_ids[mask_new_subtour] = min_s
    # subtracts 1 from all subtour ids greater than max(s1, s2)
    for i in range(len(subtours)):
        if i > max_s:
            mask = (subtour_ids == i)
            subtour_ids[mask] = i - 1
    
    # Add the new subtour info to the first subtour's position
    # and remove the second subtour
    subtours[min_s] = (length1 + length2, 0, start1, start1)
    subtours.pop(max_s)


@njit(cache=True)
def _close_chain(tour, subtours, chain_idx):
    """
    Close a chain subtour into a cycle by connecting its end to its start.
    """
    length, is_chain, start_node, end_node = subtours[chain_idx]
    assert is_chain == 1, "Subtour must be a chain to be closed."
    
    # Connect end_node to start_node
    tour[0, end_node] = start_node
    tour[1, start_node] = end_node
    
    # Update subtour to be a cycle
    subtours[chain_idx] = (length, 0, start_node, start_node)


@njit(cache=True)
def _repair(tour, distance_matrix, subtours, subtour_ids):
    """
    Repair the tour by closing chains and merging subtours until only one remains.
    """
    # Close all chains first
    for i, st in enumerate(subtours):
        if st[1] == 1:
            _close_chain(tour, subtours, i)
    
    # Merge subtours until only one remains 
    for i, j in _combine_indices(len(subtours)):
        _merge_subtours(tour, distance_matrix, subtours, subtour_ids, i, j)


@njit() # for some reason this errors when loading from cache if cache=True
def _combine_indices(n):
    """
    Yield pairs (i, j) representing positions in a shrinking list.
    Combining (i, j) replaces element at i and removes element at j.
    """
    size = n
    while size > 1:
        i = 0
        while i < size - 1:
            yield (i, i + 1)
            size -= 1        # because we removed index (i+1)
            i += 1           # continue to next pair
