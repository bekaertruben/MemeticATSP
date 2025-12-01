import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import types
from numba.typed import List
from typing import NamedTuple

"""
SUBTOURS:

This module provides functionality to manipulate and recombine subtours.
This is useful for crossover operators (e.g. EAX, GPX).
"""


# Subtour representation: (length, is_chain, start_node, end_node)
SubtourType = types.UniTuple(types.int64, 4)


# @jitclass([
#     ('tour', types.int64[:, :]),
#     ('distance_matrix', types.float64[:, :]),
#     ('subtours', types.ListType(SubtourType)),
#     ('subtour_ids', types.int64[:])
# ])

class Subtours(NamedTuple):
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
    tour: np.ndarray
    distance_matrix: np.ndarray
    subtours: List
    subtour_ids: np.ndarray


@njit(cache=True)
def make_subtour(tour, distance_matrix):
    return Subtours(
        tour,
        distance_matrix,
        List.empty_list(SubtourType),
        np.full(tour.shape[1], -1, dtype=np.int64)
    )


@njit(cache=True)
def initialize_subtour(self):
    _initialize(self.tour, self.subtours, self.subtour_ids)


@njit(cache=True)
def repair_subtour(self):
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
                # Mark this predecessor node as part of the same subtour
                if visited[prev_node]:
                    raise ValueError("Inconsistent predecessor and successor arrays detected.")
                visited[prev_node] = True
                subtour_ids[prev_node] = subtour_idx
                length += 1
                start = prev_node
        
        subtours.append((length, is_chain, start, end))
        subtour_idx += 1


@njit(cache=True)
def _merge_subtours(tour, distance_matrix, subtours, subtour_ids, s1, s2):
    """
    Merge two directed subtours (cycles) using directed 2-exchange.
    """
    length1, is_chain1, start1, end1 = subtours[s1]
    length2, is_chain2, start2, end2 = subtours[s2]
    
    assert not is_chain1 and not is_chain2, "Both subtours must be cycles to merge."
    
    best_delta = np.inf
    best_u = -1
    best_u_prime = -1
    best_v = -1
    best_v_prime = -1
    
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
                best_u = current1
                best_u_prime = next1
                best_v = current2
                best_v_prime = next2
            
            current2 = next2
        current1 = next1
        
    # Perform the 2-exchange
    tour[0, best_u] = best_v_prime
    tour[1, best_v_prime] = best_u
    tour[0, best_v] = best_u_prime
    tour[1, best_u_prime] = best_v
    
    # Update subtour_ids: merge s2 into s1's ID (use min for consistency)
    min_s, max_s = min(s1, s2), max(s1, s2)
    N = tour.shape[1]
    for i in range(N):
        if subtour_ids[i] == max_s:
            subtour_ids[i] = min_s
        elif subtour_ids[i] > max_s:
            subtour_ids[i] -= 1
    
    # Update subtours list
    subtours[min_s] = (length1 + length2, 0, best_u, best_u)
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
    for i in range(len(subtours)):
        if subtours[i][1] == 1:
            _close_chain(tour, subtours, i)

    # Merge until single tour
    pairs = _merge_indices(len(subtours))
    for i, j in pairs:
        # _initialize(tour, subtours, subtour_ids)
        _merge_subtours(tour, distance_matrix, subtours, subtour_ids, i, j)


@njit()
def _merge_indices(n):
    """
    Yield pairs (i, j) representing positions in a shrinking list.
    Combining (i, j) replaces element at i and removes element at j.
    """
    for _ in range(n - 1):
        yield (0, 1)
    return

    # Balanced merging:
    size = n
    while size > 1:
        i = 0
        while i < size - 1:
            yield (i, i + 1)
            size -= 1        # because we removed index (i+1)
            i += 1           # continue to next pair

