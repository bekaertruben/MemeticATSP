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


# Subtour representation: (length, is_chain, start_node, end_node)
SubtourType = types.UniTuple(types.int64, 4)

@jitclass([
    ('tour', types.int64[:, :]),
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
        subtours: List of subtours, each as
                    (length, is_chain, start_node, end_node)
        subtour_ids: Array mapping each node to its subtour index
    """
    def __init__(self, tour):
        self.tour = tour
        self.initialize()
    
    def initialize(self):
        """
        Extract all subtours from the permutation.

        If the tour's predecessor array is inconsistent with the successor array,
        weird behavior may arrise, such as multiple chains including the same node.
        """
        N = self.tour.shape[1]
        visited = np.zeros(N, dtype=np.bool_)
        self.subtour_ids = np.full(N, -1, dtype=np.int64)
        self.subtours = List.empty_list(SubtourType)
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
                self.subtour_ids[current] = subtour_idx
                length += 1

                next_node = self.tour[0, current]
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
                    prev_node = self.tour[1, start]
                    if prev_node < 0 or prev_node >= N:
                        break
                    start = prev_node
            
            self.subtours.append((length, is_chain, start, end))
            subtour_idx += 1
    
    def merge_subtours(self, s1, s2, distance_matrix):
        """
        Merge two directed subtours (cycles) using directed 2-exchange.
        
        s1 and s2 are indices into self.subtours list.
        Both must be cycles (not chains).
        
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
        assert s1 != s2, "Cannot merge a subtour with itself."
        assert self.subtours[s1][1] == 0, "Subtour 1 must be a cycle."
        assert self.subtours[s2][1] == 0, "Subtour 2 must be a cycle."
        
        succ = self.tour[0]
        pred = self.tour[1]
        
        st1 = self.subtours[s1]
        st2 = self.subtours[s2]
        s1_start = st1[2]
        s2_start = st2[2]
        
        # Collect nodes in each subtour
        nodes1 = List.empty_list(types.int64)
        current = s1_start
        while True:
            nodes1.append(current)
            current = succ[current]
            if current == s1_start:
                break
        
        nodes2 = List.empty_list(types.int64)
        current = s2_start
        while True:
            nodes2.append(current)
            current = succ[current]
            if current == s2_start:
                break
        
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
        
        # Update subtour_ids for nodes in both subtours to point to new index
        new_idx = len(self.subtours) - 2  # Will be at end after removing two
        for node in nodes1:
            self.subtour_ids[node] = new_idx
        for node in nodes2:
            self.subtour_ids[node] = new_idx
        
        # Create merged subtour entry
        new_length = st1[0] + st2[0]
        new_start = s1_start  # Can use either start since it's a cycle
        new_end = new_start   # For a cycle, end == start
        merged = (new_length, 0, new_start, new_end)  # is_chain = 0 (False)
        
        # Remove both subtours and add merged one
        # Remove higher index first to avoid shifting issues
        if s1 > s2:
            self.subtours.pop(s1)
            self.subtours.pop(s2)
        else:
            self.subtours.pop(s2)
            self.subtours.pop(s1)
        
        self.subtours.append(merged)
    
    def close_chain(self, chain_idx, distance_matrix):
        """
        Close a chain subtour into a cycle by connecting its end to its start.
        """
        length, is_chain, start_node, end_node = self.subtours[chain_idx]
        assert is_chain == 1, "Subtour must be a chain to be closed."
        
        # Connect end_node to start_node
        self.tour[0, end_node] = start_node
        self.tour[1, start_node] = end_node
        
        # Update subtour to be a cycle
        self.subtours[chain_idx] = (length, 0, start_node, start_node)


# @njit(cache=True) 
def repair_tour(tour, distance_matrix):
    """
    Repair a possibly fragmented tour into a single Hamiltonian cycle.
    
    First ensures all nodes have valid successors (connecting chains),
    then iteratively merges subtours until only one remains.
    """
    subtours = Subtours(tour)

    # Close chains
    # for i, st in enumerate(subtours.subtours):
    #     if st[1] == 1:
    #         subtours.close_chain(i, distance_matrix)

    # Merge subtours until only one remains
    while len(subtours.subtours) > 1:
        subtours.merge_subtours(0, 1, distance_matrix)