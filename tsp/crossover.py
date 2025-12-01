import numpy as np
from numba import njit
from numba.typed import List
from tsp.representation import Subtours, is_valid_tour, tour_cost, invert_permutation

"""
CROSSOVER OPERATORS:

This module implements crossover operators for ATSP tours:
1. EAX (Edge Assembly Crossover)
2. GPX (Generalized Partition Crossover)
3. MPX (Maximal Preservative Crossover)
4. ERX (Edge Recombination Crossover)
5. SCX (Sequential Constructive Crossover)
"""

@njit(cache=True)
def eax_find_AB_cycles(parent1, parent2):
    """
    Find all A-B cycles in the edge-difference graph for ATSP.
    
    Each cycle alternates: A-edge out (parent1) -> B-edge predecessor (parent2).
    Returns a list of node arrays, one per cycle.
    """
    N = parent1.shape[1]
    
    # Nodes where parents differ in outgoing edges
    differs = parent1[0] != parent2[0]
    visited = np.zeros(N, dtype=np.bool_)
    
    cycles = List()
    
    for start in range(N):
        if not differs[start] or visited[start]:
            continue
        
        # Trace cycle: follow A-out, then B-predecessor, until back at start
        cycle_nodes = List()
        current = start
        
        while differs[current] and not visited[current]:
            cycle_nodes.append(current)
            visited[current] = True
            # A-edge out -> B-edge predecessor
            current = parent2[1, parent1[0, current]]
        
        if len(cycle_nodes) >= 2 and current == start:
            nodes_arr = np.empty(len(cycle_nodes), dtype=np.int64)
            for i in range(len(cycle_nodes)):
                nodes_arr[i] = cycle_nodes[i]
            cycles.append(nodes_arr)
    
    return cycles


@njit(cache=True)
def EAX(parent1, parent2, distance_matrix, num_trials=1, num_cycles_to_select=2):
    """
    Edge Assembly Crossover (EAX) operator for ATSP tours.
    
    Based on Nagata's algorithm, EAX works by:
    1. Constructing A-B cycles from the edge difference between parents
    2. Selecting a random subset of cycles (the E-set)
    3. Applying the E-set: remove A-edges, add B-edges for selected cycles
    4. Repairing the fragmented result into a single Hamiltonian tour
    5. Generating multiple offspring and returning the best one
    
    Args:
        parent1: First parent tour, shape (2, N)
        parent2: Second parent tour, shape (2, N)
        distance_matrix: NxN asymmetric cost matrix
        num_trials: Number of offspring to generate (default 1)
        num_cycles_to_select: Maximum cycles to select per trial (default 2)
    
    Returns:
        best_child: The best offspring tour, shape (2, N)
        cost: Total cost of the best child (float) computed by tour_cost
    """
    N = parent1.shape[1]
    
    # Step 1: Find A-B cycles
    cycles = eax_find_AB_cycles(parent1, parent2)
    
    if len(cycles) == 0:
        # Parents are identical or very similar
        child = parent1.copy()
        return child, tour_cost(child, distance_matrix)
    
    best_child = None
    best_cost = np.inf
    
    for _trial in range(num_trials):
        # Step 2: Select random subset of cycles (E-set)
        num_cycs = len(cycles)
        k = min(num_cycles_to_select, num_cycs)
        if k > 1:
            k = np.random.randint(1, k + 1)
        
        indices = np.arange(num_cycs)
        np.random.shuffle(indices)
        selected_indices = indices[:k]
        
        # Start with copy of parent1
        child = parent1.copy()
        
        # Step 3: Apply E-set cycles
        for idx in selected_indices:
            cycle = cycles[idx]
            for i, u in enumerate(cycle):
                child[0, u] = parent2[0, u]
        child[1] = invert_permutation(child[0])
        
        # Step 4: Repair fragmented tour
        subtours = Subtours(child, distance_matrix)
        subtours.initialize()
        subtours.repair()
        assert is_valid_tour(subtours.tour), "Invalid tour after EAX repair."

        # Calculate cost via tour_cost
        cost = tour_cost(child, distance_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_child = child.copy()
    
    if best_child is None:
        child = parent1.copy()
        return child, tour_cost(child, distance_matrix)
    
    return best_child, best_cost