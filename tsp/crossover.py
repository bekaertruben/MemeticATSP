import numpy as np
from numba import njit
from numba.typed import List
from tsp.representation import is_valid_tour, tour_cost, repair_tour

"""
CROSSOVER OPERATORS:

This module implements crossover operators for ATSP tours:
1. EAX (Edge Assembly Crossover)
2. GPX (Generalized Partition Crossover)
"""

@njit(cache=True)
def eax_find_AB_cycles(parent1, parent2):
    """
    Find all independent A-B cycles in the edge-difference graph for ATSP.
    
    An A-B cycle is constructed by alternating between:
    - Following an A-edge (outgoing edge from parent1)
    - Following a B-edge (outgoing edge from parent2)
    
    For ATSP, at each node we:
    1. Follow A-edge OUT: current -> parent1[0, current]
    2. From that destination, follow B-edge OUT: dest -> parent2[0, dest]
    3. Continue until we return to start
    
    However, to ensure cycles close properly in the difference graph,
    we use the construction: A-out then B-predecessor, which traces
    through the symmetric difference structure.
    
    Returns:
        cycles: List of (nodes_array, edge_types_array) pairs
                where edge_types[i] = 0 for A-edge, 1 for B-edge
    """
    N = parent1.shape[1]
    
    # Find nodes where parents have different outgoing edges
    diff_out = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if parent1[0, i] != parent2[0, i]:
            diff_out[i] = True
    
    # Track which A-edges have been used
    used_a_edges = np.zeros(N, dtype=np.bool_)
    
    cycles = List()
    
    for start in range(N):
        if not diff_out[start] or used_a_edges[start]:
            continue
        
        # Build a cycle by alternating: A-out, then B-predecessor
        cycle_nodes = List()
        cycle_edges = List()  # 0 = A-edge, 1 = B-edge
        current = start
        
        for _ in range(2 * N):
            if used_a_edges[current]:
                break
            if not diff_out[current]:
                break
            
            cycle_nodes.append(current)
            used_a_edges[current] = True
            
            # Follow A-edge out: current -> parent1[0, current]
            a_dest = parent1[0, current]
            cycle_edges.append(np.int64(0))  # A-edge
            
            # Find B-predecessor of a_dest (node whose B-edge points to a_dest)
            b_pred = parent2[1, a_dest]
            cycle_edges.append(np.int64(1))  # B-edge (implicit)
            
            current = b_pred
            
            if current == start:
                break
        
        if len(cycle_nodes) >= 2 and current == start:
            nodes_arr = np.empty(len(cycle_nodes), dtype=np.int64)
            for i in range(len(cycle_nodes)):
                nodes_arr[i] = cycle_nodes[i]
            cycles.append(nodes_arr)
    
    return cycles


@njit(cache=True)
def eax_apply_eset(tour, cycle_nodes, parent2):
    """
    Apply the E-set to parent A: for each node in the cycle,
    remove its A-edge and add the corresponding B-edge.
    
    For each node u in cycle_nodes:
    - Remove edge u -> succ[u] (current A-edge)
    - Add edge u -> parent2[0, u] (B-edge)
    
    This modifies succ and pred in place.
    """
    succ = tour[0]
    pred = tour[1]
    N = len(succ)
    
    # For each node in the cycle, swap its outgoing edge to parent2's edge
    for i in range(len(cycle_nodes)):
        u = cycle_nodes[i]
        succ[u] = parent2[0, u]
    
    # Rebuild predecessor array from scratch
    for i in range(N):
        pred[succ[i]] = i
    tour[0] = succ
    tour[1] = pred


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
        tour = parent1.copy()
        
        # Step 3: Apply E-set (remove A-edges, add B-edges for selected cycles)
        for idx in selected_indices:
            cycle = cycles[idx]
            eax_apply_eset(tour, cycle, parent2)
        
        # Step 4: Repair if fragmented
        if not is_valid_tour(tour):
            tour = repair_tour(tour, distance_matrix)
        
        # Build child tour
        child = tour

        # Calculate cost via tour_cost
        cost = tour_cost(child, distance_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_child = child.copy()
    
    if best_child is None:
        child = parent1.copy()
        return child, tour_cost(child, distance_matrix)
    
    return best_child, best_cost


@njit(cache=True)
def GPX(parent1, parent2, distance_matrix):
    """
    Generalized Partition Crossover (GPX) for ATSP tours.
    
    GPX preserves common edges between parents and resolves differing
    edges by constructing a valid tour through greedy patching.
    
    The algorithm:
    1. Copy all common edges (edges that exist in both parents)
    2. For differing edges, use a greedy nearest-neighbor approach
       to connect the fragments while respecting tour constraints
    
    Unlike EAX, GPX:
    - Is deterministic (no random trials)
    - Preserves all common substructures from both parents
    - Is simpler and faster
    - Is less exploratory but more stable
    
    Args:
        parent1: First parent tour, shape (2, N)
        parent2: Second parent tour, shape (2, N)  
        distance_matrix: NxN asymmetric cost matrix
    
    Returns:
        child: The offspring tour, shape (2, N)
        cost: Total cost of the child (float) computed by tour_cost
    """
    N = parent1.shape[1]
    
    # Check if parents are identical
    if np.array_equal(parent1[0], parent2[0]):
        child = parent1.copy()
        cost = tour_cost(child, distance_matrix)
        return child, cost
    
    # Initialize tour (succ + pred)
    tour = np.empty((2, N), dtype=np.int64)
    tour[0] = np.full(N, -1, dtype=np.int64)
    tour[1] = np.full(N, -1, dtype=np.int64)
    
    # Step 1: Copy all common edges
    for i in range(N):
        if parent1[0, i] == parent2[0, i]:
            j = parent1[0, i]
            tour[0, i] = j
            tour[1, j] = i
    
    # Step 2: For nodes without outgoing edges, choose the better parent edge
    # if it doesn't create a conflict
    for i in range(N):
        if tour[0, i] >= 0:
            continue  # Already has outgoing edge
        
        # Try parent1's edge first
        p1_next = parent1[0, i]
        p2_next = parent2[0, i]
        
        # Check which destination is available (doesn't have incoming edge yet)
        p1_available = tour[1, p1_next] < 0
        p2_available = tour[1, p2_next] < 0
        
        if p1_available and p2_available:
            # Both available, choose cheaper edge
            if distance_matrix[i, p1_next] <= distance_matrix[i, p2_next]:
                tour[0, i] = p1_next
                tour[1, p1_next] = i
            else:
                tour[0, i] = p2_next
                tour[1, p2_next] = i
        elif p1_available:
            tour[0, i] = p1_next
            tour[1, p1_next] = i
        elif p2_available:
            tour[0, i] = p2_next
            tour[1, p2_next] = i
        # If neither is available, we'll handle it in the repair step
    
    # Step 3: Greedy repair for remaining unassigned edges
    # Find nodes without outgoing edges and nodes without incoming edges
    for iteration in range(N):
        # Find a node without outgoing edge
        source = -1
        for i in range(N):
            if tour[0, i] < 0:
                source = i
                break
        
        if source < 0:
            break  # All nodes have outgoing edges
        
        # Find best available target (node without incoming edge)
        best_target = -1
        best_cost = np.inf
        
        for j in range(N):
            if tour[1, j] < 0 and j != source:
                cost = distance_matrix[source, j]
                if cost < best_cost:
                    best_cost = cost
                    best_target = j
        
        if best_target >= 0:
            tour[0, source] = best_target
            tour[1, best_target] = source
    
    # Step 4: Final check - repair any remaining issues
    if not is_valid_tour(tour):
        tour = repair_tour(tour, distance_matrix)
    
    # Build child tour
    child = tour
    
    cost = tour_cost(child, distance_matrix)
    return child, cost