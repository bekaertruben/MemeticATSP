import numpy as np
from numba import njit
from numba.typed import List

"""
CROSSOVER OPERATOR:

This module implements the Edge Assembly Crossover (EAX) operator for ATSP tours.
"""


@njit(cache=True)
def _find_AB_cycles(parent1, parent2):
    """
    Find all independent A-B cycles in the edge-difference graph for ATSP.
    
    For ATSP, we need to consider both outgoing and incoming edges.
    An A-B cycle traces a closed path where at each step:
    - We follow an A-edge OUT from the current node
    - Then we follow a B-edge that ENTERS the next position (i.e., go to the 
      node whose B-edge points here, which is the B-predecessor)
    
    This corresponds to: from u, go to parent1[0,u] (A-edge out),
    then go to parent2[1, parent1[0,u]] (B-predecessor of where we landed).
    
    This traces through the symmetric difference graph and naturally forms cycles
    because we're following the "flow" of edge differences.
    
    Returns:
        cycles: List of numpy arrays, each containing the nodes in a cycle
    """
    N = parent1.shape[1]
    
    # Find nodes where parents have different successors OR different predecessors
    # A node is in the diff if either its outgoing or incoming edge differs
    diff_out = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if parent1[0, i] != parent2[0, i]:
            diff_out[i] = True
    
    # Track which A-edges (outgoing from parent1) have been used
    used = np.zeros(N, dtype=np.bool_)
    
    cycles = List()
    
    for start in range(N):
        if not diff_out[start] or used[start]:
            continue
        
        # Build a cycle by following: A-out, then B-in (predecessor), repeat
        cycle_nodes = List()
        current = start
        
        for _ in range(N + 1):
            if used[current]:
                break
            if not diff_out[current]:
                break
            
            cycle_nodes.append(current)
            used[current] = True
            
            # Follow A-edge out: current -> parent1[0, current]
            a_dest = parent1[0, current]
            
            # Now find where to go next: we need to find the node whose
            # B-edge (parent2) points to a_dest. That's parent2[1, a_dest]
            b_pred = parent2[1, a_dest]
            
            current = b_pred
            
            if current == start:
                break
        
        if len(cycle_nodes) >= 2 and current == start:
            cycle_arr = np.empty(len(cycle_nodes), dtype=np.int64)
            for i, node in enumerate(cycle_nodes):
                cycle_arr[i] = node
            cycles.append(cycle_arr)
    
    return cycles


@njit(cache=True)
def _apply_swap(succ, pred, nodes_to_swap, parent2):
    """
    Apply edge swaps: for each node in nodes_to_swap, 
    change its successor from current to parent2's successor.
    
    This modifies succ and pred in place.
    """
    N = len(succ)
    
    # First, record all the changes we need to make
    # old_succ[u] -> new_succ[u] for each u in nodes_to_swap
    changes = np.empty((len(nodes_to_swap), 2), dtype=np.int64)
    for i, u in enumerate(nodes_to_swap):
        changes[i, 0] = succ[u]  # old successor
        changes[i, 1] = parent2[0, u]  # new successor
        succ[u] = parent2[0, u]
    
    # Rebuild predecessor array
    for i in range(N):
        pred[succ[i]] = i


@njit(cache=True)
def _is_single_cycle(succ, N):
    """
    Check if the successor array forms a single cycle visiting all N nodes.
    """
    visited = np.zeros(N, dtype=np.bool_)
    current = 0
    count = 0
    
    for _ in range(N + 1):
        if visited[current]:
            break
        visited[current] = True
        count += 1
        current = succ[current]
    
    return count == N and current == 0


@njit(cache=True)
def _extract_subtours(succ, N):
    """
    Extract all subtours (cycles) from a successor array.
    Returns list of (start_node, length) for each subtour.
    """
    visited = np.zeros(N, dtype=np.bool_)
    subtours = List()
    
    for start in range(N):
        if visited[start]:
            continue
        
        # Trace this cycle with safeguard against infinite loops
        length = 0
        current = start
        for _ in range(N + 1):
            if visited[current]:
                break
            visited[current] = True
            length += 1
            next_node = succ[current]
            if next_node < 0 or next_node >= N:
                break
            current = next_node
        
        subtours.append(np.array([start, length], dtype=np.int64))
    
    return subtours


@njit(cache=True)
def _merge_two_subtours(succ, pred, s1_start, s2_start, distance_matrix, N):
    """
    Merge two subtours by finding the best 2-exchange between them.
    """
    if s1_start == s2_start:
        return succ, pred
    
    # Collect nodes in each subtour
    nodes1 = np.empty(N, dtype=np.int64)
    count1 = 0
    current = s1_start
    for _ in range(N):
        nodes1[count1] = current
        count1 += 1
        next_node = succ[current]
        if next_node < 0 or next_node >= N:
            break
        current = next_node
        if current == s1_start:
            break
    
    nodes2 = np.empty(N, dtype=np.int64)
    count2 = 0
    current = s2_start
    for _ in range(N):
        nodes2[count2] = current
        count2 += 1
        next_node = succ[current]
        if next_node < 0 or next_node >= N:
            break
        current = next_node
        if current == s2_start:
            break
    
    if count1 == 0 or count2 == 0:
        return succ, pred
    
    # Find best merge
    best_delta = np.inf
    best_u, best_v = -1, -1
    
    for i in range(count1):
        u = nodes1[i]
        u_next = succ[u]
        for j in range(count2):
            v = nodes2[j]
            v_next = succ[v]
            
            old_cost = distance_matrix[u, u_next] + distance_matrix[v, v_next]
            new_cost = distance_matrix[u, v_next] + distance_matrix[v, u_next]
            delta = new_cost - old_cost
            
            if delta < best_delta:
                best_delta = delta
                best_u, best_v = u, v
    
    if best_u >= 0:
        u_next = succ[best_u]
        v_next = succ[best_v]
        
        succ[best_u] = v_next
        pred[v_next] = best_u
        succ[best_v] = u_next
        pred[u_next] = best_v
    
    return succ, pred


@njit(cache=True) 
def _repair_to_single_tour(succ, pred, distance_matrix):
    """
    Repair a possibly fragmented tour into a single Hamiltonian cycle.
    """
    N = len(succ)
    
    for _ in range(N):
        subtours = _extract_subtours(succ, N)
        
        if len(subtours) <= 1:
            break
        
        s1 = subtours[0][0]
        s2 = subtours[1][0]
        succ, pred = _merge_two_subtours(succ, pred, s1, s2, distance_matrix, N)
    
    return succ, pred


@njit(cache=True)
def EAX(parent1, parent2, distance_matrix, num_trials=30, num_cycles_to_select=2):
    """
    Edge Assembly Crossover (EAX) operator for TSP tours in (2, N) edge-map representation.
    
    EAX works by:
    1. Finding A-B cycles (alternating edges between parent1 and parent2)
    2. Selecting a random subset of cycles
    3. Swapping all edges in selected cycles from parent1 to parent2
    4. Repairing the result if it fragments into multiple subtours
    5. Repeating multiple trials and returning the best offspring
    
    Args:
        parent1: First parent tour, shape (2, N)
        parent2: Second parent tour, shape (2, N)
        distance_matrix: NxN cost matrix
        num_trials: Number of offspring to generate (default 30)
        num_cycles_to_select: Max cycles to select (default 2)
    
    Returns:
        best_child: The best offspring tour, shape (2, N)
    """
    N = parent1.shape[1]
    
    # Find A-B cycles
    cycles = _find_AB_cycles(parent1, parent2)
    
    if len(cycles) == 0:
        return parent1.copy()
    
    best_child = None
    best_cost = np.inf
    
    for _trial in range(num_trials):
        # Select random subset of cycles
        num_cycs = len(cycles)
        k = min(num_cycles_to_select, num_cycs)
        if k > 1:
            k = np.random.randint(1, k + 1)
        
        indices = np.arange(num_cycs)
        np.random.shuffle(indices)
        selected_indices = indices[:k]
        
        # Start with copy of parent1
        succ = parent1[0].copy()
        pred = parent1[1].copy()
        
        # Apply swaps for selected cycles
        for idx in selected_indices:
            cycle = cycles[idx]
            _apply_swap(succ, pred, cycle, parent2)
        
        # Repair if needed
        if not _is_single_cycle(succ, N):
            succ, pred = _repair_to_single_tour(succ, pred, distance_matrix)
        
        # Build child tour
        child = np.empty((2, N), dtype=np.int64)
        child[0] = succ
        child[1] = pred
        
        # Calculate cost
        cost = 0.0
        for i in range(N):
            cost += distance_matrix[i, child[0, i]]
        
        if cost < best_cost:
            best_cost = cost
            best_child = child.copy()
    
    if best_child is None:
        return parent1.copy()
    
    return best_child


@njit(cache=True)
def EAX_single(parent1, parent2, distance_matrix, num_cycles_to_select=2):
    """
    Single-trial version of EAX for when you want to generate one offspring quickly.
    """
    return EAX(parent1, parent2, distance_matrix, num_trials=1, 
               num_cycles_to_select=num_cycles_to_select)