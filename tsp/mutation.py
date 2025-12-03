import numpy as np
from numba import njit

"""
MUTATION OPERATORS:

This module implements mutation operators for TSP tours.
These operators modify tours in-place and return the cost delta.
"""


@njit(cache=True)
def displacement(tour, D):
    """
    Perform a random displacement mutation on a tour in (2, N) edge-map representation.
    Removes a node from its current position and inserts it after another node.
    
    Returns the cost delta (negative means improvement).
    """
    N = tour.shape[1]
    if N < 3:
        return 0.0

    node_to_move = np.random.randint(N)
    target = np.random.randint(N)
    # Ensure target is not the node itself or its current successor
    while target == node_to_move or target == tour[0, node_to_move]:
        target = np.random.randint(N)

    # Get predecessor and successor of node_to_move
    pred = tour[1, node_to_move]
    succ = tour[0, node_to_move]
    
    # Get successor of target
    target_succ = tour[0, target]

    # Calculate cost delta
    # Edges removed: pred -> node_to_move, node_to_move -> succ, target -> target_succ
    # Edges added: pred -> succ, target -> node_to_move, node_to_move -> target_succ
    old_cost = D[pred, node_to_move] + D[node_to_move, succ] + D[target, target_succ]
    new_cost = D[pred, succ] + D[target, node_to_move] + D[node_to_move, target_succ]
    delta = new_cost - old_cost

    # Remove node_to_move: connect pred -> succ
    tour[0, pred] = succ
    tour[1, succ] = pred

    # Insert node_to_move after target: target -> node_to_move -> target_succ
    tour[0, target] = node_to_move
    tour[1, node_to_move] = target
    tour[0, node_to_move] = target_succ
    tour[1, target_succ] = node_to_move

    return delta


@njit(cache=True)
def reverse(tour, D):
    """
    Perform a random 2-opt move on a tour in (2, N) edge-map representation.
    Picks two random nodes and reverses the segment between them.
    
    Returns the cost delta (negative means improvement).
    """
    N = tour.shape[1]
    if N < 4:
        return 0.0

    # Pick two distinct random nodes that are not adjacent
    a = np.random.randint(N)
    c = np.random.randint(N)
    while c == a or c == tour[0, a] or a == tour[0, c]:
        c = np.random.randint(N)

    b = tour[0, a]  # a -> b
    d = tour[0, c]  # c -> d

    # Collect nodes in the segment b -> ... -> c
    segment = []
    curr = b
    while curr != c:
        segment.append(curr)
        curr = tour[0, curr]
    segment.append(c)
    
    # Calculate cost delta (for ATSP, all reversed edges change cost)
    # Edges removed: a -> b, c -> d, and all internal edges in segment
    # Edges added: a -> c, b -> d, and all internal edges reversed
    old_cost = D[a, b] + D[c, d]
    new_cost = D[a, c] + D[b, d]
    
    # Add cost change from reversing internal edges
    for i in range(len(segment) - 1):
        old_cost += D[segment[i], segment[i + 1]]
        new_cost += D[segment[i + 1], segment[i]]
    
    delta = new_cost - old_cost
    
    # Reverse the segment: now goes c -> ... -> b
    # New connections: a -> c, c -> prev(c), ..., b -> d
    
    # First, set up the new forward edges within the reversed segment
    for i in range(len(segment) - 1):
        # segment[i] now points to segment[i-1] (reversed direction)
        # But we need to be careful: segment[0] = b should point to d
        # segment[-1] = c should be pointed to by a
        tour[0, segment[i + 1]] = segment[i]
        tour[1, segment[i]] = segment[i + 1]
    
    # Connect the endpoints
    tour[0, a] = c
    tour[1, c] = a
    tour[0, b] = d
    tour[1, d] = b

    return delta


@njit(cache=True)
def double_bridge(tour, D):
    """
    Perform a double bridge mutation (4-opt kick) in-place.
    
    Original edges: a -> a', b -> b', c -> c', d -> d'
    New edges:      a -> c', b -> d', c -> a', d -> b'
    
    This reconnects segments as A, D, C, B (where original was A, B, C, D).
    Only 4 edges are modified - O(1) operation.
    
    Returns the cost delta (negative means improvement).
    """
    N = tour.shape[1]
    if N < 8:
        return 0.0

    # Pick 4 distinct random nodes n1, n2, n3, n4
    n1 = np.random.randint(N)
    n2 = np.random.randint(N)
    while n2 == n1:
        n2 = np.random.randint(N)
    n3 = np.random.randint(N)
    while n3 in (n1, n2):
        n3 = np.random.randint(N)
    n4 = np.random.randint(N)
    while n4 in (n1, n2, n3):
        n4 = np.random.randint(N)

    # Determine order (labelled a, b, c, d)
    abcd = np.empty(4, dtype=np.int64)
    abcd[0] = n1 
    order_idx = 1
    curr = tour[0, n1]
    while order_idx < 4:
        if curr in (n2, n3, n4):
            abcd[order_idx] = curr
            order_idx += 1
        curr = tour[0, curr]

    # Get successors before modification
    a_succ = tour[0, abcd[0]]
    b_succ = tour[0, abcd[1]]
    c_succ = tour[0, abcd[2]]
    d_succ = tour[0, abcd[3]]

    # Calculate cost delta
    # Edges removed: a->a', b->b', c->c', d->d'
    # Edges added: a->c', b->d', c->a', d->b'
    old_cost = D[abcd[0], a_succ] + D[abcd[1], b_succ] + D[abcd[2], c_succ] + D[abcd[3], d_succ]
    new_cost = D[abcd[0], c_succ] + D[abcd[1], d_succ] + D[abcd[2], a_succ] + D[abcd[3], b_succ]
    delta = new_cost - old_cost

    # Reconnect using array indexing: a->c', c->a', b->d', d->b'
    targets = np.array(
        [c_succ, d_succ, a_succ, b_succ],
        dtype=np.int64
    )
    
    tour[0, abcd] = targets
    tour[1, targets] = abcd

    return delta


@njit
def random_k_segment_perturbation(tour, k):
    """
    Perform a random k-segment perturbation (non-reversing k-opt).
    Breaks the tour into k segments and permutes them.
    
    Args:
        tour: The tour in (2, N) representation.
        k: Number of segments (cuts).
    """
    N = tour.shape[1]
    if N < k * 2:
        return # Too small to cut into k segments

    # 1. Select k random cut points (edges to break)
    # We need k distinct edges.
    # An edge is defined by its start node. (start -> next)
    cuts = np.zeros(k, dtype=np.int64)
    
    # Simple random selection without replacement
    # For small k relative to N, rejection sampling is fine.
    count = 0
    while count < k:
        candidate = np.random.randint(N)
        # Check if already selected or adjacent to selected (to avoid empty segments)
        # Actually, empty segments are just single nodes, which is fine?
        # But for "segment" permutation, usually we want at least 1 node.
        # Let's just ensure distinct edges.
        is_new = True
        for i in range(count):
            if cuts[i] == candidate:
                is_new = False
                break
        if is_new:
            cuts[count] = candidate
            count += 1
            
    # Sort cuts to define segments in order of appearance in the tour
    # We cannot just sort by index. We must traverse the tour.
    # To do this efficiently:
    # 1. Mark cuts in a boolean array
    # 2. Traverse tour and collect cuts in order
    
    is_cut = np.zeros(N, dtype=np.bool_)
    for i in range(k):
        is_cut[cuts[i]] = True
        
    sorted_cuts = np.zeros(k, dtype=np.int64)
    current = 0 # Start traversal from 0 (arbitrary start of cycle)
    found = 0
    
    # We need to find the first cut to align the cycle?
    # Actually, any cyclic order is fine.
    # But we need to traverse the whole tour to find all k cuts in order.
    
    # Optimization: if we just traverse from 0, we find them in order 0->...
    
    idx = 0
    curr = 0
    visited_count = 0
    while visited_count < N and idx < k:
        if is_cut[curr]:
            sorted_cuts[idx] = curr
            idx += 1
        curr = tour[0, curr]
        visited_count += 1
        
    cuts = sorted_cuts
    
    # Segments are defined between cuts.
    # If cuts are c1, c2, ..., ck (in tour order)
    # Segment 1: next[c1] ... c2
    # ...
    
    starts = np.zeros(k, dtype=np.int64)
    ends = np.zeros(k, dtype=np.int64)
    
    for i in range(k):
        starts[i] = tour[0, cuts[i]]
        ends[i] = cuts[(i + 1) % k]
        
    # 2. Permute the segments
    # We need a random permutation of 0..k-1
    # But we must ensure it's not the identity (0, 1, ..., k-1)
    # and preferably not a simple rotation (which is just the same tour shifted).
    # Since we are reconnecting, any reordering that changes adjacency is a valid move.
    
    perm = np.random.permutation(k)
    
    # Check if identity (unlikely for k >= 4)
    is_identity = True
    for i in range(k):
        if perm[i] != i:
            is_identity = False
            break
            
    if is_identity:
        # Swap first two
        perm[0], perm[1] = perm[1], perm[0]
        
    # 3. Reconnect segments
    # New order: Segment perm[0] -> Segment perm[1] -> ... -> Segment perm[k-1] -> Segment perm[0]
    # Connection: End of Segment perm[i] -> Start of Segment perm[i+1]
    
    for i in range(k):
        u = ends[perm[i]]
        v = starts[perm[(i + 1) % k]]
        
        # Create edge u -> v
        tour[0, u] = v
        tour[1, v] = u