import numpy as np
from numba import njit

"""
MUTATION OPERATORS:

This module implements mutation operators for TSP tours.
These operators modify tours in-place.
"""


@njit(cache=True)
def displacement(tour):
    """
    Perform a random displacement mutation on a tour in (2, N) edge-map representation.
    Removes a node from its current position and inserts it after another node.
    """
    N = tour.shape[1]
    if N < 3:
        return

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

    # Remove node_to_move: connect pred -> succ
    tour[0, pred] = succ
    tour[1, succ] = pred

    # Insert node_to_move after target: target -> node_to_move -> target_succ
    tour[0, target] = node_to_move
    tour[1, node_to_move] = target
    tour[0, node_to_move] = target_succ
    tour[1, target_succ] = node_to_move


@njit(cache=True)
def double_bridge(tour):
    """
    Perform a double bridge mutation (4-opt kick) in-place.
    
    Original edges: a -> a', b -> b', c -> c', d -> d'
    New edges:      a -> c', b -> d', c -> a', d -> b'
    
    This reconnects segments as A, D, C, B (where original was A, B, C, D).
    Only 4 edges are modified - O(1) operation.
    """
    N = tour.shape[1]
    if N < 8:
        return

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

    # Reconnect using array indexing: a->c', c->a', b->d', d->b'
    targets = np.array(
        [tour[0, abcd[2]], tour[0, abcd[3]], tour[0, abcd[0]], tour[0, abcd[1]]],
        dtype=np.int64
    )
    
    tour[0, abcd] = targets
    tour[1, targets] = abcd