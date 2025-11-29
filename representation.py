import numpy as np
from numba import njit

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