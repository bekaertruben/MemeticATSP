import numpy as np
from numba import njit
from tsp.representation import hamming_distance

"""
GENETIC ALGORITHM

This module implements various selection and elimination strategies for genetic algorithms.
"""


@njit(cache=True)
def tournament_selection(fitness, k=4):
    """
    Select an individual from the population using tournament selection.
    """

    while True:
        best = -1
        best_fitness = np.inf
        for _ in range(k):
            candidate = np.random.randint(0, len(fitness))
            if fitness[candidate] < best_fitness:
                best_fitness = fitness[candidate]
                best = candidate

        yield best


@njit(cache=True)
def fitness_proportionate_selection(fitness):
    """
    Select individuals based on fitness proportionate selection (roulette wheel).
    """
    fitness_sum = np.sum(fitness)
    probabilities = fitness_sum - fitness # We want to minimize the fitness values
    probabilities /= np.sum(probabilities)

    cumulative_probabilities = np.cumsum(probabilities)

    while True:
        r = np.random.rand()
        for j in range(len(cumulative_probabilities)):
            if r <= cumulative_probabilities[j]:
                yield j
                break


@njit(cache=True)
def rank_selection(fitness, s=2):
    """
    Select individuals based on rank selection.
    """
    N = len(fitness)
    ranks = N - np.argsort(np.argsort(fitness))
    
    probs = (2 - s) / N + (2 * ranks * (s - 1)) / (N * (N - 1))
    cumulative_probabilities = np.cumsum(probs)

    while True:
        r = np.random.rand()
        for j in range(len(cumulative_probabilities)):
            if r <= cumulative_probabilities[j]:
                yield j
                break


@njit(cache=True)
def fitness_sharing(fitness, population, sigma_share=5.0, alpha=1.0):
    """
    Select individuals using fitness sharing.
    """
    pop_size = population.shape[0]
    niche_counts = np.ones(pop_size, dtype=np.float64)
    
    for i in range(pop_size):
        for j in range(i+1, pop_size):
            if i != j:
                dist = hamming_distance(population[i], population[j])
                if dist < sigma_share:
                    count = 1 - (dist / sigma_share) ** alpha
                    niche_counts[i] += count
                    niche_counts[j] += count

    shared_fitness = fitness * niche_counts
    return shared_fitness


@njit(cache=True)
def restricted_tournament_replacement(population, population_fitness, offspring, offspring_fitness, window_size):
    """
    Eliminate individuals to maintain population size, using Restricted Tournament Replacement (RTR).
    Returns both the new population and updated fitness values.
    """
    pop_size = population.shape[0]

    new_population = population.copy()
    new_fitness = population_fitness.copy()

    for i in range(offspring.shape[0]):
        offspring_tour = offspring[i]
        offspring_fit = offspring_fitness[i]

        # Select random window
        start_idx = np.random.randint(0, pop_size)
        indices = [(start_idx + j) % pop_size for j in range(window_size)]

        # Find the most similar individual in the window
        best_idx = -1
        best_distance = np.inf

        for idx in indices:
            dist = hamming_distance(offspring_tour, new_population[idx])
            if dist < best_distance:
                best_distance = dist
                best_idx = idx

        # Replace if offspring is better
        if offspring_fit < new_fitness[best_idx]:
            new_population[best_idx] = offspring_tour
            new_fitness[best_idx] = offspring_fit

    return new_population, new_fitness