import numpy as np
import numba
from tsp.solver import MemeticATSP
from tsp.representation import is_valid_tour, tour_cost

# numba.set_num_threads(2)

D = np.loadtxt("tours/tour500.csv", delimiter=",")
ea = MemeticATSP(D)
ea.run(200)

assert is_valid_tour(ea.best_tour), "Best tour is invalid!"
print(f"Best fitness: {tour_cost(ea.best_tour, D)}")