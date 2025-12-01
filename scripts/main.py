import numpy as np
import numba
import resource
from tsp.solver import MemeticATSP
from tsp.representation import is_valid_tour, tour_cost

# Set resource limits:
# numba.set_num_threads(2)
# soft, hard = 1_000_000_000, 1_000_000_000 # 1 GB
# resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

D = np.loadtxt("tours/tour500.csv", delimiter=",", dtype=np.float64)
ea = MemeticATSP(D)
ea.run(100)

assert is_valid_tour(ea.best_tour), "Best tour is invalid!"
print(f"Best fitness: {tour_cost(ea.best_tour, D)}")