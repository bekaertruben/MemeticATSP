import numpy as np
from tsp.solver import MemeticATSP
from tsp.representation import is_valid_tour, to_city_order, tour_cost

D = np.loadtxt("tours/tour750.csv", delimiter=",")

ea = MemeticATSP(D)
ea.run(200)
assert is_valid_tour(ea.best_tour), "Best tour is invalid!"
# print(to_city_order(ea.best_tour))
print(f"Best fitness: {tour_cost(ea.best_tour, D)}")