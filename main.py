import numpy as np
from tsp.solver import MemeticATSP
from tsp.representation import is_valid_tour, to_city_order

D = np.loadtxt("tours/tour1000.csv", delimiter=",")

ea = MemeticATSP(D)
ea.run(500)
assert is_valid_tour(ea.best_tour), "Best tour is invalid!"
print(to_city_order(ea.best_tour))