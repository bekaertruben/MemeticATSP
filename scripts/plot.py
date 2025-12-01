# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: tsp
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import os
import glob

# %%
# find most recent file in output/ directory
list_of_files = glob.glob('../output/run_*.csv')
logfile = max(list_of_files, key=os.path.getctime)
# logfile = "./output/run_20251130_004600.csv"

# print header information
with open(logfile, 'r') as f:
    for _ in range(8):
        print(f.readline().strip())

data = pd.read_csv(logfile, header=8)
data.tail()

# %%
plt.plot(data["generation"], data["best_fitness"], label="Best Fitness")
plt.plot(data["generation"], data["mean_fitness"], label="Mean Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness over Generations")
plt.legend()

# %%
plt.figure()
plt.plot(data["generation"], data["avg_hamming_distance"], label="Avg Hamming Distance")
plt.xlabel("Generation")
plt.ylabel("Avg Hamming Distance")
plt.title("Diversity Over Generations")
plt.legend()
plt.show()

# %%
plt.figure()
x = data["generation"]
offspring = data["time_offspring"]
mutation = data["time_mutation"]
search = data["time_search"]
elimination = data["time_elimination"]

# Stacked bar plot with flush bars
bar_width = 1.0  # Make bars fully flush
plt.yscale('log')
plt.bar(x, offspring, width=bar_width, label="Offspring Generation")
plt.bar(x, mutation, width=bar_width, bottom=offspring, label="Mutation")
plt.bar(x, search, width=bar_width, bottom=offspring+mutation, label="Local Search")
plt.bar(x, elimination, width=bar_width, bottom=offspring+mutation+search, label="Elimination")
plt.xlabel("Generation")
plt.ylabel("Time (s)")
plt.title("Time Breakdown per Generation")
plt.legend()
plt.show()
