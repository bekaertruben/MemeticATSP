"""
Hyperparameter tuning for MemeticATSP using Ray Tune.
"""

import os
import numpy as np
import ray
from ray import tune
from ray.tune.stopper import TrialPlateauStopper
from tsp.solver import MemeticATSP

# Get the absolute path to the data file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "tours", "tour500.csv")


def train_memetic(config):
    """Training function for Ray Tune."""
    # Load distance matrix
    D = np.loadtxt(DATA_PATH, delimiter=",")

    # Create solver with hyperparameters from config
    ea = MemeticATSP(
        distance_matrix=D,
        population_size=100, # config["lambda"]
        offspring_size=100, # config["mu"]
        init_temp=config["T"],
        tournament_size=config["k"],
        window_size=config["window"],
        mutation_rates=(config["db"], config["rev"]),
        search_iterations=(
            config["3opt"],
            config["oropt"],
            config["2opt"],
        ),
    )

    ea.initialize()
    i = 0
    while True:
        i += 1
        ea.step()
        # Report metrics to Ray Tune
        tune.report(metrics={
            "best_fitness": ea.best_fitness,
            "mean_fitness": ea.mean_fitness,
            "iteration": i,
        })


def main():
    # Define the search space
    search_space = {
        # "lambda": tune.randint(50, 301),
        # "mu": tune.randint(50, 301),
        "T": tune.loguniform(0.01, 1.0),
        "db": tune.uniform(0.05, 0.5),
        "rev": tune.uniform(0.05, 0.5),
        "k": tune.randint(2, 11),
        "window": tune.randint(5, 100),
        "3opt": tune.randint(0, 11),
        "oropt": tune.randint(0, 11),
        "2opt": tune.randint(0, 11),
    }

    # Define the Stopper
    stopper = TrialPlateauStopper(
        metric="best_fitness",
        std=0.0001,
        num_results=15,
        grace_period=20,
        # mode="min"
    )

    tuner = tune.Tuner(
        train_memetic,
        param_space=search_space,
        run_config=tune.RunConfig(
            name="memetic_tsp_tuning",
            stop=stopper,
            verbose=1,
        ),
        tune_config=tune.TuneConfig(
            metric="best_fitness",
            mode="min",
            num_samples=50,
            max_concurrent_trials=1,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="best_fitness", mode="min")

    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS FOUND:")
    print("=" * 60)
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")
    print(f"\nBest fitness achieved: {best_result.metrics['best_fitness']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    ray.init(runtime_env={"excludes": ["*"]})  # Don't package the working directory
    main()
