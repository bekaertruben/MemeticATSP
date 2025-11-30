"""
Hyperparameter tuning for MemeticATSP using Ray Tune.
"""

import os
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from tsp.solver import MemeticATSP

# Get the absolute path to the data file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "tours", "tour500.csv")


def train_memetic(config):
    """Training function for Ray Tune."""
    # Load distance matrix
    D = np.loadtxt(DATA_PATH, delimiter=",")
    
    # Create solver with hyperparameters from config
    ea = MemeticATSP(
        distance_matrix=D,
        population_size=config["population_size"],
        offspring_size=config.get("offspring_size"),
        sharing_radius=config["sharing_radius"],
        sharing_alpha=config["sharing_alpha"],
        # Convert scalar total mutation probability to per-operator absolute
        # probabilities (even split across both operators).
        mutation_probs=(config["mutation_prob_total"]/2.0, config["mutation_prob_total"]/2.0),
        tournament_size=config["tournament_size"],
        init_temp=config["init_temp"],
        search_iterations=(
            config["search_iters_3opt"],
            config["search_iters_oropt"],
            config["search_iters_2opt"],
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
        "population_size": tune.randint(50, 301),
        "offspring_size": tune.randint(50, 301),
        "sharing_radius": tune.uniform(0.05, 0.5),
        "sharing_alpha": tune.uniform(0.5, 2.0),
        # Total mutation probability to be split across available operators.
        "mutation_prob_total": tune.uniform(0.05, 0.5),
        "tournament_size": tune.randint(2, 11),
        "init_temp": tune.loguniform(0.01, 1.0),
        "search_iters_3opt": tune.randint(0, 6),
        "search_iters_oropt": tune.randint(0, 11),
        "search_iters_2opt": tune.randint(0, 21),
    }
    
    # Use Optuna as the search algorithm
    optuna_search = OptunaSearch(
        metric="best_fitness",
        mode="min",
    )
    
    # ASHA scheduler for early stopping of bad trials
    scheduler = ASHAScheduler(
        metric="best_fitness",
        mode="min",
        max_t=1000,  # Max iterations
        grace_period=10,  # Min iterations before pruning
        reduction_factor=2,
    )
    
    # Run the tuning
    tuner = tune.Tuner(
        train_memetic,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=500,  # Number of trials to run
            max_concurrent_trials=10,  # Limit simultaneous trials
            time_budget_s=5 * 60,  # Quit each trial after 5 minutes
        ),
        run_config=tune.RunConfig(
            name="memetic_tsp_tuning",
            verbose=1,
        ),
    )
    
    results = tuner.fit()
    
    # Get the best result
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
