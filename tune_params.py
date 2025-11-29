"""
Hyperparameter tuning for MemeticATSP using Ray Tune.
"""

import os
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from tsp.solver import MemeticATSP

# Get the absolute path to the data file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "tours", "tour250.csv")


def train_memetic(config):
    """Training function for Ray Tune."""
    # Load distance matrix
    D = np.loadtxt(DATA_PATH, delimiter=",")
    
    # Create solver with hyperparameters from config
    ea = MemeticATSP(
        distance_matrix=D,
        population_size=config["population_size"],
        offspring_size=config.get("offspring_size"),
        window_size=config.get("window_size"),
        mutation_rate=config["mutation_rate"],
        tournament_size=config["tournament_size"],
        init_temp=config["init_temp"],
        search_iterations=config["search_iterations"],
    )
    
    ea.initialize()
    
    # Run for 50 iterations, reporting progress
    for i in range(50):
        ea.step()
        # Report metrics to Ray Tune
        tune.report(metrics={
            "best_fitness": ea.best_fitness,
            "mean_fitness": ea.mean_fitness,
            "iteration": i + 1,
        })


def main():
    # Define the search space
    search_space = {
        "population_size": tune.choice([50, 100, 200, 300, 500]),
        "offspring_size": tune.choice([None, 25, 50, 100, 200]),
        "window_size": tune.choice([None, 2, 5, 10, 20]),
        "mutation_rate": tune.uniform(0.05, 0.5),
        "tournament_size": tune.choice([2, 3, 5, 7]),
        "init_temp": tune.loguniform(0.01, 1.0),
        "search_iterations": tune.choice([1, 3, 5, 10]),
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
        max_t=50,  # Max iterations
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
            num_samples=50,  # Number of trials to run
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
    
    # Optionally run a final verification with the best params
    print("\nRunning verification with best parameters...")
    D = np.loadtxt(DATA_PATH, delimiter=",")
    ea = MemeticATSP(
        distance_matrix=D,
        **best_result.config
    )
    ea.run(50, verbose=True)
    print(f"\nFinal best fitness: {ea.best_fitness:.2f}")


if __name__ == "__main__":
    main()
