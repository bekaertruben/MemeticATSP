import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live

if TYPE_CHECKING:
    from tsp.solver import MemeticATSP

from tsp.representation import hamming_distance


class Reporter:
    """Reporter class for logging EA progress to CSV and rich console."""

    def __init__(self, output_dir: str = "output", filename: str = None):
        self.console = Console()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"run_{timestamp}.csv"
        
        self.csv_path = self.output_dir / filename
        self.start_time = None
        self.csv_file = None
        self.csv_writer = None

    def start(self, ea: "MemeticATSP" = None):
        """Start the reporter and print the log file location.
        
        Args:
            ea: Optional MemeticATSP instance to extract config metadata from.
        """
        self.start_time = time.time()
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write config metadata as comments
        if ea is not None:
            config = ea.config
            self.csv_file.write(f"# num_cities,{config.distance_matrix.shape[0]}\n")
            self.csv_file.write(f"# population_size,{config.population_size}\n")
            self.csv_file.write(f"# offspring_size,{config.offspring_size}\n")
            self.csv_file.write(f"# window_size,{config.window_size}\n")
            self.csv_file.write(f"# mutation_rate,{config.mutation_rate}\n")
            self.csv_file.write(f"# tournament_size,{config.tournament_size}\n")
            self.csv_file.write(f"# init_temp,{config.init_temp}\n")
            self.csv_file.write(f"# search_iterations,{(config.search_iters_3opt, config.search_iters_oropt, config.search_iters_2opt)}\n")
        
        self.csv_writer.writerow([
            "generation",
            "time_elapsed",
            "best_fitness",
            "mean_fitness",
            "avg_hamming_distance"
        ])
        
        self.console.print(f"[bold green]Logging to:[/bold green] {self.csv_path.absolute()}")

    def stop(self):
        """Close the CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def _compute_avg_hamming_distance(self, population: np.ndarray) -> float:
        """
        Compute average pairwise Hamming distance in the population.
        Uses sampling for efficiency with large populations.
        """
        pop_size = population.shape[0]
        
        # Sample pairs for efficiency if population is large
        if pop_size > 50:
            num_samples = min(500, pop_size * (pop_size - 1) // 2)
            total_dist = 0
            for _ in range(num_samples):
                i, j = np.random.choice(pop_size, size=2, replace=False)
                total_dist += hamming_distance(population[i], population[j])
            return total_dist / num_samples
        else:
            # Compute all pairwise distances for small populations
            total_dist = 0
            count = 0
            for i in range(pop_size):
                for j in range(i + 1, pop_size):
                    total_dist += hamming_distance(population[i], population[j])
                    count += 1
            return total_dist / count if count > 0 else 0.0

    def log(self, ea: "MemeticATSP"):
        """Log the current state of the EA."""
        elapsed = time.time() - self.start_time
        best_fitness = ea.best_fitness
        mean_fitness = ea.mean_fitness
        avg_hamming = self._compute_avg_hamming_distance(ea.population)
        
        # Write to CSV
        self.csv_writer.writerow([
            ea.generation,
            f"{elapsed:.2f}",
            f"{best_fitness:.2f}",
            f"{mean_fitness:.2f}",
            f"{avg_hamming:.2f}"
        ])
        self.csv_file.flush()
        
        # Print to console using rich
        self.console.print(
            f"[cyan]Gen {ea.generation:>4}[/cyan] | "
            f"[yellow]Time: {elapsed:>7.2f}s[/yellow] | "
            f"[green]Best: {best_fitness:>10.2f}[/green] | "
            f"[blue]Mean: {mean_fitness:>10.2f}[/blue] | "
            f"[magenta]Diversity: {avg_hamming:>6.2f}[/magenta]"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
