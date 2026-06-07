"""Experiment runner and orchestration."""

from __future__ import annotations
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils.seed import set_seed


class ExperimentRunner:
    """Orchestrates experiment execution across seeds and methods."""

    def __init__(self, config, output_dir: str = "results"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        experiment_fn: Callable,
        methods: List[str],
        seeds: List[int],
        **kwargs,
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Run an experiment across multiple methods and seeds.

        Args:
            experiment_fn: function(method, seed, **kwargs) -> Dict[str, float]
            methods: list of method names
            seeds: list of seeds to run
        Returns:
            {method: {metric: (mean, std)}}
        """
        all_results = {}

        for method in methods:
            method_results = []
            for seed in seeds:
                set_seed(seed)
                print(f"\n{'=' * 60}")
                print(f"Running {method} with seed={seed}")
                print(f"{'=' * 60}")

                result = experiment_fn(method=method, seed=seed, **kwargs)
                method_results.append(result)

                print(f"  Result: {result}")

            # Aggregate across seeds
            aggregated = {}
            for key in method_results[0]:
                values = np.array([r[key] for r in method_results])
                aggregated[key] = (float(np.mean(values)), float(np.std(values)))

            all_results[method] = aggregated

            # Print summary
            print(f"\n--- {method} Summary ---")
            for metric, (mean, std) in aggregated.items():
                print(f"  {metric}: {mean:.4f} ± {std:.4f}")

        # Save results
        self._save_results(all_results)

        return all_results

    def _save_results(self, results: dict):
        """Save results to JSON."""
        serializable = {}
        for method, metrics in results.items():
            serializable[method] = {
                k: {"mean": float(v[0]), "std": float(v[1])}
                for k, v in metrics.items()
            }
        path = os.path.join(self.output_dir, "results.json")
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {path}")

    def save_table(self, results: dict, table_name: str):
        """Save results as CSV table."""
        import pandas as pd
        rows = []
        for method, metrics in results.items():
            row = {"Method": method}
            for metric, (mean, std) in metrics.items():
                row[f"{metric}"] = f"{mean:.4f} ± {std:.4f}"
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, f"{table_name}.csv")
        df.to_csv(path, index=False)
        print(f"Table saved to {path}")
        return df


def measure_time(fn: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Measure execution time of a function."""
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed, result
