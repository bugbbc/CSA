#!/usr/bin/env python3
"""
Entry point: run a single experiment by name.

Usage:
    python scripts/run_experiment.py exp1 --device cuda --seed 42
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from csa.utils.seed import set_seed
from csa.experiments.runner import ExperimentRunner
from csa.experiments.exp1_longbench import run_exp1_longbench, METHODS as METHODS_EXP1
from csa.experiments.exp2_infinitebench import run_exp2_infinitebench, METHODS as METHODS_EXP2
from csa.experiments.exp3_needle import run_exp3_needle, METHODS as METHODS_EXP3
from csa.experiments.exp4_causal import run_exp4_causal, METHODS as METHODS_EXP4
from csa.experiments.exp5_proxy import run_exp5_proxy, METHODS as METHODS_EXP5
from csa.experiments.exp6_ablation import run_exp6_ablation
from csa.experiments.exp7_efficiency import run_exp7_efficiency, METHODS as METHODS_EXP7

EXPERIMENTS = {
    "exp1": {
        "fn": run_exp1_longbench,
        "methods": METHODS_EXP1,
        "description": "LongBench Main Benchmarks",
    },
    "exp2": {
        "fn": run_exp2_infinitebench,
        "methods": METHODS_EXP2,
        "description": "InfiniteBench Very Long Context",
    },
    "exp3": {
        "fn": run_exp3_needle,
        "methods": METHODS_EXP3,
        "description": "Needle-in-a-Haystack",
    },
    "exp4": {
        "fn": run_exp4_causal,
        "methods": METHODS_EXP4,
        "description": "Causal Robustness",
    },
    "exp5": {
        "fn": run_exp5_proxy,
        "methods": METHODS_EXP5,
        "description": "Proxy Validation",
    },
    "exp6": {
        "fn": run_exp6_ablation,
        "methods": ["csa"],
        "description": "Ablation Study",
    },
    "exp7": {
        "fn": run_exp7_efficiency,
        "methods": METHODS_EXP7,
        "description": "Efficiency Analysis",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run CSA experiments")
    parser.add_argument("experiment", type=str, choices=list(EXPERIMENTS.keys()),
                       help="Experiment to run")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                       help="Methods to evaluate (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                       help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run a quick smoke test with minimal config")

    args = parser.parse_args()

    exp_info = EXPERIMENTS[args.experiment]
    exp_fn = exp_info["fn"]
    methods = args.methods or exp_info["methods"]
    seeds = args.seeds

    output_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Experiment: {args.experiment} - {exp_info['description']}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Device: {args.device}")
    print(f"{'=' * 70}\n")

    config = type('Config', (), {'attention': type('AttentionConfig', (), {
        'type': 'csa',
    })()})()

    runner = ExperimentRunner(config, output_dir=output_dir)

    kwargs = {
        "device": args.device,
    }
    if args.smoke_test:
        kwargs.update({
            "batch_size": 2,
            "max_batches": 2,
            "num_examples": 10,
            "seq_length": 64,
            "num_train": 10,
            "num_test": 5,
            "num_epochs": 1,
            "window": 32,
            "k": 8,
        })

    results = runner.run(
        experiment_fn=exp_fn,
        methods=methods,
        seeds=seeds,
        **kwargs,
    )

    runner.save_table(results, f"table_{args.experiment}")

    print(f"\nExperiment {args.experiment} complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
