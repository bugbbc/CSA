#!/usr/bin/env python3
"""
Batch run all experiments.

Usage:
    python scripts/run_all_experiments.py [--smoke-test]
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EXPERIMENTS = [
    "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7",
]


def main():
    parser = argparse.ArgumentParser(description="Run all CSA experiments")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run quick smoke tests")
    parser.add_argument("--experiments", type=str, nargs="+",
                       default=EXPERIMENTS,
                       help=f"Experiments to run (default: all: {EXPERIMENTS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 3407])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    run_script = os.path.join(script_dir, "run_experiment.py")

    for exp in args.experiments:
        cmd = [
            sys.executable, run_script, exp,
            "--device", args.device,
            "--seeds"] + [str(s) for s in args.seeds] + [
            "--output-dir", "results",
        ]
        if args.smoke_test:
            cmd.append("--smoke-test")

        print(f"\n{'#' * 70}")
        print(f"# Running experiment: {exp}")
        print(f"{'#' * 70}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: Experiment {exp} failed with code {result.returncode}")
            if not args.smoke_test:
                sys.exit(1)

    print("\n" + "=" * 70)
    print("All experiments complete!")
    print("=" * 70)

    # Generate tables
    print("\nGenerating summary tables...")

    import pandas as pd
    results_dir = "results"
    if os.path.exists(results_dir):
        summary = []
        for exp in args.experiments:
            csv_path = os.path.join(results_dir, f"table_{exp}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                summary.append(f"\n--- {exp.upper()} ---")
                summary.append(df.to_string(index=False))
        print("\n".join(summary))


if __name__ == "__main__":
    main()
