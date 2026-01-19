#!/usr/bin/env python3
import argparse

from config import ExperimentConfig
from experiment_runner import run_experiment_grid, create_treatment_grid
from plotting import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Run supply chain convergence experiments"
    )
    
    parser.add_argument("--n_seeds", type=int, default=50,
                        help="Number of random seeds")
    parser.add_argument("--rounds", type=int, default=365,
                        help="Training rounds")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup rounds to exclude from metrics")
    
    args = parser.parse_args()
    
    # Build config
    base_config = ExperimentConfig(
        rounds=args.rounds,
        warmup=args.warmup,
    )
    
    # Create treatments (simplified: 4 agent pairs, 1 grid, random init only)
    treatments = create_treatment_grid()
    seeds = list(range(args.n_seeds))
    output_dir = "results"
    
    print("=" * 60)
    print("Supply Chain Convergence Experiment")
    print("=" * 60)
    print(f"Training rounds: {args.rounds}, Warmup: {args.warmup}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Treatments: {len(treatments)}")
    print("All metrics and plots exclude the first warmup rounds.")
    print("=" * 60)
    
    # Run experiment
    results = run_experiment_grid(
        treatments=treatments,
        base_config=base_config,
        seeds=seeds,
        output_dir=output_dir,
        verbose=True,
    )
    
    # Generate plots
    generate_all_plots(results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"DONE. Results in: {output_dir}/")
    print("  - summary.csv: treatment-level aggregates")
    print("  - runs.csv: per-seed metrics")
    print("  - figures/: all plots")
    print("=" * 60)


if __name__ == "__main__":
    main()
