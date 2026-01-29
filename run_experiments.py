#!/usr/bin/env python3
import argparse
import os
from simulation.config import ExperimentConfig
from experiment_runner import run_experiment_grid, TreatmentConfig
from analysis.plotting import generate_all_plots

SCENARIOS = {
    # Baseline: each agent minimizes its own local cost (default behavior).
    "baseline": {
        "init_mode": "random",
        "prior_knowledge": "none",
    },

    # Prior knowledge: agents start with a "demand_known" prior over actions.
    "demand_known_prior": {
        "init_mode": "random",
        "prior_knowledge": "demand_known",
    },

    # Benchmark init: seed beliefs towards the centralized benchmark optimum (s1*, s2*).
    "benchmark_init": {
        "init_mode": "benchmark",
        "prior_knowledge": "none",
    },
}

AGENTS = ["greedy", "ucb", "thompson", "exp3", "etc"]
DEFAULT_PAIRS = [("greedy", "greedy"), ("ucb", "ucb"), ("greedy", "ucb"), ("ucb", "greedy")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--s_lower", type=int, default=0)
    parser.add_argument("--s_upper", type=int, required=True)
    parser.add_argument("--s_step", type=int, default=1)
    parser.add_argument("--full_grid", action="store_true")
    parser.add_argument("--n_seeds", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    overrides = SCENARIOS[args.scenario]
    pairs = [(r, s) for r in AGENTS for s in AGENTS] if args.full_grid else DEFAULT_PAIRS
    grid_label = "full" if args.full_grid else "subset"
    output = os.path.join(args.output_dir, args.scenario, f"s{args.s_lower}-{args.s_upper}-{args.s_step}", grid_label)
    
    treatments = [
        TreatmentConfig(
            agent_retailer=r, 
            agent_supplier=s, 
            s_lower=args.s_lower,
            s_upper=args.s_upper,
            s_step=args.s_step,
            **overrides
        ) 
        for r, s in pairs
    ]
    base = ExperimentConfig(rounds=args.rounds, warmup=args.warmup)

    print(f"{'='*60}\nScenario: {args.scenario}, Grid: {args.s_lower}-{args.s_upper} (step={args.s_step}), {grid_label}\n"
          f"Treatments: {len(treatments)}, Seeds: {args.n_seeds}, Rounds: {args.rounds}\n{'='*60}")

    results = run_experiment_grid(treatments=treatments, base_config=base, seeds=list(range(args.n_seeds)), output_dir=output)
    generate_all_plots(results, output)
    print(f"\n{'='*60}\nDONE: {output}/\n{'='*60}")


if __name__ == "__main__":
    main()
