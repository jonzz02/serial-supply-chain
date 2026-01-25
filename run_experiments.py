#!/usr/bin/env python3
import argparse
import os
from simulation.config import ExperimentConfig
from experiment_runner import run_experiment_grid, TreatmentConfig
from analysis.plotting import generate_all_plots

SCENARIOS = {
    # Baseline: each agent minimizes its own (perceived) local cost.
    "baseline": {
        "reward_mode": "local",            # reward = -own cost (possibly biased via bias_backorder_factor)
        "reward_beta": 0.0,                # only used for "weighted_global"
        "init_mode": "random",             # random belief initialization
        "prior_knowledge": "none",         # no domain prior
        "utility_mode": "risk_neutral",    # use rewards as-is
        "risk_rho": 0.0,                   # only used for "risk_averse"
        "bias_backorder_factor": 1.0,      # 1.0 = unbiased perceived backorder cost
    },

    # Global reward: both agents are rewarded by the negative total system cost.
    "global_reward": {
        "reward_mode": "global",           # reward = -(H1 + H2) for both agents
        "reward_beta": 0.0,
        "init_mode": "random",
        "prior_knowledge": "none",
        "utility_mode": "risk_neutral",
        "risk_rho": 0.0,
        "bias_backorder_factor": 1.0,
    },

    # Weighted global reward: each agent optimizes a mix of total cost and the other agent's cost.
    # (See your model.step(): r1 = -(H1 + beta*H2), r2 = -(H2 + beta*H1))
    "weighted_global_reward": {
        "reward_mode": "weighted_global",  # reward mixes own + cross-term cost
        "reward_beta": 0.5,                # cross-weight beta (symmetric)
        "init_mode": "random",
        "prior_knowledge": "none",
        "utility_mode": "risk_neutral",
        "risk_rho": 0.0,
        "bias_backorder_factor": 1.0,
    },

    # Prior knowledge: agents start with a "demand-known" prior over actions (pseudo-count prior means).
    # Note: In experiment_runner.py you forbid combining special init/prior with non-local rewards.
    "demand_known_prior": {
        "reward_mode": "local",
        "reward_beta": 0.0,
        "init_mode": "random",
        "prior_knowledge": "demand_known", # initialize from compute_prior_knowledge_rewards(...)
        "utility_mode": "risk_neutral",
        "risk_rho": 0.0,
        "bias_backorder_factor": 1.0,
    },

    # Benchmark init: seed beliefs towards the centralized benchmark optimum (s1*, s2*).
    # Note: Must stay with local + unbiased reward due to your consistency check.
    "benchmark_init": {
        "reward_mode": "local",
        "reward_beta": 0.0,
        "init_mode": "benchmark",          # initialize_agent_benchmark(...)
        "prior_knowledge": "none",
        "utility_mode": "risk_neutral",
        "risk_rho": 0.0,
        "bias_backorder_factor": 1.0,
    },

    # Random-prior init: randomized pseudo-counts and means (a "noisy prior" over arms).
    # Note: Must stay with local + unbiased reward due to your consistency check.
    "random_prior_init": {
        "reward_mode": "local",
        "reward_beta": 0.0,
        "init_mode": "random_prior",       # initialize_agent_random_prior(...)
        "prior_knowledge": "none",
        "utility_mode": "risk_neutral",
        "risk_rho": 0.0,
        "bias_backorder_factor": 1.0,
    },

    # Risk-averse utility: agents trade off mean reward vs. reward volatility (mean - rho * std).
    # (Only affects agents that use BanditAgent._get_scores(), i.e., Greedy/UCB/ETC.)
    "risk_averse": {
        "reward_mode": "local",
        "reward_beta": 0.0,
        "init_mode": "random",
        "prior_knowledge": "none",
        "utility_mode": "risk_averse",     # activate risk penalty in _get_scores()
        "risk_rho": 0.5,                   # risk aversion strength
        "bias_backorder_factor": 1.0,
    },

    # Biased backorder perception: agents *perceive* backorders as more costly than reality.
    # This changes the reward signal (H1_p, H2_p) while the physical system dynamics stay the same.
    # Note: Counts as "non-local" in your consistency check (bias_backorder_factor != 1.0).
    "biased_backorder": {
        "reward_mode": "local",
        "reward_beta": 0.0,
        "init_mode": "random",
        "prior_knowledge": "none",
        "utility_mode": "risk_neutral",
        "risk_rho": 0.0,
        "bias_backorder_factor": 2.0,      # perceived backorder cost multiplier
    },
}

AGENTS = ["greedy", "ucb", "thompson", "exp3", "etc"]
DEFAULT_PAIRS = [("greedy", "greedy"), ("ucb", "ucb"), ("greedy", "ucb"), ("ucb", "greedy")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--n_actions", type=int, required=True)
    parser.add_argument("--full_grid", action="store_true")
    parser.add_argument("--n_seeds", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    overrides = SCENARIOS[args.scenario]
    pairs = [(r, s) for r in AGENTS for s in AGENTS] if args.full_grid else DEFAULT_PAIRS
    grid_label = "full" if args.full_grid else "subset"
    output = os.path.join(args.output_dir, args.scenario, f"K{args.n_actions}", grid_label)
    
    treatments = [TreatmentConfig(agent_retailer=r, agent_supplier=s, n_actions=args.n_actions, **overrides) for r, s in pairs]
    base = ExperimentConfig(rounds=args.rounds, warmup=args.warmup)

    print(f"{'='*60}\nScenario: {args.scenario}, K={args.n_actions}, Grid: {grid_label}\n"
          f"Treatments: {len(treatments)}, Seeds: {args.n_seeds}, Rounds: {args.rounds}\n{'='*60}")

    results = run_experiment_grid(treatments=treatments, base_config=base, seeds=list(range(args.n_seeds)), output_dir=output)
    generate_all_plots(results, output)
    print(f"\n{'='*60}\nDONE: {output}/\n{'='*60}")


if __name__ == "__main__":
    main()
