import os
import json
import time
import argparse
import pandas as pd
from itertools import product
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

from experiment_runner import TreatmentConfig, run_treatment, AVAILABLE_AGENTS
from simulation.config import ExperimentConfig
from analysis.plotting import generate_all_charts


USE_FULL_ALGORITHM_GRID = True

# Prior knowledge: how agents initialize their beliefs about the environment.
# Options:
#   - "none": agents start without any prior knowledge (default behavior)
#   - "demand_known": agents start with priors based on known demand distribution
PRIOR_KNOWLEDGE = ["none", "demand_known"]

# Action grid sizes: controls the discretization of the action space.
GRID_SIZE = ["coarse", "medium", "fine"]

# Initialization modes: how Q-tables are initialized at the start.
# Options:
#   - "random": Q-tables initialized randomly (default behavior)
#   - "benchmark": Q-tables seeded towards centralized benchmark optimum (s1*, s2*)
# NOTE: Full factorial design - testing all combinations of prior_knowledge × init_mode
#   to examine potential interaction effects between demand knowledge and action initialization
INIT_MODE = ["random", "benchmark"]

# Cooperation modes: how agents consider costs in their reward signal.
# - "competitive": r1 = -H1, r2 = -H2 (each optimizes own cost)
# - "cooperative": r1 = r2 = -(H1 + H2) (both optimize total cost)
# - "partial": r1 = -(H1 + beta*H2), r2 = -(H2 + beta*H1) - handled separately with COOP_BETAS
COOPERATION_MODE = ["competitive", "cooperative"]  # partial added separately with beta variation
COOP_BETAS = [0.25, 0.5, 0.75]  # beta values for partial mode (0.0=competitive, 1.0=max internalization)

GRID_SIZE_MAP = {
    "coarse": {"s_lower": 0, "s_upper": 40, "s_step": 1},   # 41 actions
    "medium": {"s_lower": 0, "s_upper": 60, "s_step": 1},   # 61 actions
    "fine": {"s_lower": 0, "s_upper": 80, "s_step": 1},     # 81 actions
}


def create_master_treatment_grid() -> List[TreatmentConfig]:
    agent_pairs = (list(product(AVAILABLE_AGENTS, AVAILABLE_AGENTS)) if USE_FULL_ALGORITHM_GRID
                   else [("greedy", "greedy"), ("ucb", "ucb"), ("greedy", "ucb"), ("ucb", "greedy")])
    treatments = []
    
    # Full factorial design: all combinations of prior_knowledge × init_mode
    # This allows us to examine potential interaction effects between these mechanisms
    
    # competitive and cooperative modes
    for (agent_r, agent_s), grid_size, cooperation_mode, prior_knowledge, init_mode in product(
        agent_pairs, GRID_SIZE, COOPERATION_MODE, PRIOR_KNOWLEDGE, INIT_MODE
    ):
        grid_params = GRID_SIZE_MAP[grid_size]
        
        treatment = TreatmentConfig(
            agent_retailer=agent_r,
            agent_supplier=agent_s,
            s_lower=grid_params["s_lower"],
            s_upper=grid_params["s_upper"],
            s_step=grid_params["s_step"],
            init_mode=init_mode,
            prior_knowledge=prior_knowledge,
            cooperation_mode=cooperation_mode,
        )
        treatments.append(treatment)
    
    # partial mode with varying beta
    for (agent_r, agent_s), grid_size, beta, prior_knowledge, init_mode in product(
        agent_pairs, GRID_SIZE, COOP_BETAS, PRIOR_KNOWLEDGE, INIT_MODE
    ):
        grid_params = GRID_SIZE_MAP[grid_size]
        
        treatment = TreatmentConfig(
            agent_retailer=agent_r,
            agent_supplier=agent_s,
            s_lower=grid_params["s_lower"],
            s_upper=grid_params["s_upper"],
            s_step=grid_params["s_step"],
            init_mode=init_mode,
            prior_knowledge=prior_knowledge,
            cooperation_mode="partial",
            cooperation_beta=beta,
        )
        treatments.append(treatment)
    
    return treatments


def run_treatment_wrapper(args: tuple) -> Dict[str, Any]:
    treatment_dict, base_config_dict, seeds, conv_window, conv_threshold, verbose = args
    treatment = TreatmentConfig(**treatment_dict)
    base_config = ExperimentConfig(**base_config_dict)
    return run_treatment(treatment, base_config, seeds, conv_window, conv_threshold, verbose)


def run_master_experiment(
    treatments: List[TreatmentConfig],
    base_config: ExperimentConfig,
    seeds: List[int],
    max_workers: int = 4,
    conv_window: int = 50,
    conv_threshold: float = 0.9,
    output_dir: str = "results_master",
    verbose: bool = True
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    if verbose:
        total_runs = len(treatments) * len(seeds)
        print(f"{'='*80}")
        print(f"Master Experiment: {len(treatments)} treatments × {len(seeds)} seeds = {total_runs:,} total runs")
        print(f"Parallel workers: {max_workers}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
    
    base_config_dict = asdict(base_config)
    treatment_args = [
        (asdict(t), base_config_dict, seeds, conv_window, conv_threshold, False)
        for t in treatments
    ]
    
    all_results, all_runs, all_ts, all_bench, all_nash = [], [], [], [], {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_treatment = {
            executor.submit(run_treatment_wrapper, args): i
            for i, args in enumerate(treatment_args)
        }
        
        completed = 0
        for future in as_completed(future_to_treatment):
            i = future_to_treatment[future]
            try:
                result = future.result()
                treatment = treatments[i]
                
                all_results.append(result["summary"])
                all_nash[treatment.full_name] = result["nash"]
                all_bench.append({
                    "treatment": treatment.name,
                    "treatment_full": treatment.full_name,
                    "s1_opt": result["summary"]["s1_opt"],
                    "s2_opt": result["summary"]["s2_opt"],
                    "ctot_opt": result["summary"]["ctot_opt"],
                    "ne_count": result["summary"]["ne_count"]
                })
                
                for m in result["run_metrics"]:
                    m.update({
                        "treatment": treatment.name,
                        "treatment_full": treatment.full_name,
                        "init_mode": treatment.init_mode,
                        "prior_knowledge": treatment.prior_knowledge,
                        "cooperation_mode": treatment.cooperation_mode,
                        "cooperation_beta": treatment.cooperation_beta,
                        "s_lower": treatment.s_lower,
                        "s_upper": treatment.s_upper,
                        "s_step": treatment.s_step,
                    })
                    all_runs.append(m)
                
                all_ts.extend([
                    {**ts, "treatment": treatment.name, "treatment_full": treatment.full_name}
                    for ts in result["timeseries"]
                ])
                
                completed += 1
                if verbose:
                    elapsed = time.time() - start_time
                    coop_str = treatment.cooperation_mode
                    if treatment.cooperation_mode == "partial":
                        coop_str = f"partial(β={treatment.cooperation_beta})"
                    treatment_info = (
                        f"Agents: {treatment.agent_retailer}×{treatment.agent_supplier} | "
                        f"Grid: {treatment.s_lower}-{treatment.s_upper} (step={treatment.s_step}) | "
                        f"Prior: {treatment.prior_knowledge} | "
                        f"Init: {treatment.init_mode} | "
                        f"Coop: {coop_str}"
                    )
                    print(f"\n[{completed}/{len(treatments)}] Treatment completed:")
                    print(f"  {treatment_info}")
                    print(f"  Time: {elapsed:.1f}s elapsed")
            
            except Exception as e:
                treatment = treatments[i]
                print(f"ERROR in treatment {treatment.full_name}: {e}")
                import traceback
                traceback.print_exc()
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average: {total_time/len(treatments):.1f}s per treatment")
        print(f"{'='*80}\n")
    
    summary_df = pd.DataFrame(all_results)
    run_df = pd.DataFrame(all_runs)
    bench_df = pd.DataFrame(all_bench)
    
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    run_df.to_csv(os.path.join(output_dir, "runs.csv"), index=False)
    bench_df.to_csv(os.path.join(output_dir, "benchmarks.csv"), index=False)
    
    with open(os.path.join(output_dir, "treatments.jsonl"), "w") as f:
        for t in treatments:
            f.write(json.dumps(t.to_dict()) + "\n")
    
    metadata = {
        "n_treatments": len(treatments),
        "n_seeds": len(seeds),
        "total_runs": len(treatments) * len(seeds),
        "max_workers": max_workers,
        "conv_window": conv_window,
        "conv_threshold": conv_threshold,
        "parameter_variations": {
            "algorithms": "full_grid" if USE_FULL_ALGORITHM_GRID else "subset",
            "prior_knowledge": PRIOR_KNOWLEDGE,
            "grid_size": GRID_SIZE,
            "init_mode": INIT_MODE,
            "cooperation_mode": COOPERATION_MODE + ["partial"],
            "coop_betas": COOP_BETAS,
        },
        "base_config": asdict(base_config),
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"Saved results to {output_dir}/")
        print(f"  - summary.csv: {len(summary_df)} treatments")
        print(f"  - runs.csv: {len(run_df)} runs")
        print(f"  - benchmarks.csv: {len(bench_df)} benchmarks")
        print(f"  - treatments.jsonl: {len(treatments)} treatments")
        print(f"  - metadata.json: experiment configuration")

    print(f"\nGenerating charts...")
    generate_all_charts(results_dir=output_dir, output_dir=os.path.join(output_dir, "figures"))

    return {
        "summary_df": summary_df,
        "run_df": run_df,
        "benchmarks_df": bench_df,
        "timeseries": all_ts,
        "treatments": treatments,
        "seeds": seeds,
        "base_config": base_config,
        "output_dir": output_dir,
        "nash_results": all_nash,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Run master experiment with full factorial design")
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of random seeds per treatment")
    parser.add_argument("--rounds", type=int, default=365, help="Number of simulation rounds")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup rounds")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--output_dir", type=str, default="results_master", help="Output directory")
    parser.add_argument("--conv_window", type=int, default=50, help="Convergence window size")
    parser.add_argument("--conv_threshold", type=float, default=0.9, help="Convergence threshold")
    
    args = parser.parse_args()
    
    treatments = create_master_treatment_grid()
    base_config = ExperimentConfig(rounds=args.rounds, warmup=args.warmup)
    seeds = list(range(args.n_seeds))
    
    results = run_master_experiment(
        treatments=treatments,
        base_config=base_config,
        seeds=seeds,
        max_workers=args.max_workers,
        conv_window=args.conv_window,
        conv_threshold=args.conv_threshold,
        output_dir=args.output_dir,
        verbose=True
    )
    
    print(f"\nMaster experiment completed successfully!")
    print(f"Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
