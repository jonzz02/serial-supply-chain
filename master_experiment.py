"""
Master Experiment Runner for Holistic Benchmark

This script runs a full factorial design across all treatment variables to create
a comprehensive benchmark as described in the project overview document.

Total treatments: Algorithms (5×5) × Prior Knowledge × Grid Size × Init Mode × ...
Seeds per treatment: 100 (configurable)
Total runs: treatments × seeds
"""

import os
import json
import time
import argparse
import pandas as pd
from itertools import product
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

from experiment_runner import TreatmentConfig, run_treatment, AVAILABLE_AGENTS
from simulation.config import ExperimentConfig


# ============================================================================
# PARAMETER VARIATIONS - Edit these lists to control the experiment design
# ============================================================================

# Algorithm pairs: full 5×5 grid (25 combinations)
# Set to True to use all combinations, or provide specific pairs
USE_FULL_ALGORITHM_GRID = True

# Prior Knowledge modes
# Options: "none", "demand", "orders", "both"
PRIOR_KNOWLEDGE = ["none", "demand", "orders", "both"] # TODO JONAS: not yet implemented like this

# Action Grid sizes
# Maps to (s_lower, s_upper, s_step) or n_actions
GRID_SIZE = ["coarse", "medium", "fine"]

# Initialization modes
# Options: "random", "benchmark"
INIT_MODE = ["random", "benchmark"] # TODO JONAS: not yet implemented like this

# Additional parameters (if needed for future extensions)
COOPERATION_MODE = ["competitive", "partial", "cooperative"]  # TODO JONAS: not yet implemented like this


# ============================================================================
# Grid Size Mapping
# ============================================================================

GRID_SIZE_MAP = {
    "coarse": {"s_lower": 0, "s_upper": 40, "s_step": 5},
    "medium": {"s_lower": 0, "s_upper": 60, "s_step": 1},
    "fine": {"s_lower": 0, "s_upper": 80, "s_step": 1},
}


# ============================================================================
# Treatment Generation
# ============================================================================

def create_master_treatment_grid() -> List[TreatmentConfig]:
    """
    Create full factorial grid of all treatment combinations.
    
    Returns:
        List of TreatmentConfig objects covering all parameter combinations.
    """
    treatments = []
    
    # Algorithm pairs
    if USE_FULL_ALGORITHM_GRID:
        agent_pairs = list(product(AVAILABLE_AGENTS, AVAILABLE_AGENTS))
    else:
        # Fallback to default pairs if needed
        agent_pairs = [("greedy", "greedy"), ("ucb", "ucb"), ("greedy", "ucb"), ("ucb", "greedy")]
    
    # Generate all combinations
    for (agent_r, agent_s), prior_knowledge, grid_size, init_mode, cooperation_mode in product(
        agent_pairs,
        PRIOR_KNOWLEDGE,
        GRID_SIZE,
        INIT_MODE,
        COOPERATION_MODE,
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
            # Note: cooperation_mode is not yet supported in TreatmentConfig
            # It's included in the parameter grid but not passed to TreatmentConfig
            # TODO: Add cooperation_mode field to TreatmentConfig or handle separately
        )
        # Store cooperation_mode as a custom attribute for future use
        treatment.cooperation_mode = cooperation_mode
        treatments.append(treatment)
    
    return treatments


# ============================================================================
# Parallel Execution
# ============================================================================

def run_treatment_wrapper(args: tuple) -> Dict[str, Any]:
    """
    Wrapper function for parallel execution of run_treatment.
    
    Args:
        args: Tuple of (treatment_dict, base_config_dict, seeds, conv_window, conv_threshold, verbose)
    
    Returns:
        Result dictionary from run_treatment
    """
    treatment_dict, base_config_dict, seeds, conv_window, conv_threshold, verbose = args
    
    # Reconstruct objects from dictionaries (needed for multiprocessing)
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
    """
    Run master experiment with parallel execution.
    
    Args:
        treatments: List of treatment configurations
        base_config: Base experiment configuration
        seeds: List of random seeds to run
        max_workers: Maximum number of parallel workers
        conv_window: Convergence window size
        conv_threshold: Convergence threshold
        output_dir: Output directory for results
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing all results and dataframes
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    if verbose:
        total_runs = len(treatments) * len(seeds)
        print(f"{'='*80}")
        print(f"Master Experiment: {len(treatments)} treatments × {len(seeds)} seeds = {total_runs:,} total runs")
        print(f"Parallel workers: {max_workers}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
    
    # Prepare arguments for parallel execution
    # Convert to dicts for pickling (multiprocessing requirement)
    base_config_dict = asdict(base_config)
    treatment_args = [
        (asdict(t), base_config_dict, seeds, conv_window, conv_threshold, False)  # verbose=False in parallel
        for t in treatments
    ]
    
    all_results, all_runs, all_ts, all_bench, all_nash = [], [], [], [], {}
    start_time = time.time()
    
    # Run treatments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_treatment = {
            executor.submit(run_treatment_wrapper, args): i
            for i, args in enumerate(treatment_args)
        }
        
        # Process completed tasks
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
                        "reward_mode": treatment.reward_mode,
                        "utility_mode": treatment.utility_mode
                    })
                    all_runs.append(m)
                
                all_ts.extend([
                    {**ts, "treatment": treatment.name, "treatment_full": treatment.full_name}
                    for ts in result["timeseries"]
                ])
                
                completed += 1
                if verbose:
                    elapsed = time.time() - start_time
                    
                    # Get cooperation_mode if it exists (custom attribute)
                    coop_mode = getattr(treatment, 'cooperation_mode', 'N/A')
                    
                    # Build detailed treatment info
                    treatment_info = (
                        f"Agents: {treatment.agent_retailer}×{treatment.agent_supplier} | "
                        f"Grid: {treatment.s_lower}-{treatment.s_upper} (step={treatment.s_step}) | "
                        f"Prior: {treatment.prior_knowledge} | "
                        f"Init: {treatment.init_mode} | "
                        f"Cooperation: {coop_mode}"
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
    
    # Save results
    summary_df = pd.DataFrame(all_results)
    run_df = pd.DataFrame(all_runs)
    bench_df = pd.DataFrame(all_bench)
    
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    run_df.to_csv(os.path.join(output_dir, "runs.csv"), index=False)
    bench_df.to_csv(os.path.join(output_dir, "benchmarks.csv"), index=False)
    
    with open(os.path.join(output_dir, "treatments.jsonl"), "w") as f:
        for t in treatments:
            f.write(json.dumps(t.to_dict()) + "\n")
    
    # Save experiment metadata
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
            "cooperation_mode": COOPERATION_MODE,
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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run master experiment with full factorial design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python master_experiment.py --n_seeds 100 --rounds 365 --warmup 50 --max_workers 8 --output_dir results_master
        """
    )
    parser.add_argument("--n_seeds", type=int, default=100, help="Number of random seeds per treatment")
    parser.add_argument("--rounds", type=int, default=365, help="Number of simulation rounds")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup rounds")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--output_dir", type=str, default="results_master", help="Output directory")
    parser.add_argument("--conv_window", type=int, default=50, help="Convergence window size")
    parser.add_argument("--conv_threshold", type=float, default=0.9, help="Convergence threshold")
    
    args = parser.parse_args()
    
    # Create treatment grid
    treatments = create_master_treatment_grid()
    
    # Create base config
    base_config = ExperimentConfig(rounds=args.rounds, warmup=args.warmup)
    
    # Generate seeds
    seeds = list(range(args.n_seeds))
    
    # Run experiment
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
