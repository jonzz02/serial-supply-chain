import os
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
from itertools import product

from config import ExperimentConfig
from model import TwoStageSupplyChainModel
from centralsolver import compute_benchmark, compute_prior_rewards
from metrics import compute_run_metrics, aggregate_metrics
from agents import initialize_agent_benchmark


@dataclass
class TreatmentConfig:
    """One treatment: agent pair + action grid + initialization mode."""
    agent_retailer: str
    agent_supplier: str
    s_lower: int = 0
    s_upper: int = 60
    s_step: int = 1
    init_mode: str = "random"  # "random" or "benchmark"
    
    @property
    def name(self) -> str:
        return (f"{self.agent_retailer}_{self.agent_supplier}_"
                f"s{self.s_lower}-{self.s_upper}-{self.s_step}")
    
    @property
    def agent_pair(self) -> Tuple[str, str]:
        return (self.agent_retailer, self.agent_supplier)


# Default treatment grid (simplified for student scope)
DEFAULT_AGENT_PAIRS = [
    ("greedy", "greedy"),
    ("ucb", "ucb"),
    ("greedy", "ucb"),
    ("ucb", "greedy"),
]

# Single action grid
DEFAULT_ACTION_GRID = (0, 60, 1)

# Default initialization mode
DEFAULT_INIT_MODE = "random"


def create_treatment_grid(
    agent_pairs: List[Tuple[str, str]] = None,
    action_grid: Tuple[int, int, int] = None,
    init_mode: str = None,
) -> List[TreatmentConfig]:
    """Create treatment grid (agents × single grid × single init mode)."""
    pairs = agent_pairs or DEFAULT_AGENT_PAIRS
    grid = action_grid or DEFAULT_ACTION_GRID
    init = init_mode or DEFAULT_INIT_MODE
    
    treatments = []
    for (ret, sup) in pairs:
        treatments.append(TreatmentConfig(
            agent_retailer=ret,
            agent_supplier=sup,
            s_lower=grid[0],
            s_upper=grid[1],
            s_step=grid[2],
            init_mode=init,
        ))
    return treatments


def run_single_seed(config: ExperimentConfig, seed: int, ctot_opt: float,
                    s1_opt: int, s2_opt: int,
                    conv_window: int = 50, conv_threshold: float = 0.9) -> Dict[str, Any]:
    """Run one simulation with given seed, return metrics."""
    model = TwoStageSupplyChainModel(config=config, seed=seed)
    
    # Benchmark initialization if enabled
    if config.init_mode == "benchmark":
        r1_prior, r2_prior = compute_prior_rewards(config, s1_opt, s2_opt, seed_offset=1000 + seed)
        initialize_agent_benchmark(model.retailer, s1_opt, r1_prior, config.init_prior_strength)
        initialize_agent_benchmark(model.supplier, s2_opt, r2_prior, config.init_prior_strength)
    
    # Training phase
    model.run(config.rounds)
    
    metrics = compute_run_metrics(model, ctot_opt, config.rounds, config.warmup,
                                  conv_window, conv_threshold)
    metrics["seed"] = seed
    
    return metrics


def run_treatment(
    treatment: TreatmentConfig,
    base_config: ExperimentConfig,
    seeds: List[int],
    conv_window: int = 50,
    conv_threshold: float = 0.9,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run treatment across all seeds, compute benchmark, return aggregated results."""
    # Create config for this treatment
    config = ExperimentConfig(
        h1=base_config.h1,
        h2=base_config.h2,
        p_bo=base_config.p_bo,
        alpha=base_config.alpha,
        lam=base_config.lam,
        s_lower=treatment.s_lower,
        s_upper=treatment.s_upper,
        s_step=treatment.s_step,
        eps_start=base_config.eps_start,
        eps_end=base_config.eps_end,
        rounds=base_config.rounds,
        warmup=base_config.warmup,
        benchmark_rounds=base_config.benchmark_rounds,
        benchmark_warmup=base_config.benchmark_warmup,
        benchmark_n_seeds=base_config.benchmark_n_seeds,
        init_mode=treatment.init_mode,
        init_prior_strength=base_config.init_prior_strength,
        agent_retailer=treatment.agent_retailer,
        agent_supplier=treatment.agent_supplier,
    )
    
    # Compute benchmark (cached)
    s1_opt, s2_opt, ctot_opt = compute_benchmark(config)
    
    if verbose:
        print(f"  Treatment: {treatment.name}")
        print(f"    Benchmark: s1*={s1_opt}, s2*={s2_opt}, c*={ctot_opt:.2f}")
    
    # Run all seeds
    run_metrics = []
    timeseries_data = []
    
    for seed in seeds:
        m = run_single_seed(config, seed, ctot_opt, s1_opt, s2_opt,
                           conv_window, conv_threshold)
        
        ts = {
            "seed": seed,
            "total_costs": m.pop("_total_costs"),
            "s1_actions": m.pop("_s1_actions"),
            "s2_actions": m.pop("_s2_actions"),
            "cumulative_regret": m.pop("_cumulative_regret"),
        }
        timeseries_data.append(ts)
        run_metrics.append(m)
    
    # Aggregate
    agg = aggregate_metrics(run_metrics)
    agg["treatment"] = treatment.name
    agg["agent_retailer"] = treatment.agent_retailer
    agg["agent_supplier"] = treatment.agent_supplier
    agg["s_lower"] = treatment.s_lower
    agg["s_upper"] = treatment.s_upper
    agg["s_step"] = treatment.s_step
    agg["init_mode"] = treatment.init_mode
    agg["s1_opt"] = s1_opt
    agg["s2_opt"] = s2_opt
    agg["ctot_opt"] = ctot_opt
    
    return {
        "summary": agg,
        "run_metrics": run_metrics,
        "timeseries": timeseries_data,
        "config": config,
    }


def run_experiment_grid(
    treatments: List[TreatmentConfig] = None,
    base_config: ExperimentConfig = None,
    seeds: List[int] = None,
    n_seeds: int = 50,
    conv_window: int = 50,
    conv_threshold: float = 0.9,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run full experiment grid across treatments and seeds."""
    treatments = treatments or create_treatment_grid()
    base_config = base_config or ExperimentConfig()
    seeds = seeds or list(range(n_seeds))
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    if verbose:
        print(f"Running {len(treatments)} treatments × {len(seeds)} seeds")
        print(f"Base config: rounds={base_config.rounds}, warmup={base_config.warmup}, λ={base_config.lam}")
    
    all_results = []
    all_run_metrics = []
    all_timeseries = []
    
    start = time.time()
    
    for i, treatment in enumerate(treatments):
        if verbose:
            print(f"\n[{i+1}/{len(treatments)}]")
        
        result = run_treatment(
            treatment, base_config, seeds,
            conv_window, conv_threshold, verbose
        )
        
        all_results.append(result["summary"])
        
        for m in result["run_metrics"]:
            m["treatment"] = treatment.name
            m["init_mode"] = treatment.init_mode
            all_run_metrics.append(m)
        
        for ts in result["timeseries"]:
            ts["treatment"] = treatment.name
            all_timeseries.append(ts)
    
    elapsed = time.time() - start
    if verbose:
        print(f"\nTotal time: {elapsed:.1f}s")
    
    summary_df = pd.DataFrame(all_results)
    run_df = pd.DataFrame(all_run_metrics)
    
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    run_df.to_csv(os.path.join(output_dir, "runs.csv"), index=False)
    
    if verbose:
        print(f"\nResults saved to {output_dir}/")
        print(f"  summary.csv: {len(summary_df)} treatment summaries")
        print(f"  runs.csv: {len(run_df)} individual runs")
    
    return {
        "summary_df": summary_df,
        "run_df": run_df,
        "timeseries": all_timeseries,
        "treatments": treatments,
        "seeds": seeds,
        "base_config": base_config,
        "output_dir": output_dir,
    }


def run_robustness_scenarios(
    base_config: ExperimentConfig = None,
    seeds: List[int] = None,
    n_seeds: int = 50,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run robustness checks with alternative parameter scenarios."""
    base_config = base_config or ExperimentConfig()
    seeds = seeds or list(range(n_seeds))
    
    scenarios = {
        "baseline": {},
        "high_backorder": {"p_bo": 10.0},
        "low_demand": {"lam": 10.0},
        "high_demand": {"lam": 30.0},
        "asymmetric_alpha": {"alpha": 0.3},
    }
    
    all_summaries = []
    
    for scenario_name, overrides in scenarios.items():
        if verbose:
            print(f"\n=== Scenario: {scenario_name} ===")
        
        cfg_dict = asdict(base_config)
        cfg_dict.update(overrides)
        scenario_config = ExperimentConfig(**cfg_dict)
        
        treatments = create_treatment_grid()
        
        result = run_experiment_grid(
            treatments=treatments,
            base_config=scenario_config,
            seeds=seeds,
            output_dir=os.path.join(output_dir, scenario_name),
            verbose=verbose,
        )
        
        df = result["summary_df"].copy()
        df["scenario"] = scenario_name
        all_summaries.append(df)
    
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "robustness_summary.csv"), index=False)
    
    return {"robustness_df": combined}
