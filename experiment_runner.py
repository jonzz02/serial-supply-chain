import os
import json
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
from itertools import product

from simulation.config import ExperimentConfig
from simulation.model import TwoStageSupplyChainModel
from simulation.agents import initialize_agent_benchmark, initialize_agent_prior_knowledge
from analysis.centralsolver import compute_benchmark, compute_pure_nash, compute_prior_rewards, compute_prior_knowledge_rewards
from analysis.metrics import compute_run_metrics, aggregate_metrics


@dataclass
class TreatmentConfig:
    agent_retailer: str
    agent_supplier: str
    s_lower: int = 0
    s_upper: int = 60
    s_step: int = 1
    init_mode: str = "random"
    prior_knowledge: str = "none"
    cooperation_mode: str = "competitive"
    cooperation_beta: float = 0.5
    
    @property
    def name(self) -> str:
        return f"{self.agent_retailer}_{self.agent_supplier}_s{self.s_lower}-{self.s_upper}-{self.s_step}"
    
    @property
    def full_name(self) -> str:
        parts = [self.name]
        if self.init_mode != "random": parts.append(f"init={self.init_mode}")
        if self.prior_knowledge != "none": parts.append(f"prior={self.prior_knowledge}")
        if self.cooperation_mode != "competitive":
            if self.cooperation_mode == "partial":
                parts.append(f"coop=partial_beta={self.cooperation_beta}")
            else:
                parts.append(f"coop={self.cooperation_mode}")
        return "_".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AVAILABLE_AGENTS = ["greedy", "ucb", "thompson", "exp3", "etc"]
DEFAULT_AGENT_PAIRS = [("greedy", "greedy"), ("ucb", "ucb"), ("greedy", "ucb"), ("ucb", "greedy")]


def create_treatment_grid(agent_pairs: List[Tuple[str, str]] = None, 
                          full_grid: bool = False, **kwargs) -> List[TreatmentConfig]:
    pairs = list(product(AVAILABLE_AGENTS, AVAILABLE_AGENTS)) if full_grid else (agent_pairs or DEFAULT_AGENT_PAIRS)
    return [TreatmentConfig(agent_retailer=r, agent_supplier=s, **kwargs) for r, s in pairs]


def _build_config(treatment: TreatmentConfig, base: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(
        h1=base.h1, h2=base.h2, p_bo=base.p_bo, alpha=base.alpha, lam=base.lam,
        s_lower=treatment.s_lower, s_upper=treatment.s_upper, s_step=treatment.s_step,
        eps_start=base.eps_start, eps_end=base.eps_end,
        exp3_gamma=base.exp3_gamma, etc_explore_rounds=base.etc_explore_rounds,
        rounds=base.rounds, warmup=base.warmup,
        benchmark_rounds=base.benchmark_rounds, benchmark_warmup=base.benchmark_warmup,
        benchmark_n_seeds=base.benchmark_n_seeds,
        payoff_rounds=base.payoff_rounds, payoff_warmup=base.payoff_warmup,
        payoff_n_seeds=base.payoff_n_seeds,
        init_mode=treatment.init_mode, init_prior_strength=base.init_prior_strength,
        prior_knowledge=treatment.prior_knowledge, prior_assumed_other=base.prior_assumed_other,
        prior_strength=base.prior_strength,
        agent_retailer=treatment.agent_retailer, agent_supplier=treatment.agent_supplier,
        cooperation_mode=treatment.cooperation_mode, cooperation_beta=treatment.cooperation_beta,
    )


def run_single_seed(config: ExperimentConfig, seed: int, ctot_opt: float, s1_opt: int, s2_opt: int,
                    ne_set: List[Tuple[int, int]] = None, conv_window: int = 50, conv_threshold: float = 0.9) -> Dict[str, Any]:
    model = TwoStageSupplyChainModel(config=config, seed=seed)
    
    if config.prior_knowledge == "demand_known":
        r1 = compute_prior_knowledge_rewards(config, "retailer", config.prior_assumed_other, seed + 3000)
        r2 = compute_prior_knowledge_rewards(config, "supplier", config.prior_assumed_other, seed + 4000)
        initialize_agent_prior_knowledge(model.retailer, r1, config.prior_strength)
        initialize_agent_prior_knowledge(model.supplier, r2, config.prior_strength)
    elif config.init_mode == "benchmark":
        r1, r2 = compute_prior_rewards(config, s1_opt, s2_opt, seed_offset=1000 + seed)
        initialize_agent_benchmark(model.retailer, s1_opt, r1, config.init_prior_strength)
        initialize_agent_benchmark(model.supplier, s2_opt, r2, config.init_prior_strength)
    
    model.run(config.rounds)
    metrics = compute_run_metrics(model, ctot_opt, config.rounds, config.warmup, conv_window, conv_threshold,
                                  s1_opt=s1_opt, s2_opt=s2_opt, ne_set=ne_set, config=config)
    metrics["seed"] = seed
    return metrics


def run_treatment(treatment: TreatmentConfig, base_config: ExperimentConfig, seeds: List[int],
                  conv_window: int = 50, conv_threshold: float = 0.9, verbose: bool = True) -> Dict[str, Any]:
    config = _build_config(treatment, base_config)
    s1_opt, s2_opt, ctot_opt = compute_benchmark(config)
    nash = compute_pure_nash(config)
    
    if verbose:
        print(f"  {treatment.full_name}: s*=({s1_opt},{s2_opt}), c*={ctot_opt:.2f}, NE={nash['ne_count']}")
    
    run_metrics, timeseries = [], []
    for seed in seeds:
        m = run_single_seed(config, seed, ctot_opt, s1_opt, s2_opt, nash["ne_set"], conv_window, conv_threshold)
        timeseries.append({
            "seed": seed, "total_costs": m.pop("_total_costs"), "s1_actions": m.pop("_s1_actions"),
            "s2_actions": m.pop("_s2_actions"), "cumulative_regret": m.pop("_cumulative_regret"),
            "agent_retailer": treatment.agent_retailer, "agent_supplier": treatment.agent_supplier,
            "treatment_full": treatment.full_name,
        })
        run_metrics.append(m)
    
    agg = aggregate_metrics(run_metrics)
    agg.update({
        "treatment": treatment.name, "treatment_full": treatment.full_name,
        "agent_retailer": treatment.agent_retailer, "agent_supplier": treatment.agent_supplier,
        "s_lower": treatment.s_lower, "s_upper": treatment.s_upper, "s_step": treatment.s_step,
        "init_mode": treatment.init_mode, "prior_knowledge": treatment.prior_knowledge,
        "cooperation_mode": treatment.cooperation_mode, "cooperation_beta": treatment.cooperation_beta,
        "s1_opt": s1_opt, "s2_opt": s2_opt, "ctot_opt": ctot_opt, "ne_count": nash["ne_count"],
    })
    
    return {"summary": agg, "run_metrics": run_metrics, "timeseries": timeseries, "config": config, "nash": nash, "treatment": treatment}


def run_experiment_grid(treatments: List[TreatmentConfig] = None, base_config: ExperimentConfig = None,
                        seeds: List[int] = None, n_seeds: int = 50, conv_window: int = 50, conv_threshold: float = 0.9,
                        output_dir: str = "results", verbose: bool = True) -> Dict[str, Any]:
    treatments = treatments or create_treatment_grid()
    base_config = base_config or ExperimentConfig()
    seeds = seeds or list(range(n_seeds))
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    if verbose:
        print(f"Running {len(treatments)} × {len(seeds)} seeds")
    
    all_results, all_runs, all_ts, all_bench, all_nash = [], [], [], [], {}
    start = time.time()
    
    for i, t in enumerate(treatments):
        if verbose:
            print(f"[{i+1}/{len(treatments)}]", end=" ")
        
        result = run_treatment(t, base_config, seeds, conv_window, conv_threshold, verbose)
        all_results.append(result["summary"])
        all_nash[t.full_name] = result["nash"]
        all_bench.append({"treatment": t.name, "treatment_full": t.full_name,
                          "s1_opt": result["summary"]["s1_opt"], "s2_opt": result["summary"]["s2_opt"],
                          "ctot_opt": result["summary"]["ctot_opt"], "ne_count": result["summary"]["ne_count"]})
        
        for m in result["run_metrics"]:
            m.update({"treatment": t.name, "treatment_full": t.full_name, "init_mode": t.init_mode,
                      "prior_knowledge": t.prior_knowledge,
                      "cooperation_mode": t.cooperation_mode, "cooperation_beta": t.cooperation_beta})
            all_runs.append(m)
        
        all_ts.extend([{**ts, "treatment": t.name, "treatment_full": t.full_name} for ts in result["timeseries"]])
    
    if verbose:
        print(f"\nDone in {time.time() - start:.1f}s")
    
    summary_df = pd.DataFrame(all_results)
    run_df = pd.DataFrame(all_runs)
    bench_df = pd.DataFrame(all_bench)
    
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    run_df.to_csv(os.path.join(output_dir, "runs.csv"), index=False)
    bench_df.to_csv(os.path.join(output_dir, "benchmarks.csv"), index=False)
    
    with open(os.path.join(output_dir, "treatments.jsonl"), "w") as f:
        for t in treatments:
            f.write(json.dumps(t.to_dict()) + "\n")
    
    if verbose:
        print(f"Saved: {output_dir}/ ({len(summary_df)} summaries, {len(run_df)} runs)")
    
    return {"summary_df": summary_df, "run_df": run_df, "benchmarks_df": bench_df, "timeseries": all_ts,
            "treatments": treatments, "seeds": seeds, "base_config": base_config, "output_dir": output_dir, "nash_results": all_nash}
