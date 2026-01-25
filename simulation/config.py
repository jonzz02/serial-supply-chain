from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ExperimentConfig:
    
    # Cost structure
    h1: float = 2
    h2: float = 1
    p_bo: float = 5.0
    alpha: float = 0.6
    
    # Demand
    lam: float = 20.0
    
    # Action space (step-based)
    s_lower: int = 0
    s_upper: int = 60
    s_step: int = 1
    # Action space (count-based, alternative to step)
    n_actions: Optional[int] = None  # If set, overrides s_step
    
    # Epsilon-greedy schedule
    eps_start: float = 0.8
    eps_end: float = 0.05
    
    # Exp3 parameters
    exp3_gamma: float = 0.1
    
    # ETC (Explore-Then-Commit) parameters
    etc_explore_rounds: int = 3
    
    # Simulation
    rounds: int = 365
    warmup: int = 50
    
    # Benchmark estimation
    benchmark_rounds: int = 1500
    benchmark_warmup: int = 300
    benchmark_n_seeds: int = 3
    
    # Payoff matrix estimation
    payoff_rounds: int = 300
    payoff_warmup: int = 60
    payoff_n_seeds: int = 2
    
    # Initialization mode: "random", "benchmark", "random_prior"
    init_mode: str = "random"
    init_prior_strength: int = 5
    init_mean_scale: float = 1.0  # Scale for random_prior means
    
    # Prior knowledge: "none" or "demand_known"
    prior_knowledge: str = "none"
    prior_assumed_other: str = "central"  # "central" or specific action value
    prior_strength: int = 3  # Pseudo-count for prior knowledge init
    
    # Reward design: "local", "global", "weighted_global"
    reward_mode: str = "local"
    reward_beta: float = 0.0  # Weight for weighted_global mode
    
    # Utility function: "risk_neutral", "risk_averse"
    utility_mode: str = "risk_neutral"
    risk_rho: float = 0.0  # Risk aversion parameter for mean-variance
    
    # Biased utility
    bias_backorder_factor: float = 1.0  # Scale backorder cost in perceived reward
    
    # Agent configuration
    agent_retailer: str = "greedy"
    agent_supplier: str = "greedy"
    
    def action_space(self) -> np.ndarray:
        if self.n_actions and self.n_actions > 1:
            return _gen_grid(self.s_lower, self.s_upper, self.n_actions)
        return np.arange(self.s_lower, self.s_upper + 1, self.s_step, dtype=int)
    
    def epsilon_at(self, t: int) -> float:
        if self.rounds <= 1:
            return self.eps_end
        frac = np.clip(t, 0, self.rounds - 1) / (self.rounds - 1)
        return (1.0 - frac) * self.eps_start + frac * self.eps_end
    
    @property
    def agent_types(self) -> Tuple[str, str]:
        return (self.agent_retailer, self.agent_supplier)
    
    def benchmark_key(self) -> str:
        acts = ",".join(str(a) for a in self.action_space())
        return f"lam{self.lam}_h1{self.h1}_h2{self.h2}_pbo{self.p_bo}_alpha{self.alpha}_grid[{acts}]_br{self.benchmark_rounds}_bw{self.benchmark_warmup}_bn{self.benchmark_n_seeds}"
    
    def game_key(self) -> str:
        return f"{self.benchmark_key()}_rm{self.reward_mode}_rb{self.reward_beta}_pr{self.payoff_rounds}_pw{self.payoff_warmup}_pn{self.payoff_n_seeds}_bbf{self.bias_backorder_factor}"


def _gen_grid(lower: int, upper: int, n: int) -> np.ndarray:
    N = upper - lower + 1
    K = min(n, N)
    full = np.arange(lower, upper + 1, dtype=int)
    idx = np.linspace(0, N - 1, K).round().astype(int)
    grid = np.unique(full[idx])
    if len(grid) < K:
        grid_set = set(grid)
        for v in full:
            if v not in grid_set:
                grid_set.add(v)
                if len(grid_set) >= K:
                    break
        grid = np.array(sorted(grid_set), dtype=int)
    return np.sort(grid)


def generate_action_grid(lower: int, upper: int, step: int = None, n_actions: int = None) -> np.ndarray:
    if n_actions and n_actions > 1:
        return _gen_grid(lower, upper, n_actions)
    return np.arange(lower, upper + 1, step or 1, dtype=int)


DEFAULT_CONFIG = ExperimentConfig()

