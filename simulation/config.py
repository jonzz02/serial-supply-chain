from dataclasses import dataclass
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
    
    # Initialization mode: "random", "benchmark"
    init_mode: str = "random"
    init_prior_strength: int = 5
    
    # Prior knowledge: "none" or "demand_known"
    prior_knowledge: str = "none"
    prior_assumed_other: str = "central"  # "central" or specific action value
    prior_strength: int = 3  # Pseudo-count for prior knowledge init
    
    # Agent configuration
    agent_retailer: str = "greedy"
    agent_supplier: str = "greedy"
    
    # Cooperation mode: "competitive", "cooperative", "partial"
    cooperation_mode: str = "competitive"
    cooperation_beta: float = 0.5
    
    def action_space(self) -> np.ndarray:
        return np.arange(self.s_lower, self.s_upper + 1, self.s_step, dtype=int)
    
    def epsilon_at(self, t: int) -> float:
        if self.rounds <= 1:
            return self.eps_end
        frac = np.clip(t, 0, self.rounds - 1) / (self.rounds - 1)
        return (1.0 - frac) * self.eps_start + frac * self.eps_end
    
    def benchmark_key(self) -> str:
        acts = ",".join(str(a) for a in self.action_space())
        return f"lam{self.lam}_h1{self.h1}_h2{self.h2}_pbo{self.p_bo}_alpha{self.alpha}_grid[{acts}]_br{self.benchmark_rounds}_bw{self.benchmark_warmup}_bn{self.benchmark_n_seeds}"
    
    def game_key(self) -> str:
        acts = ",".join(str(a) for a in self.action_space())
        return (
            f"lam{self.lam}_h1{self.h1}_h2{self.h2}_pbo{self.p_bo}_alpha{self.alpha}"
            f"_coop{self.cooperation_mode}_beta{self.cooperation_beta}"
            f"_grid[{acts}]_pr{self.payoff_rounds}_pw{self.payoff_warmup}_pn{self.payoff_n_seeds}"
        )


DEFAULT_CONFIG = ExperimentConfig()

