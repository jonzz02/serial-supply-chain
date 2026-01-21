from dataclasses import dataclass
from typing import Tuple
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
    
    # Action space
    s_lower: int = 0
    s_upper: int = 60
    s_step: int = 1
    
    # Epsilon-greedy schedule
    eps_start: float = 0.8
    eps_end: float = 0.05
    
    # Simulation
    rounds: int = 365
    warmup: int = 50  # Rounds excluded from metrics (transient)
    
    # Benchmark estimation (decoupled from training)
    benchmark_rounds: int = 1500
    benchmark_warmup: int = 300
    benchmark_n_seeds: int = 3
    
    # Initialization mode: "random" (cold start) or "benchmark" (warm start)
    init_mode: str = "random"
    init_prior_strength: int = 5  # pseudo-count for benchmark init
    
    # Agent configuration
    agent_retailer: str = "greedy"
    agent_supplier: str = "greedy"
    
    def action_space(self) -> np.ndarray:
        return np.arange(self.s_lower, self.s_upper + 1, self.s_step, dtype=int)
    
    def epsilon_at(self, t: int) -> float:
        if self.rounds <= 1:
            return self.eps_end
        frac = np.clip(t, 0, self.rounds - 1) / (self.rounds - 1)
        return (1.0 - frac) * self.eps_start + frac * self.eps_end
    
    @property
    def agent_types(self) -> Tuple[str, str]:
        return (self.agent_retailer, self.agent_supplier)
    
    def config_key(self) -> str:
        return (f"lam{self.lam}_h1{self.h1}_h2{self.h2}_pbo{self.p_bo}_"
                f"alpha{self.alpha}_s{self.s_lower}-{self.s_upper}-{self.s_step}_"
                f"br{self.benchmark_rounds}_bw{self.benchmark_warmup}_bn{self.benchmark_n_seeds}")


# Default config
DEFAULT_CONFIG = ExperimentConfig()
