"""
Two-stage serial supply chain model with fixed agent roles.

RNG Design (Common Random Numbers):
- demand_rng: Independent stream for demand draws (seed). Identical across treatments
  for the same seed, enabling fair comparisons even when algorithms differ.
- algo_rng: Separate stream for algorithm sampling like Thompson (seed + 1).
- Mesa's self.random: Used by Greedy/UCB for ε-exploration and tie-breaking.
"""
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from agents import create_agent
from config import ExperimentConfig, DEFAULT_CONFIG


class TwoStageSupplyChainModel(Model):
    """
    Two-stage serial supply chain: Retailer (stage 1) -> Supplier (stage 2).
    Agent roles are FIXED: self.retailer is always stage 1, self.supplier is always stage 2.
    """

    def __init__(self, config: ExperimentConfig = None, seed: int = None):
        self.config = config or DEFAULT_CONFIG
        self._seed = seed if seed is not None else 42
        
        # Initialize Mesa with seed for reproducible self.random
        super().__init__(seed=self._seed)
        
        # Separate RNG streams for common random numbers
        self.demand_rng = np.random.default_rng(self._seed)      # demand only
        self.algo_rng = np.random.default_rng(self._seed + 1)    # Thompson sampling
        
        self.t = 0
        
        # Create agents with FIXED roles (no shuffle!)
        self.retailer = create_agent(self, self.config.agent_retailer, "retailer")
        self.supplier = create_agent(self, self.config.agent_supplier, "supplier")
        
        # Reward mapping
        self.rewards = {self.retailer: 0.0, self.supplier: 0.0}
        
        # Inventory state
        self.I1 = self.I2 = 0
        self.B1 = self.B2 = 0
        self.U1_prev = self.U2_prev = 0
        
        # Tracking
        self.last_total_cost = 0.0
        self.cost_retailer = 0.0
        self.cost_supplier = 0.0
        
        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Total Cost": "last_total_cost",
                "Cost Retailer": "cost_retailer",
                "Cost Supplier": "cost_supplier",
                "S1": lambda m: m.retailer.action,
                "S2": lambda m: m.supplier.action,
            },
            agent_reporters={
                "Role": "role",
                "Base Stock Level": "action",
                "Reward": "reward",
                "Cumulative Reward": "reward_cum",
            },
        )

    def env_step(self, s1_loc: int, s2_loc: int):
        """
        One period of the 2-stage serial supply chain.
        Returns (reward_retailer, reward_supplier) = negative costs.
        """
        cfg = self.config
        I1, I2, B1, B2 = self.I1, self.I2, self.B1, self.B2
        U1_prev, U2_prev = self.U1_prev, self.U2_prev

        # (1) arrivals
        I1 += U1_prev
        I2 += U2_prev

        # (2) local IP
        IP1 = I1 - B1
        IP2 = I2 - B2

        # (3) order-up-to
        O1 = max(0, s1_loc - IP1)
        O2 = max(0, s2_loc - IP2)

        # (4) releases
        ship = min(I2, B2 + O1)
        I2 -= ship
        B2 = B2 + O1 - ship
        U1 = ship
        U2 = O2

        # (5) demand
        D = int(self.demand_rng.poisson(cfg.lam))
        sales = min(I1, B1 + D)
        I1 -= sales
        B1 = B1 + D - sales

        # (6) costs
        H1 = (cfg.h1 + cfg.h2) * I1 + cfg.alpha * cfg.p_bo * B1
        H2 = cfg.h2 * (I2 + U1) + (1.0 - cfg.alpha) * cfg.p_bo * B1
        total_cost = float(H1 + H2)

        # Commit state
        self.I1, self.I2 = I1, I2
        self.B1, self.B2 = B1, B2
        self.U1_prev, self.U2_prev = U1, U2
        self.last_total_cost = total_cost
        self.cost_retailer = float(H1)
        self.cost_supplier = float(H2)

        return -float(H1), -float(H2)

    def step(self):
        
        # (1) Agents select actions (fixed order, no shuffle)
        self.retailer.select_action()
        self.supplier.select_action()
        s1 = int(self.retailer.action)
        s2 = int(self.supplier.action)

        # (2) Environment step
        r1, r2 = self.env_step(s1, s2)

        # (3) Assign rewards
        self.rewards[self.retailer] = r1
        self.retailer.reward = r1
        self.rewards[self.supplier] = r2
        self.supplier.reward = r2

        # (4) Update beliefs
        self.retailer.update_belief()
        self.supplier.update_belief()
        self.t += 1

        # (5) Cumulative rewards
        self.retailer.reward_cum += r1
        self.supplier.reward_cum += r2

        # (6) Collect data
        self.datacollector.collect(self)

    def run(self, rounds: int = None):
        n = rounds or self.config.rounds
        for _ in range(n):
            self.step()
        return self
