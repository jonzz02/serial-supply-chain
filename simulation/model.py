import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from .agents import create_agent
from .config import ExperimentConfig, DEFAULT_CONFIG


class TwoStageSupplyChainModel(Model):
    
    def __init__(self, config: ExperimentConfig = None, seed: int = None):
        super().__init__(seed=seed)
        self.config = config or DEFAULT_CONFIG
        
        self.retailer = create_agent(self, self.config.agent_retailer, "retailer")
        self.supplier = create_agent(self, self.config.agent_supplier, "supplier")
        self.rewards = {self.retailer: 0.0, self.supplier: 0.0}
        
        self.I1 = self.I2 = self.B1 = self.B2 = 0
        self.U1_prev = self.U2_prev = 0
        self.last_total_cost = self.cost_retailer = self.cost_supplier = 0.0
        self._max_steps = self.config.rounds
        
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
        cfg = self.config
        I1, I2, B1, B2 = self.I1, self.I2, self.B1, self.B2

        I1 += self.U1_prev
        I2 += self.U2_prev

        O1 = max(0, s1_loc - (I1 - B1))
        O2 = max(0, s2_loc - (I2 - B2))

        ship = min(I2, B2 + O1)
        I2 -= ship
        B2 = B2 + O1 - ship
        U1, U2 = ship, O2

        D = int(self.rng.poisson(cfg.lam))
        sales = min(I1, B1 + D)
        I1 -= sales
        B1 = B1 + D - sales

        H1 = (cfg.h1 + cfg.h2) * I1 + cfg.alpha * cfg.p_bo * B1
        H2 = cfg.h2 * (I2 + U1) + (1.0 - cfg.alpha) * cfg.p_bo * B1

        self.I1, self.I2, self.B1, self.B2 = I1, I2, B1, B2
        self.U1_prev, self.U2_prev = U1, U2
        self.last_total_cost = float(H1 + H2)
        self.cost_retailer, self.cost_supplier = float(H1), float(H2)

        bias = cfg.bias_backorder_factor
        if bias != 1.0:
            H1_p = (cfg.h1 + cfg.h2) * I1 + cfg.alpha * cfg.p_bo * B1 * bias
            H2_p = cfg.h2 * (I2 + U1) + (1.0 - cfg.alpha) * cfg.p_bo * B1 * bias
        else:
            H1_p, H2_p = H1, H2

        return float(H1), float(H2), float(H1_p), float(H2_p)

    def step(self):
        cfg = self.config
        self.agents.do("select_action")
        
        H1, H2, H1_p, H2_p = self.env_step(int(self.retailer.action), int(self.supplier.action))
        
        if cfg.reward_mode == "local":
            r1, r2 = -H1_p, -H2_p
        elif cfg.reward_mode == "global":
            r1 = r2 = -(H1_p + H2_p)
        elif cfg.reward_mode == "weighted_global":
            r1 = -(H1_p + cfg.reward_beta * H2_p)
            r2 = -(H2_p + cfg.reward_beta * H1_p)
        else:
            r1, r2 = -H1_p, -H2_p

        self.rewards[self.retailer], self.rewards[self.supplier] = r1, r2
        self.retailer.reward, self.supplier.reward = r1, r2

        self.agents.do("update_belief")
        self.retailer.reward_cum += r1
        self.supplier.reward_cum += r2
        self.datacollector.collect(self)
        
        if self.steps >= self._max_steps:
            self.running = False

    def run(self, rounds: int = None):
        self._max_steps = rounds or self.config.rounds
        self.run_model()
        return self
