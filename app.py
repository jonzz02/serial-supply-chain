from mesa.visualization import SolaraViz, make_plot_component
from simulation.model import TwoStageSupplyChainModel
from simulation.config import ExperimentConfig

AGENTS = ["greedy", "ucb", "thompson", "exp3", "etc"]
REWARD_MODES = ["local", "global", "weighted_global"]
UTILITY_MODES = ["risk_neutral", "risk_averse"]


class ModelWrapper(TwoStageSupplyChainModel):
    def __init__(
        self,
        seed=42,
        rounds=1000,
        agent_retailer="greedy",
        agent_supplier="greedy",
        n_actions=20,
        lam=20.0,
        h1=2.0,
        h2=1.0,
        p_bo=5.0,
        alpha=0.6,
        eps_start=0.8,
        eps_end=0.05,
        exp3_gamma=0.1,
        etc_explore_rounds=3,
        reward_mode="local",
        reward_beta=0.0,
        utility_mode="risk_neutral",
        risk_rho=0.0,
        bias_backorder_factor=1.0,
    ):
        cfg = ExperimentConfig(
            rounds=rounds,
            agent_retailer=agent_retailer,
            agent_supplier=agent_supplier,
            n_actions=n_actions,
            lam=lam,
            h1=h1,
            h2=h2,
            p_bo=p_bo,
            alpha=alpha,
            eps_start=eps_start,
            eps_end=eps_end,
            exp3_gamma=exp3_gamma,
            etc_explore_rounds=etc_explore_rounds,
            reward_mode=reward_mode,
            reward_beta=reward_beta,
            utility_mode=utility_mode,
            risk_rho=risk_rho,
            bias_backorder_factor=bias_backorder_factor,
        )
        super().__init__(config=cfg, seed=seed)


model_params = {
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 1000, "label": "Seed"},
    "rounds": {"type": "SliderInt", "value": 1000, "min": 50, "max": 5000, "step": 50, "label": "Rounds"},
    "agent_retailer": {"type": "Select", "value": "greedy", "values": AGENTS, "label": "Retailer"},
    "agent_supplier": {"type": "Select", "value": "greedy", "values": AGENTS, "label": "Supplier"},
    "n_actions": {"type": "SliderInt", "value": 20, "min": 5, "max": 61, "label": "Actions"},
    "lam": {"type": "SliderFloat", "value": 20.0, "min": 1.0, "max": 60.0, "step": 1.0, "label": "Demand (lambda)"},
    "h1": {"type": "SliderFloat", "value": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "label": "Holding cost h1"},
    "h2": {"type": "SliderFloat", "value": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "label": "Holding cost h2"},
    "p_bo": {"type": "SliderFloat", "value": 5.0, "min": 0.0, "max": 20.0, "step": 0.5, "label": "Backorder penalty"},
    "alpha": {"type": "SliderFloat", "value": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Alpha"},
    "eps_start": {"type": "SliderFloat", "value": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Epsilon start"},
    "eps_end": {"type": "SliderFloat", "value": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Epsilon end"},
    "exp3_gamma": {"type": "SliderFloat", "value": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "label": "Exp3 gamma"},
    "etc_explore_rounds": {"type": "SliderInt", "value": 3, "min": 1, "max": 20, "label": "ETC explore rounds"},
    "reward_mode": {"type": "Select", "value": "local", "values": REWARD_MODES, "label": "Reward mode"},
    "reward_beta": {"type": "SliderFloat", "value": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Reward beta"},
    "utility_mode": {"type": "Select", "value": "risk_neutral", "values": UTILITY_MODES, "label": "Utility mode"},
    "risk_rho": {"type": "SliderFloat", "value": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "label": "Risk rho"},
    "bias_backorder_factor": {"type": "SliderFloat", "value": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "label": "Backorder bias"},
}

CostPlot = make_plot_component("Total Cost")
S1Plot = make_plot_component("S1")
S2Plot = make_plot_component("S2")

model = ModelWrapper()

page = SolaraViz(
    model,
    components=[CostPlot, S1Plot, S2Plot],
    model_params=model_params,
    name="Supply Chain Coordination",
    play_interval=50,
)
page

