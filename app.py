from mesa.visualization import SolaraViz, make_plot_component
from simulation.model import TwoStageSupplyChainModel
from simulation.config import ExperimentConfig

AGENTS = ["greedy", "ucb", "thompson", "exp3", "etc"]


class ModelWrapper(TwoStageSupplyChainModel):
    def __init__(
        self,
        seed=42,
        rounds=1000,
        agent_retailer="greedy",
        agent_supplier="greedy",
        s_lower=0,
        s_upper=60,
        s_step=3,
        lam=20.0,
        h1=2.0,
        h2=1.0,
        p_bo=5.0,
        alpha=0.6,
        eps_start=0.8,
        eps_end=0.05,
        exp3_gamma=0.1,
        etc_explore_rounds=3,
    ):
        cfg = ExperimentConfig(
            rounds=rounds,
            agent_retailer=agent_retailer,
            agent_supplier=agent_supplier,
            s_lower=s_lower,
            s_upper=s_upper,
            s_step=s_step,
            lam=lam,
            h1=h1,
            h2=h2,
            p_bo=p_bo,
            alpha=alpha,
            eps_start=eps_start,
            eps_end=eps_end,
            exp3_gamma=exp3_gamma,
            etc_explore_rounds=etc_explore_rounds,
        )
        super().__init__(config=cfg, seed=seed)


model_params = {
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 1000, "label": "Seed"},
    "rounds": {"type": "SliderInt", "value": 1000, "min": 50, "max": 5000, "step": 50, "label": "Rounds"},
    "agent_retailer": {"type": "Select", "value": "greedy", "values": AGENTS, "label": "Retailer"},
    "agent_supplier": {"type": "Select", "value": "greedy", "values": AGENTS, "label": "Supplier"},
    "s_lower": {"type": "SliderInt", "value": 0, "min": 0, "max": 20, "label": "Action space lower"},
    "s_upper": {"type": "SliderInt", "value": 60, "min": 20, "max": 100, "label": "Action space upper"},
    "s_step": {"type": "SliderInt", "value": 3, "min": 1, "max": 10, "label": "Action space step"},
    "lam": {"type": "SliderFloat", "value": 20.0, "min": 1.0, "max": 60.0, "step": 1.0, "label": "Demand (lambda)"},
    "h1": {"type": "SliderFloat", "value": 2.0, "min": 0.0, "max": 10.0, "step": 0.1, "label": "Holding cost h1"},
    "h2": {"type": "SliderFloat", "value": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "label": "Holding cost h2"},
    "p_bo": {"type": "SliderFloat", "value": 5.0, "min": 0.0, "max": 20.0, "step": 0.5, "label": "Backorder penalty"},
    "alpha": {"type": "SliderFloat", "value": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Alpha"},
    "eps_start": {"type": "SliderFloat", "value": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Epsilon start"},
    "eps_end": {"type": "SliderFloat", "value": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Epsilon end"},
    "exp3_gamma": {"type": "SliderFloat", "value": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "label": "Exp3 gamma"},
    "etc_explore_rounds": {"type": "SliderInt", "value": 3, "min": 1, "max": 20, "label": "ETC explore rounds"},
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

