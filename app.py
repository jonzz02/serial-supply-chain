from mesa.visualization import SolaraViz, make_plot_component
from simulation.model import TwoStageSupplyChainModel
from simulation.config import ExperimentConfig

AGENTS = ["greedy", "ucb", "thompson", "exp3", "etc"]


class ModelWrapper(TwoStageSupplyChainModel):
    def __init__(self, seed=42, agent_retailer="greedy", agent_supplier="greedy", n_actions=20):
        cfg = ExperimentConfig(rounds=1000, agent_retailer=agent_retailer, agent_supplier=agent_supplier, n_actions=n_actions)
        super().__init__(config=cfg, seed=seed)


model_params = {
    "seed": {"type": "SliderInt", "value": 42, "min": 0, "max": 1000, "label": "Seed"},
    "agent_retailer": {"type": "Select", "value": "greedy", "values": AGENTS, "label": "Retailer"},
    "agent_supplier": {"type": "Select", "value": "greedy", "values": AGENTS, "label": "Supplier"},
    "n_actions": {"type": "SliderInt", "value": 20, "min": 5, "max": 61, "label": "Actions"},
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

