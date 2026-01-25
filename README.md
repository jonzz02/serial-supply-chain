# Serial Supply Chain Coordination Simulation

Mesa-based simulation of a 2-stage serial supply chain where bandit agents (retailer stage 1, supplier stage 2) learn discrete base-stock levels.

## Project Scope Questions

This codebase addresses the "Supply Chain Coordination" project:
- **Do agents converge?** Track `both_convergence_rate` and per-agent convergence metrics
- **How fast do they converge?** Track `s1_conv_time`, `s2_conv_time` (absolute round indices)
- **To what solutions?** Compare against Nash equilibria (`converged_to_ne`) and centralized optimum (`converged_to_central`)
- **Which mechanisms influence convergence?** Reward design, prior knowledge, initialization, action grid design, utility functions

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

You can verify the installation by running a quick test:
```bash
python run_experiments.py --scenario baseline --n_actions 61 --n_seeds 5 --rounds 100 --warmup 20 --output_dir results/
```

## Quick Start

```bash
# Run full 5×5 algorithm grid for baseline scenario
python run_experiments.py --scenario baseline --n_actions 61 --full_grid --n_seeds 50 --rounds 365 --warmup 50 --output_dir results/

# Run subset grid (default 4 pairs) for global reward scenario
python run_experiments.py --scenario global_reward --n_actions 61 --n_seeds 30 --rounds 365 --warmup 50 --output_dir results/

# Run with benchmark initialization
python run_experiments.py --scenario benchmark_init --n_actions 61 --full_grid --n_seeds 30 --rounds 365 --warmup 50 --output_dir results/
```

## Available Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | Local rewards, random init, risk neutral |
| `global_reward` | Global rewards (both agents see total cost) |
| `weighted_global_reward` | Weighted global with β=0.5 |
| `demand_known_prior` | Prior knowledge from demand distribution |
| `benchmark_init` | Warm start at centralized optimum |
| `random_prior_init` | Random prior means |
| `risk_averse` | Mean-variance utility with ρ=0.5 |
| `biased_backorder` | 2× backorder cost weight |

## Treatment Parameters

### Reward Design (`reward_mode`)
- `local` (default): Retailer reward = -H1, Supplier reward = -H2
- `global`: Both agents receive reward = -(H1+H2)
- `weighted_global`: Retailer = -(H1+β·H2), Supplier = -(H2+β·H1) where β = `reward_beta`

### Initialization (`init_mode`)
- `random` (default): Cold start with zero/uniform priors
- `benchmark`: Warm start with pseudo-counts at centralized optimum
- `random_prior`: Random prior means and small pseudo-counts for all arms

### Prior Knowledge (`prior_knowledge`)
- `none` (default): No prior knowledge
- `demand_known`: Initialize priors using offline Monte Carlo estimates assuming known demand distribution

### Utility Function (`utility_mode`)
- `risk_neutral` (default): Maximize expected reward
- `risk_averse`: Mean-variance utility with score = mean - ρ·std where ρ = `risk_rho`

### Biased Utility (`bias_backorder_factor`)
Scale backorder costs in perceived rewards (default 1.0). Higher values make agents overweight stockouts.

### Action Grid
- Step-based: `s_lower`, `s_upper`, `s_step` (default 0-60 step 1)
- Count-based: `n_actions` overrides step-based, uses linspace rounded to integers

## Output Files

All outputs are saved to `output_dir` (default: `results/`):

| File | Description |
|------|-------------|
| `summary.csv` | Treatment-level aggregates across seeds |
| `runs.csv` | Per-seed metrics with outcome classification |
| `benchmarks.csv` | Centralized optimum and Nash equilibrium counts |
| `treatments.jsonl` | Full treatment config dictionaries |
| `figures/` | All generated plots |

### Key Metrics in `runs.csv`
- `s1_mode`, `s2_mode`: Final converged action pair
- `converged_to_central`: Boolean, mode equals centralized optimum
- `converged_to_ne`: Boolean, mode is a Nash equilibrium
- `delta1`, `delta2`: Deviation incentives (≈0 at Nash equilibrium)
- `distance_to_central`: L1 distance to centralized optimum

### Key Metrics in `summary.csv`
- `converged_to_central_rate`, `converged_to_ne_rate`: Rates across seeds
- `ne_count`: Number of pure Nash equilibria
- `delta1_mean`, `delta2_mean`: Average deviation incentives among converged runs

## Nash Equilibrium Computation

The `centralsolver` module computes:
1. **Payoff matrices**: H1(s1,s2), H2(s1,s2), Htot(s1,s2) via Monte Carlo
2. **Best responses**: BR1(s2) = argmin H1(·,s2), BR2(s1) = argmin H2(s1,·)
3. **Pure Nash equilibria**: (s1,s2) where s1 ∈ BR1(s2) and s2 ∈ BR2(s1)

Results are cached by `config.config_key()` for efficiency.

## Plots

| Plot | Description |
|------|-------------|
| `learning_curves_all.png` | Mean total cost over time by algorithm pair |
| `convergence_comparison.png` | Convergence rates and times |
| `regret_comparison.png` | Total regret by treatment |
| `final_action_scatter.png` | Final (s1,s2) pairs with central optimum and NE overlay |
| `algorithm_heatmaps.png` | 5×5 heatmaps for regret, convergence, NE/central rates |
| `best_response_curves.png` | BR1/BR2 correspondences with NE intersections |
| `ne_classification_comparison.png` | Outcome classification rates |
| `deviation_incentives.png` | δ1, δ2 by treatment (≈0 = Nash stable) |

## Agent Types

| Type | Algorithm |
|------|-----------|
| `greedy` | ε-greedy with decaying exploration |
| `ucb` | UCB1 with optimism bonus |
| `thompson` | Thompson Sampling (Normal-Inverse-Gamma) |
| `exp3` | Exponential weights (adversarial) |
| `etc` | Explore-Then-Commit |

## Example: Treatment Sweep

```python
from experiment_runner import create_treatment_grid, run_experiment_grid
from config import ExperimentConfig

# Create 5×5 algorithm grid with specific settings
treatments = create_treatment_grid(
    full_grid=True,
    n_actions=61,
    init_mode="random",
    reward_mode="local",
)

results = run_experiment_grid(
    treatments=treatments,
    base_config=ExperimentConfig(rounds=365, warmup=50),
    n_seeds=50,
    output_dir="results_sweep",
)
```

## Reproducibility

- All RNG is seeded: `demand_rng` (demand draws), `algo_rng` (Thompson), Mesa's `self.random` (exploration)
- Benchmark and Nash computations are cached by config key
- Treatment configs are saved to `treatments.jsonl`