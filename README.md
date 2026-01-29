# Serial Supply Chain Coordination Simulation

Mesa-based simulation of a 2-stage serial supply chain where bandit agents (retailer stage 1, supplier stage 2) learn discrete base-stock levels.

## Project Scope Questions

This codebase addresses the "Supply Chain Coordination" project:
- **Do agents converge?** Track `both_convergence_rate` and per-agent convergence metrics
- **How fast do they converge?** Track `s1_conv_time`, `s2_conv_time` (absolute round indices)
- **To what solutions?** Compare against Nash equilibria (`converged_to_ne`) and centralized optimum (`converged_to_central`)
- **Which mechanisms influence convergence?** Cooperation modes (competitive/cooperative/partial), prior knowledge, initialization, action grid design, learning algorithms

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
python run_experiments.py --scenario baseline --s_upper 60 --s_step 5 --n_seeds 5 --rounds 100 --warmup 20 --output_dir results/
```

## Quick Start

```bash
# Run full 5×5 algorithm grid for baseline scenario
python run_experiments.py --scenario baseline --s_upper 60 --s_step 1 --full_grid --n_seeds 50 --rounds 365 --warmup 50 --output_dir results/

# Run with benchmark initialization
python run_experiments.py --scenario benchmark_init --s_upper 60 --s_step 1 --full_grid --n_seeds 30 --rounds 365 --warmup 50 --output_dir results/

# Run with demand knowledge prior (coarse grid for faster testing)
python run_experiments.py --scenario demand_known_prior --s_upper 60 --s_step 5 --n_seeds 30 --rounds 365 --warmup 50 --output_dir results/
```

## Available Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | Random init, no prior knowledge |
| `demand_known_prior` | Prior knowledge from demand distribution |
| `benchmark_init` | Warm start at centralized optimum |

## Treatment Parameters

### Initialization (`init_mode`)
- `random` (default): Cold start with zero/uniform priors
- `benchmark`: Warm start with pseudo-counts at centralized optimum

### Prior Knowledge (`prior_knowledge`)
- `none` (default): No prior knowledge
- `demand_known`: Initialize priors using offline Monte Carlo estimates assuming known demand distribution

### Cooperation Mode (`cooperation_mode`)
Controls how agents' rewards are computed from costs H1 (retailer) and H2 (supplier):

| Mode | Retailer Reward | Supplier Reward | Description |
|------|----------------|-----------------|-------------|
| `competitive` (default) | r1 = -H1 | r2 = -H2 | Each agent optimizes own local cost |
| `cooperative` | r1 = -(H1+H2) | r2 = -(H1+H2) | Both agents optimize joint system cost |
| `partial` | r1 = -(H1+β·H2) | r2 = -(H2+β·H1) | Partial internalization of other's cost (β ∈ [0,1]) |

**Beta parameter** (`cooperation_beta`): For `partial` mode, controls degree of cost internalization
- β = 0.0: equivalent to competitive
- β = 0.5: equal weight to own and other's cost
- β = 1.0: maximum internalization (approaches cooperative)

**Note**: Payoff matrices, Nash equilibria, and prior rewards are computed based on the cooperation mode, ensuring consistency between learning signals and equilibrium analysis.

### Action Grid
- Action space is defined by: `s_lower`, `s_upper`, `s_step`
- Default: 0 to 60 with step 1 (61 actions)
- Example: `s_lower=0, s_upper=40, s_step=5` gives actions [0, 5, 10, 15, 20, 25, 30, 35, 40]

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
- `converged_to_ne`: Boolean/None, mode is a Nash equilibrium (None if no pure NE exists)
- `delta1`, `delta2`: Deviation incentives (≈0 at Nash equilibrium)
- `distance_to_central`: L1 distance to centralized optimum
- `cooperation_mode`, `cooperation_beta`: Cooperation settings for this run

### Key Metrics in `summary.csv`
- `converged_to_central_rate`, `converged_to_ne_rate`: Rates across seeds
- `ne_exists`: Boolean, whether pure Nash equilibria exist for this treatment
- `ne_count`: Number of pure Nash equilibria
- `delta1_mean`, `delta2_mean`: Average deviation incentives among converged runs
- `cooperation_mode`, `cooperation_beta`: Cooperation settings for this treatment

## Nash Equilibrium Computation

The `centralsolver` module computes:
1. **Cost matrices**: H1(s1,s2), H2(s1,s2), Htot(s1,s2) via Monte Carlo
2. **Payoff matrices**: J1, J2 based on cooperation mode (e.g., J1=H1+β·H2 for partial)
3. **Best responses**: BR1(s2) = argmin J1(·,s2), BR2(s1) = argmin J2(s1,·)
4. **Pure Nash equilibria**: (s1,s2) where s1 ∈ BR1(s2) and s2 ∈ BR2(s1)

Results are cached by `config.game_key()` (includes cooperation_mode and beta) for efficiency.

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
from simulation.config import ExperimentConfig

# Create 5×5 algorithm grid with specific settings
treatments = create_treatment_grid(
    full_grid=True,
    s_lower=0,
    s_upper=60,
    s_step=1,
    init_mode="random",
    cooperation_mode="competitive",
)

results = run_experiment_grid(
    treatments=treatments,
    base_config=ExperimentConfig(rounds=365, warmup=50),
    n_seeds=50,
    output_dir="results_sweep",
)
```

## Master Experiment

Run nested factorial design across all treatment dimensions:

```bash
python master_experiment.py --n_seeds 100 --rounds 365 --warmup 50 --max_workers 8 --output_dir results_master
```

This creates treatments varying:
- Algorithm pairs (5×5 = 25 combinations)
- Grid sizes (coarse: 41 actions, medium: 61, fine: 81)
- Prior knowledge (none/demand_known)
- Initialization modes (random/benchmark) - nested within prior_knowledge
- Cooperation modes (competitive/cooperative/partial with β ∈ {0.25, 0.5, 0.75})

**Nested Design (no redundancy):**
- `prior_knowledge="demand_known"` → always uses `init_mode="random"` (prior handles initialization)
- `prior_knowledge="none"` → tests both `init_mode="random"` and `init_mode="benchmark"`

Total: 1,125 treatments × 100 seeds = 112,500 runs (~10-15 minutes)

## Reproducibility

- All RNG is seeded: `demand_rng` (demand draws), `algo_rng` (Thompson), Mesa's `self.random` (exploration)
- Benchmark and Nash computations are cached by config key
- Treatment configs are saved to `treatments.jsonl`