# Two-Stage Serial Supply Chain Simulation (Project 2.1)

This repository studies **convergence behavior under different agent configurations** in a **two-stage serial supply chain** where both stages learn **base-stock levels** via multi-armed bandits.

The main question: Do the agents converge, how fast, and to which outcomes when we change the learning algorithms (and allow asymmetric pairs)?


## Model Overview

### Supply Chain Structure
- **Retailer (Stage 1)** faces stochastic end-customer demand:  
  \( D_t \sim \text{Poisson}(\lambda) \), i.i.d. over time
- **Supplier (Stage 2)** fulfills retailer orders; upstream source is assumed unconstrained
- **Lead times:** \( L_1 = L_2 = 1 \)
- Each agent independently chooses a **local base-stock level** \( s \) from a discrete action grid.

### Within-Period Dynamics (per round)
1. **Arrivals:** shipments from last round arrive
2. **Decisions:** retailer chooses \( s_1 \), supplier chooses \( s_2 \)
3. **Shipping:** supplier ships available inventory to retailer
4. **Demand:** customer demand realizes at retailer
5. **Costs:** holding and backorder costs are charged; agents observe reward = negative cost

### Cost Structure (per period)
- Retailer cost:
  \[
  (h_1 + h_2)\,I_1 + \alpha\,p_{bo}\,B_1
  \]
- Supplier cost:
  \[
  h_2\,(I_2 + U_1) + (1-\alpha)\,p_{bo}\,B_1
  \]
Where:
- \( I \) = on-hand inventory, \( B \) = backorders, \( U \) = in-transit

**Default parameters:**
- \( h_1 = 0.5 \), \( h_2 = 0.5 \): holding costs
- \( p_{bo} = 5.0 \): backorder penalty
- \( \alpha = 0.5 \): backorder cost allocation (50% retailer, 50% supplier)
- \( \lambda = 20.0 \): Poisson demand rate

**Reward signal:** each agent receives **negative local cost**.


## Learning Agents

Implemented bandit policies:
- **ε-greedy** (`greedy`): explores with probability ε (decaying linearly from 0.8 to 0.05 over training rounds), otherwise exploits best estimated arm
- **UCB1** (`ucb`): selects arm maximizing  
  \( \hat{\mu}(a) + \sqrt{2 \log(t+1)/n(a)} \)
- **Thompson Sampling** (implemented but not included in default treatment grid)

**Epsilon decay schedule (for ε-greedy):**
- \( \epsilon_{\text{start}} = 0.8 \): initial exploration rate
- \( \epsilon_{\text{end}} = 0.05 \): final exploration rate
- Linear decay: \( \epsilon_t = (1 - t/T) \cdot 0.8 + (t/T) \cdot 0.05 \), where \( T \) is total training rounds


## Treatments (Student Scope)

We run **4 agent pair configurations** on **one action grid**:

| Retailer | Supplier | Treatment Name |
|---------:|---------:|----------------|
| greedy   | greedy   | `greedy_greedy_s0-60-1` |
| ucb      | ucb      | `ucb_ucb_s0-60-1` |
| greedy   | ucb      | `greedy_ucb_s0-60-1` |
| ucb      | greedy   | `ucb_greedy_s0-60-1` |

**Action grid:** \( s \in \{0,1,\dots,60\} \) (61 actions)

This setup intentionally includes **asymmetric pairs** to study coordination/frictions when only one stage uses a “stronger” algorithm.


## Warmup-Aware Evaluation

All reported metrics and plots **exclude the first `warmup` rounds** (default: 50).  
Reason: early rounds contain transient pipeline/initialization effects and distort steady-state learning comparisons.

Formally, metrics are computed on rounds:
\[
t \in [\text{warmup},\ \text{rounds})
\]


## Benchmark (System Optimum)

We compute a **system-optimal benchmark** by enumerating all pairs \((s_1, s_2)\) on the action grid and estimating long-run average total cost via Monte Carlo:

- \( (s_1^\*, s_2^\*) \): best pair on the grid
- \( c^\* \): estimated expected total cost under \((s_1^\*, s_2^\*)\)

**Benchmark estimation parameters:**
- `benchmark_rounds = 1500`: simulation length per policy evaluation
- `benchmark_warmup = 300`: warmup rounds excluded from benchmark cost estimation
- `benchmark_n_seeds = 3`: number of random seeds averaged for robustness

This benchmark is used to compute **regret** and is cached for efficiency.


## Metrics

### Convergence
- **Converged:** in the last \(W\) rounds (convergence window, default \(W=50\)), the most frequent action occurs with share ≥ 90%
- **Convergence time:** first round index (absolute time) from which the convergence condition holds until the end
- **Volatility:** fraction of action changes (post-warmup)

**Note:** The convergence window (\(W=50\)) is a separate parameter from the warmup period. Convergence is evaluated on post-warmup data using a sliding window.

### Regret
- Per-round regret (post-warmup):
  \[
  \text{regret}_t = \text{total\_cost}_t - c^\*
  \]
- **Cumulative regret (post-warmup):** cumulative sum of per-round regret
- **train_total_regret:** total post-warmup regret over the training horizon

> Note: In multi-agent learning, theoretical regret guarantees generally do not apply because each agent faces a **non-stationary environment** (the other agent is learning). We therefore use regret mainly as a **relative performance** measure across treatments, and convergence metrics to assess stabilization.

### Aggregation Across Seeds
For each treatment and metric we report: **mean**, **std**, **min**, **max** over random seeds.


## RNG Design (Fair Comparisons)

To enable fair comparisons across treatments:
- **`demand_rng` (seed)** generates the demand sequence.  
  For the same seed, **all treatments see the exact same demand draws** (common random numbers).
- **Algorithm randomness** (exploration, Thompson sampling) may differ across algorithms and is intended.


## Output

```
results/
├── summary.csv  # one row per treatment: aggregated metrics (mean/std/min/max)
├── runs.csv     # one row per (treatment, seed): individual run metrics
└── figures/
    ├── learning_curves_all.png      # mean total cost over time (all treatments overlaid, post-warmup)
    ├── regret_comparison.png        # bar chart of post-warmup total regret by treatment
    └── convergence_comparison.png   # convergence rates + convergence times by treatment
```

## CLI Options

- `--n_seeds`: Number of random seeds (default: 50)
- `--rounds`: Training rounds (default: 365)
- `--warmup`: Warmup rounds excluded from metrics (default: 50)

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
python run_experiments.py --n_seeds 5 --rounds 100
```

## Quick Start

```bash
# Run default experiment (50 seeds, 365 rounds, 50 warmup)
python run_experiments.py

# Quick test (5 seeds, 100 rounds)
python run_experiments.py --n_seeds 5 --rounds 100

# Full experiment with custom warmup
python run_experiments.py --n_seeds 100 --warmup 100
```


## Initialization Modes

The code supports two initialization modes (controlled via `ExperimentConfig.init_mode`):

1. **`"random"` (default):** Agents start with no prior knowledge; all Q-values initialized to zero
2. **`"benchmark"` (optional):** Agents initialized with pseudo-observations at the optimal policy  
   - Each agent receives `init_prior_strength` (default: 5) pseudo-observations at \((s_1^\*, s_2^\*)\)
   - Provides a "warm start" to accelerate convergence
   - Not used in the default treatment grid

This feature is implemented but not explored in the current experiment scope.


## Example: Programmatic Usage

```python
from config import ExperimentConfig
from model import TwoStageSupplyChainModel
from centralsolver import compute_benchmark
from metrics import compute_run_metrics

# Create configuration
cfg = ExperimentConfig(
    lam=20.0,           # Poisson demand rate (default)
    rounds=365,         # Training rounds
    warmup=50,          # Warmup rounds excluded from metrics
    agent_retailer="ucb",
    agent_supplier="greedy",
    init_mode="random", # "random" or "benchmark"
)

# Compute system-optimal benchmark
s1_opt, s2_opt, ctot_opt = compute_benchmark(cfg)
print(f"Benchmark: s1*={s1_opt}, s2*={s2_opt}, c*={ctot_opt:.2f}€")

# Run simulation
model = TwoStageSupplyChainModel(config=cfg, seed=42)
model.run(cfg.rounds)

# Compute metrics (post-warmup)
metrics = compute_run_metrics(model, ctot_opt, cfg.rounds, cfg.warmup)
print("Both converged:", metrics["both_converged"])
print(f"Post-warmup total regret: {metrics['train_total_regret']:.2f}€")
```
