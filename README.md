Link to Repository: https://github.com/jonzz02/serial-supply-chain.git

# Serial Supply Chain Coordination Simulation

A Mesa-based simulation of a 2-stage serial supply chain where bandit agents (retailer stage 1, supplier stage 2) learn discrete base-stock levels. The project studies whether and how agents converge, to what solutions (Nash equilibria, centralized optimum), and which mechanisms (cooperation mode, prior knowledge, initialization, learning algorithms) influence outcomes.

---

## Table of Contents

- [Requirements & Setup](#requirements--setup)
- [How to Run the Code](#how-to-run-the-code)
- [Where Results Are Saved](#where-results-are-saved)
- [How to Reproduce Results](#how-to-reproduce-results)
- [Treatment Variables](#treatment-variables)
- [Output Files & Metrics](#output-files--metrics)
- [Project Structure](#project-structure)
- [Optional: Interactive Visualization](#optional-interactive-visualization)

---

## Requirements & Setup

### Prerequisites

- **Python 3.8 or higher**
- pip (Python package installer)

### Installation

1. **Clone the repository** (if applicable) and enter the project directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

Run a short experiment to confirm everything works:

```bash
python master_experiment.py --n_seeds 5 --rounds 100 --warmup 20 --output_dir results_test
```

You should see progress output and CSV files under `results_test/`.

---

## How to Run the Code

### Master (full factorial) experiment (`master_experiment.py`)

Sweep all treatment dimensions in one run: algorithm pairs, grid size, prior knowledge, init mode, and cooperation mode (including partial with several β values). Uses parallel workers.

```bash
python master_experiment.py --n_seeds 100 --rounds 365 --warmup 50 --max_workers 8 --output_dir results_master
```

| Argument        | Default        | Description                    |
|----------------|----------------|--------------------------------|
| `--n_seeds`    | 100            | Seeds per treatment            |
| `--rounds`     | 365            | Rounds per run                 |
| `--warmup`     | 50             | Warmup rounds                  |
| `--max_workers`| 4              | Parallel processes             |
| `--output_dir` | results_master | Output directory               |
| `--conv_window`| 50             | Convergence detection window   |
| `--conv_threshold` | 0.9        | Convergence stability threshold|

**Design:** 25 algorithm pairs × 3 grid sizes × 2 prior knowledge × 2 init modes × 5 cooperation settings (competitive, cooperative, partial β=0.25/0.5/0.75) → **1,500 treatments**. With 100 seeds, that is 150,000 runs. Runtime depends on hardware (e.g. ~10–30 minutes with 8 workers).

---

## Where Results Are Saved

Results are written to the `--output_dir` (e.g. `results_master/`). The directory contains:

| File / folder   | Description |
|-----------------|-------------|
| `summary.csv`   | One row per treatment; aggregate metrics across seeds |
| `runs.csv`      | One row per (treatment, seed); per-run metrics |
| `benchmarks.csv`| Centralized optimum and Nash counts per treatment |
| `treatments.jsonl` | One JSON object per treatment (full config) |
| `figures/`      | Plots (convergence heatmaps, regret, factor importance, etc.) |
| `metadata.json` | Experiment config and parameter ranges |

---

## How to Reproduce Results

1. **Same environment**  
   Use Python 3.8+ and install exact dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Same commands**  
   Re-run the same script with the same arguments. Example for the master experiment:
   ```bash
   python master_experiment.py --n_seeds 100 --rounds 365 --warmup 50 --max_workers 8 --output_dir results_master
   ```

3. **Seeds**
   Seeds are fixed: `master_experiment.py` uses `seeds = list(range(n_seeds))` (e.g. 0..99 for `--n_seeds 100`). So the same `n_seeds` yields the same sequence of RNG seeds.

4. **Caching**  
   Centralized optimum and Nash equilibrium (and related payoff) computations are cached by config key (e.g. `config.benchmark_key()`, `config.game_key()`). Same config → same cache → same benchmarks across runs.

---

## Treatment Variables

These are the dimensions that define a *treatment* (one configuration of the simulation).

### 1. Algorithm pair (`agent_retailer`, `agent_supplier`)

Each stage uses one of five bandit algorithms:

| Algorithm  | Description |
|-----------|-------------|
| `greedy`  | ε-greedy with decaying exploration |
| `ucb`     | UCB1 with optimism bonus |
| `thompson`| Thompson Sampling (Normal-Inverse-Gamma) |
| `exp3`    | Exponential weights (adversarial) |
| `etc`     | Explore-Then-Commit |

There are 25 pairs (5×5) when using the full grid.

### 2. Action grid (`s_lower`, `s_upper`, `s_step`)

Discrete base-stock levels. Actions are `s_lower, s_lower+s_step, ..., s_upper`.  
Example: `s_lower=0`, `s_upper=40`, `s_step=1` → 41 actions. In the master experiment, “coarse” / “medium” / “fine” map to 41 / 61 / 81 actions.

### 3. Prior knowledge (`prior_knowledge`)

- **`none`** (default): No prior; agents start without demand knowledge.
- **`demand_known`**: Priors are initialized from offline Monte Carlo estimates assuming the demand distribution is known.

### 4. Initialization mode (`init_mode`)

- **`random`** (default): Cold start; zero/uniform-style priors.
- **`benchmark`**: Warm start with pseudo-counts at the centralized optimum (s1*, s2*).

In the master experiment, all combinations of `prior_knowledge` × `init_mode` are run (full factorial).

### 5. Cooperation mode (`cooperation_mode`, `cooperation_beta`)

How rewards are derived from costs H1 (retailer) and H2 (supplier):

| Mode | Retailer reward | Supplier reward |
|------|-----------------|-----------------|
| `competitive` | r1 = −H1 | r2 = −H2 |
| `cooperative` | r1 = −(H1+H2) | r2 = −(H1+H2) |
| `partial`     | r1 = −(H1+β·H2) | r2 = −(H2+β·H1) |

For `partial`, `cooperation_beta` (β) is the weight on the other’s cost (0 = competitive, 1 = full internalization). The master experiment uses β ∈ {0.25, 0.5, 0.75}.

Payoff matrices, Nash equilibria, and prior rewards are computed consistently with the chosen cooperation mode.

---

## Output Files & Metrics

### `summary.csv` (per treatment)

- `converged_to_central_rate`, `converged_to_ne_rate`: Fraction of seeds that converged to central optimum or to a Nash equilibrium.
- `both_convergence_rate`: Fraction of seeds where both agents converged.
- `ne_exists`, `ne_count`: Whether a pure Nash exists and how many.
- `delta1_mean`, `delta2_mean`: Average deviation incentives (≈0 at Nash).
- Treatment identifiers: `agent_retailer`, `agent_supplier`, `init_mode`, `prior_knowledge`, `cooperation_mode`, `cooperation_beta`, grid params.

### `runs.csv` (per treatment × seed)

- `s1_mode`, `s2_mode`: Final converged action pair.
- `converged_to_central`, `converged_to_ne`: Boolean (or missing if no pure NE).
- `delta1`, `delta2`: Deviation incentives for that run.
- `distance_to_central`: L1 distance to centralized optimum.
- `s1_conv_time`, `s2_conv_time`: Round at which each agent converged (if applicable).

### Plots

Generated in `figures/` inside the output directory. Includes convergence heatmaps by algorithm pair, regret heatmaps, convergence by grid size, convergence time, convergence by prior/init, convergence by cooperation mode, solution quality, time distribution, and factor importance.

---

## Project Structure

```
serial-supply-chain/
├── README.md
├── requirements.txt
├── app.py                 # Optional Mesa+Solara visualization
├── master_experiment.py   # Full factorial experiment (parallel)
├── experiment_runner.py   # Grid runner and treatment execution
├── generate_latex_charts.py # LaTeX chart generation for papers
├── simulation/
│   ├── config.py         # ExperimentConfig, defaults
│   ├── environment.py    # Shared supply chain step function
│   ├── model.py          # TwoStageSupplyChainModel (Mesa)
│   └── agents.py         # Bandit agents (greedy, UCB, Thompson, etc.)
└── analysis/
    ├── centralsolver.py  # Centralized optimum, Nash, payoff matrices
    ├── metrics.py        # Convergence and run metrics
    └── plotting.py      # Plot generation for experiment runner
```

---

## Optional: Interactive Visualization

You can run a single simulation in the browser with sliders for parameters:

```bash
solara run app.py
```

Then open the URL shown in the terminal. This uses `app.py` (Mesa 3 + Solara) and the same `simulation` package. It is for exploration only and does not replace the batch experiment above.

---

## Reproducibility Summary

- **RNG:** All randomness flows through Mesa’s `model.rng` (numpy `default_rng`), seeded per run.
- **Benchmarks:** Centralized optimum and Nash (and payoffs) are cached by config key.
- **Config:** Full treatment configs are in `treatments.jsonl`; master runs also write `metadata.json`.
- **Versions:** Use `requirements.txt` and the same Python version for reproducible runs.
