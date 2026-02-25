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

Run a short scenario to confirm everything works:

```bash
python run_experiments.py --scenario baseline --s_upper 60 --s_step 5 --n_seeds 5 --rounds 100 --warmup 20 --output_dir results/
```

You should see progress output and CSV files under `results/`.

---

## How to Run the Code

There are three main entry points.

### 1. Scenario-based experiments (`run_experiments.py`)

Run predefined scenarios (baseline, demand-known prior, benchmark init) with a chosen action grid and seed count. Use this for quick or scenario-focused runs.

```bash
# Baseline: random init, no prior (full 5×5 algorithm grid)
python run_experiments.py --scenario baseline --s_upper 60 --s_step 1 --full_grid --n_seeds 50 --rounds 365 --warmup 50 --output_dir results/

# Demand-known prior (coarser grid for faster runs)
python run_experiments.py --scenario demand_known_prior --s_upper 60 --s_step 5 --n_seeds 30 --rounds 365 --warmup 50 --output_dir results/

# Benchmark init: warm start at centralized optimum
python run_experiments.py --scenario benchmark_init --s_upper 60 --s_step 1 --full_grid --n_seeds 30 --rounds 365 --warmup 50 --output_dir results/
```

| Argument       | Description                                      |
|----------------|--------------------------------------------------|
| `--scenario`   | `baseline`, `demand_known_prior`, or `benchmark_init` |
| `--s_lower`    | Action space lower bound (default: 0)            |
| `--s_upper`    | Action space upper bound (required)               |
| `--s_step`     | Action grid step (default: 1)                     |
| `--full_grid`  | Use all 5×5 algorithm pairs; otherwise subset    |
| `--n_seeds`    | Number of random seeds per treatment              |
| `--rounds`     | Simulation rounds per run                         |
| `--warmup`     | Warmup rounds before convergence metrics          |
| `--output_dir` | Directory for all outputs (required)              |

Outputs are written under `output_dir/<scenario>/s<s_lower>-<s_upper>-<s_step>/full` or `subset`, and include `summary.csv`, `runs.csv`, `benchmarks.csv`, `treatments.jsonl`, and `figures/`.

### 2. Master (full factorial) experiment (`master_experiment.py`)

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

### 3. Analysis of results (`analyze_results.py`)

After running experiments (especially the master experiment), generate reports, plots, and summary statistics from the saved CSVs.

```bash
python analyze_results.py --results_dir results_master --output_dir analysis_output
```

| Argument        | Default          | Description                    |
|----------------|------------------|--------------------------------|
| `--results_dir`| results_master   | Directory containing experiment outputs |
| `--output_dir` | analysis_output  | Where to write analysis outputs |

This produces convergence overviews, speed and solution-quality plots, mechanism-effect plots, algorithm deep-dive, summary dashboard, and a text report (`detailed_analysis_report.txt`).

---

## Where Results Are Saved

- **`run_experiments.py`**  
  - Path: `<output_dir>/<scenario>/s<s_lower>-<s_upper>-<s_step>/full` or `subset`  
  - Example: `results/baseline/s0-60-1/full/`

- **`master_experiment.py`**  
  - Path: whatever you pass as `--output_dir` (e.g. `results_master/`).

In both cases the directory contains:

| File / folder   | Description |
|-----------------|-------------|
| `summary.csv`   | One row per treatment; aggregate metrics across seeds |
| `runs.csv`      | One row per (treatment, seed); per-run metrics |
| `benchmarks.csv`| Centralized optimum and Nash counts per treatment |
| `treatments.jsonl` | One JSON object per treatment (full config) |
| `figures/`      | Plots from the experiment runner (e.g. learning curves, heatmaps) |
| `metadata.json` | (Master only) Experiment config and parameter ranges |

Analysis outputs from `analyze_results.py` go to `--output_dir` (e.g. `analysis_output/`) and include PNGs and `detailed_analysis_report.txt`.

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
   Seeds are fixed: `run_experiments.py` and `master_experiment.py` use `seeds = list(range(n_seeds))` (e.g. 0..99 for `--n_seeds 100`). So the same `n_seeds` yields the same sequence of RNG seeds.

4. **Caching**  
   Centralized optimum and Nash equilibrium (and related payoff) computations are cached by config key (e.g. `config.benchmark_key()`, `config.game_key()`). Same config → same cache → same benchmarks across runs.

5. **Reproducing analysis**  
   After reproducing the experiment outputs, run:
   ```bash
   python analyze_results.py --results_dir results_master --output_dir analysis_output
   ```
   Same inputs produce the same analysis outputs.

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

### Plots (from experiment runner)

In `figures/`: e.g. learning curves, convergence comparison, regret, final-action scatter, algorithm heatmaps, best-response curves, deviation incentives. Exact set depends on the runner.

### Plots from `analyze_results.py`

In the analysis `--output_dir`: convergence overview, convergence speed, by grid size, outcome by prior/init, solution quality, mechanism effects, algorithm deep-dive, summary dashboard.

---

## Project Structure

```
serial-supply-chain/
├── README.md
├── requirements.txt
├── app.py                 # Optional Mesa+Solara visualization
├── run_experiments.py     # Scenario-based experiments
├── master_experiment.py   # Full factorial experiment (parallel)
├── experiment_runner.py   # Grid runner and treatment execution
├── analyze_results.py     # Post-hoc analysis and reporting
├── simulation/
│   ├── config.py         # ExperimentConfig, defaults
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
mesa runserver
```

Then open the Solara-based UI (URL shown in the terminal). This uses `app.py` and the same `simulation` package. It is for exploration only and does not replace the batch experiments above.

---

## Reproducibility Summary

- **RNG:** Demand, Thompson Sampling, and Mesa’s random are all seeded; seeds are deterministic from the seed index.
- **Benchmarks:** Centralized optimum and Nash (and payoffs) are cached by config key.
- **Config:** Full treatment configs are in `treatments.jsonl`; master runs also write `metadata.json`.
- **Versions:** Use `requirements.txt` and the same Python version for reproducible runs.
