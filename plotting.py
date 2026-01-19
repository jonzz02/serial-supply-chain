import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_convergence_comparison(
    summary_df: pd.DataFrame,
    output_dir: str,
):
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    treatments = summary_df["treatment"].values
    x = np.arange(len(treatments))
    width = 0.35
    
    ax = axes[0]
    ax.bar(x - width/2, summary_df["s1_convergence_rate"], width, label="S1 (Retailer)", color="steelblue")
    ax.bar(x + width/2, summary_df["s2_convergence_rate"], width, label="S2 (Supplier)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Convergence Rate")
    ax.set_title("Convergence Rate by Treatment")
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    ax = axes[1]
    has_s1_time = "s1_conv_time_mean" in summary_df.columns
    has_s2_time = "s2_conv_time_mean" in summary_df.columns
    
    plotted_any = False
    if has_s1_time:
        s1_means = summary_df["s1_conv_time_mean"].values
        s1_stds = summary_df.get("s1_conv_time_std", pd.Series([np.nan]*len(summary_df))).values
        # Only plot if we have at least one non-NaN value
        if not np.all(np.isnan(s1_means)):
            s1_means_plot = np.nan_to_num(s1_means, nan=0.0)
            s1_stds_plot = np.nan_to_num(s1_stds, nan=0.0)
            ax.errorbar(x - 0.15, s1_means_plot, yerr=s1_stds_plot, fmt="o", label="S1", color="steelblue", capsize=3)
            plotted_any = True
    
    if has_s2_time:
        s2_means = summary_df["s2_conv_time_mean"].values
        s2_stds = summary_df.get("s2_conv_time_std", pd.Series([np.nan]*len(summary_df))).values
        # Only plot if we have at least one non-NaN value
        if not np.all(np.isnan(s2_means)):
            s2_means_plot = np.nan_to_num(s2_means, nan=0.0)
            s2_stds_plot = np.nan_to_num(s2_stds, nan=0.0)
            ax.errorbar(x + 0.15, s2_means_plot, yerr=s2_stds_plot, fmt="s", label="S2", color="coral", capsize=3)
            plotted_any = True
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Convergence Time (rounds)")
    ax.set_title("Convergence Time by Treatment")
    if plotted_any:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No convergence data", ha="center", va="center", transform=ax.transAxes)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=150)
    plt.close(fig)


def plot_regret_comparison(
    summary_df: pd.DataFrame,
    output_dir: str,
):
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    treatments = summary_df["treatment"].values
    regret_col = "train_total_regret_mean"
    std_col = "train_total_regret_std"
    
    means = summary_df[regret_col].values if regret_col in summary_df.columns else np.zeros(len(treatments))
    stds = summary_df.get(std_col, pd.Series([0]*len(summary_df))).fillna(0).values
    
    x = np.arange(len(treatments))
    ax.bar(x, means, yerr=stds, capsize=4, color="teal", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Training Total Regret (€)")
    ax.set_title("Training Regret by Treatment")
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regret_comparison.png"), dpi=150)
    plt.close(fig)


def plot_all_learning_curves_comparison(
    timeseries: List[Dict],
    treatments: List,
    output_dir: str,
    warmup: int = 0,
):
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(treatments)))
    
    for treatment, color in zip(treatments, colors):
        name = treatment.name
        ts_data = [t for t in timeseries if t["treatment"] == name]
        if not ts_data:
            continue
        
        costs = np.array([t["total_costs"] for t in ts_data])
        rounds = np.arange(warmup + 1, warmup + costs.shape[1] + 1)
        mean_cost = np.mean(costs, axis=0)
        
        ax.plot(rounds, mean_cost, color=color, label=name)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Total Cost (€)")
    ax.set_title("Learning Curves Comparison")
    ax.legend(loc="upper right", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learning_curves_all.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(
    results: Dict[str, Any],
    output_dir: str = None,
):
    output_dir = output_dir or results.get("output_dir", "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    summary_df = results["summary_df"]
    timeseries = results["timeseries"]
    treatments = results["treatments"]
    warmup = results["base_config"].warmup
    
    print("Generating plots...")
    
    # Comparison plots only
    plot_all_learning_curves_comparison(timeseries, treatments, fig_dir, warmup)
    plot_regret_comparison(summary_df, fig_dir)
    plot_convergence_comparison(summary_df, fig_dir)
    
    print(f"Plots saved to {fig_dir}/")
