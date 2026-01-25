import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


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


def plot_convergence_comparison(summary_df: pd.DataFrame, output_dir: str):
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
        if not np.all(np.isnan(s1_means)):
            s1_means_plot = np.nan_to_num(s1_means, nan=0.0)
            s1_stds_plot = np.nan_to_num(s1_stds, nan=0.0)
            ax.errorbar(x - 0.15, s1_means_plot, yerr=s1_stds_plot, fmt="o", label="S1", color="steelblue", capsize=3)
            plotted_any = True
    
    if has_s2_time:
        s2_means = summary_df["s2_conv_time_mean"].values
        s2_stds = summary_df.get("s2_conv_time_std", pd.Series([np.nan]*len(summary_df))).values
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


def plot_regret_comparison(summary_df: pd.DataFrame, output_dir: str):
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


def plot_all_learning_curves_comparison(timeseries: List[Dict], treatments: List,
                                        output_dir: str, warmup: int = 0):
    set_style()
    
    # Find unique retailer and supplier algorithms from the timeseries
    retailer_algos_set = set()
    supplier_algos_set = set()
    for ts in timeseries:
        if "agent_retailer" in ts:
            retailer_algos_set.add(ts["agent_retailer"])
        if "agent_supplier" in ts:
            supplier_algos_set.add(ts["agent_supplier"])
    
    if not retailer_algos_set or not supplier_algos_set:
        return
    
    retailer_algos = sorted(retailer_algos_set)
    supplier_algos = sorted(supplier_algos_set)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(supplier_algos), 1)))
    
    ncols = min(3, len(retailer_algos))
    nrows = (len(retailer_algos) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, max(ncols, 1), figsize=(6 * ncols, 5 * nrows))
    if len(retailer_algos) == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes).flatten()
    
    for idx, ret_algo in enumerate(retailer_algos):
        ax = axes[idx]
        
        # Collect entries with this retailer algorithm
        ret_entries = [t for t in timeseries if t.get("agent_retailer") == ret_algo]
        
        for sup_idx, sup_algo in enumerate(supplier_algos):
            color = colors[sup_idx % len(colors)]
            # Filter by supplier algorithm
            ts_data = [t for t in ret_entries if t.get("agent_supplier") == sup_algo]
            if not ts_data:
                continue
            
            costs = np.array([t["total_costs"] for t in ts_data])
            rounds = np.arange(warmup + 1, warmup + costs.shape[1] + 1)
            mean_cost = np.mean(costs, axis=0)
            
            ax.plot(rounds, mean_cost, color=color, label=f"vs {sup_algo}", linewidth=1.8)
        
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel("Mean Total Cost (€)", fontsize=10)
        ax.set_title(f"Retailer: {ret_algo.upper()}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(retailer_algos), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle("Learning Curves by Retailer Algorithm", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learning_curves_all.png"), dpi=150)
    plt.close(fig)


def plot_final_action_scatter(run_df: pd.DataFrame, summary_df: pd.DataFrame,
                              nash_results: Dict[str, Any], output_dir: str):
    set_style()
    
    treatments = run_df["treatment"].unique()
    n_treat = len(treatments)
    
    ncols = min(5, n_treat)
    nrows = (n_treat + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if n_treat == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    
    for idx, treatment in enumerate(treatments):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        tdf = run_df[run_df["treatment"] == treatment]
        s1_modes = tdf["s1_mode"].dropna().values
        s2_modes = tdf["s2_mode"].dropna().values
        
        if len(s1_modes) > 0 and len(s2_modes) > 0:
            ax.scatter(s1_modes, s2_modes, alpha=0.5, s=30, c="blue", label="Final modes")
        
        srow = summary_df[summary_df["treatment"] == treatment]
        if len(srow) > 0:
            s1_opt = srow["s1_opt"].values[0]
            s2_opt = srow["s2_opt"].values[0]
            ax.scatter([s1_opt], [s2_opt], marker="*", s=200, c="green", label=f"Central ({s1_opt},{s2_opt})", zorder=5)
        
        full_name = tdf["treatment_full"].iloc[0] if "treatment_full" in tdf.columns else treatment
        if full_name in nash_results:
            ne_set = nash_results[full_name]["ne_set"]
            if ne_set:
                ne_s1, ne_s2 = zip(*ne_set)
                ax.scatter(ne_s1, ne_s2, marker="x", s=150, c="red", label=f"Nash ({len(ne_set)})", zorder=4, linewidths=2)
        
        ax.set_xlabel("S1 (Retailer)")
        ax.set_ylabel("S2 (Supplier)")
        ax.set_title(treatment, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_treat, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    fig.suptitle("Final Action Pairs by Treatment", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "final_action_scatter.png"), dpi=150)
    plt.close(fig)


def plot_algorithm_heatmaps(summary_df: pd.DataFrame, output_dir: str):
    set_style()
    
    algos = ["greedy", "ucb", "thompson", "exp3", "etc"]
    n = len(algos)
    
    metrics_to_plot = [
        ("train_total_regret_mean", "Mean Total Regret", "Reds"),
        ("both_convergence_rate", "Both Converged Rate", "Greens"),
    ]
    
    if "converged_to_ne_rate" in summary_df.columns:
        metrics_to_plot.append(("converged_to_ne_rate", "Converged to NE Rate", "Blues"))
    if "converged_to_central_rate" in summary_df.columns:
        metrics_to_plot.append(("converged_to_central_rate", "Converged to Central Rate", "Purples"))
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for ax, (metric, title, cmap) in zip(axes, metrics_to_plot):
        matrix = np.zeros((n, n))
        matrix[:] = np.nan
        
        for i, ret in enumerate(algos):
            for j, sup in enumerate(algos):
                mask = (summary_df["agent_retailer"] == ret) & (summary_df["agent_supplier"] == sup)
                if mask.any() and metric in summary_df.columns:
                    val = summary_df.loc[mask, metric].values
                    if len(val) > 0:
                        matrix[i, j] = val[0]
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(algos, rotation=45, ha="right")
        ax.set_yticklabels(algos)
        ax.set_xlabel("Supplier")
        ax.set_ylabel("Retailer")
        ax.set_title(title)
        
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    val = matrix[i, j]
                    text = f"{val:.2f}" if val < 100 else f"{val:.0f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                           color="white" if val > matrix[~np.isnan(matrix)].mean() else "black")
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle("Algorithm Performance Heatmaps (5×5 Grid)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "algorithm_heatmaps.png"), dpi=150)
    plt.close(fig)


def plot_best_response_curves(nash_results: Dict[str, Any], output_dir: str,
                              treatment_name: str = None):
    set_style()
    
    if not nash_results:
        return
    
    if treatment_name is None:
        treatment_name = list(nash_results.keys())[0]
    
    if treatment_name not in nash_results:
        return
    
    result = nash_results[treatment_name]
    payoff = result.get("payoff")
    br = result.get("best_responses")
    ne_set = result.get("ne_set", [])
    
    if payoff is None or br is None:
        return
    
    actions = payoff["actions"]
    BR1 = br["BR1"]
    BR2 = br["BR2"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot BR1: for each s2, plot the best s1 responses
    br1_s2 = []
    br1_s1 = []
    for j, s2 in enumerate(actions):
        for i in BR1[j]:
            br1_s2.append(s2)
            br1_s1.append(actions[i])
    if br1_s2:
        ax.scatter(br1_s1, br1_s2, c="blue", s=50, alpha=0.6, label="BR1(s2): Retailer best response")
    
    # Plot BR2: for each s1, plot the best s2 responses
    br2_s1 = []
    br2_s2 = []
    for i, s1 in enumerate(actions):
        for j in BR2[i]:
            br2_s1.append(s1)
            br2_s2.append(actions[j])
    if br2_s1:
        ax.scatter(br2_s1, br2_s2, c="red", s=50, alpha=0.6, marker="s", label="BR2(s1): Supplier best response")
    
    # Mark Nash equilibria
    if ne_set:
        ne_s1, ne_s2 = zip(*ne_set)
        ax.scatter(ne_s1, ne_s2, marker="*", s=300, c="gold", edgecolors="black",
                  linewidths=1.5, label=f"Nash Equilibria ({len(ne_set)})", zorder=10)
    
    ax.set_xlabel("S1 (Retailer base stock)")
    ax.set_ylabel("S2 (Supplier base stock)")
    ax.set_title(f"Best Response Correspondences - {treatment_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "best_response_curves.png"), dpi=150)
    plt.close(fig)


def plot_ne_classification_comparison(summary_df: pd.DataFrame, output_dir: str):
    set_style()
    
    if "converged_to_ne_rate" not in summary_df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    treatments = summary_df["treatment"].values
    x = np.arange(len(treatments))
    width = 0.25
    
    central_rate = summary_df.get("converged_to_central_rate", pd.Series([0]*len(summary_df))).fillna(0).values
    ne_rate = summary_df.get("converged_to_ne_rate", pd.Series([0]*len(summary_df))).fillna(0).values
    both_rate = summary_df.get("both_convergence_rate", pd.Series([0]*len(summary_df))).fillna(0).values
    
    ax.bar(x - width, both_rate, width, label="Both Converged", color="gray", alpha=0.7)
    ax.bar(x, ne_rate, width, label="Converged to NE", color="blue", alpha=0.7)
    ax.bar(x + width, central_rate, width, label="Converged to Central", color="green", alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Outcome Classification by Treatment")
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ne_classification_comparison.png"), dpi=150)
    plt.close(fig)


def plot_deviation_incentives(summary_df: pd.DataFrame, output_dir: str):
    set_style()
    
    if "delta1_mean" not in summary_df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    treatments = summary_df["treatment"].values
    x = np.arange(len(treatments))
    width = 0.35
    
    d1_mean = summary_df.get("delta1_mean", pd.Series([0]*len(summary_df))).fillna(0).values
    d2_mean = summary_df.get("delta2_mean", pd.Series([0]*len(summary_df))).fillna(0).values
    d1_std = summary_df.get("delta1_std", pd.Series([0]*len(summary_df))).fillna(0).values
    d2_std = summary_df.get("delta2_std", pd.Series([0]*len(summary_df))).fillna(0).values
    
    ax.bar(x - width/2, d1_mean, width, yerr=d1_std, label="δ1 (Retailer)", color="steelblue", capsize=3)
    ax.bar(x + width/2, d2_mean, width, yerr=d2_std, label="δ2 (Supplier)", color="coral", capsize=3)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=45, ha="right")
    ax.set_ylabel("Deviation Incentive (cost units)")
    ax.set_title("Deviation Incentives by Treatment (δ≈0 indicates Nash stability)")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "deviation_incentives.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(results: Dict[str, Any], output_dir: str = None):
    output_dir = output_dir or results.get("output_dir", "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    summary_df = results["summary_df"]
    run_df = results.get("run_df")
    timeseries = results["timeseries"]
    treatments = results["treatments"]
    warmup = results["base_config"].warmup
    nash_results = results.get("nash_results", {})
    
    print("Generating plots...")
    
    plot_all_learning_curves_comparison(timeseries, treatments, fig_dir, warmup)
    plot_regret_comparison(summary_df, fig_dir)
    plot_convergence_comparison(summary_df, fig_dir)
    
    # New plots
    if run_df is not None and not run_df.empty:
        plot_final_action_scatter(run_df, summary_df, nash_results, fig_dir)
    
    plot_algorithm_heatmaps(summary_df, fig_dir)
    
    if nash_results:
        plot_best_response_curves(nash_results, fig_dir)
    
    plot_ne_classification_comparison(summary_df, fig_dir)
    plot_deviation_incentives(summary_df, fig_dir)
    
    print(f"Plots saved to {fig_dir}/")
