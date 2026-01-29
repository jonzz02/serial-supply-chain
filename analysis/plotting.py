import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

MAX_TREATMENTS_TO_PLOT = 40


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


def _filter_treatments_for_plot(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to top/bottom treatments if too many for readable plots."""
    if len(summary_df) <= MAX_TREATMENTS_TO_PLOT:
        return summary_df
    half = MAX_TREATMENTS_TO_PLOT // 2
    sorted_df = summary_df.sort_values("both_convergence_rate", ascending=False)
    return pd.concat([sorted_df.head(half), sorted_df.tail(half)], axis=0)


def plot_convergence_comparison(summary_df: pd.DataFrame, output_dir: str):
    """Q1+Q2: Do agents converge? How fast?"""
    set_style()
    
    n_show = 50
    sorted_df = summary_df.sort_values("both_convergence_rate", ascending=False)
    top_df = sorted_df.head(n_show)
    bottom_df = sorted_df.tail(n_show)
    
    treatment_col = "treatment_full" if "treatment_full" in summary_df.columns else "treatment"
    
    fig, axes = plt.subplots(4, 1, figsize=(24, 20))
    width = 0.25
    
    ax = axes[0]
    treatments = top_df[treatment_col].values
    x = np.arange(len(treatments))
    
    ax.bar(x - width, top_df["s1_convergence_rate"], width, label="S1 (Retailer)", color="steelblue")
    ax.bar(x, top_df["s2_convergence_rate"], width, label="S2 (Supplier)", color="coral")
    if "pair_convergence_rate" in top_df.columns:
        ax.bar(x + width, top_df["pair_convergence_rate"], width, label="Pair (S1,S2)", color="forestgreen")
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Convergence Rate", fontsize=10)
    ax.set_title(f"Top {n_show} Treatments by Convergence Rate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1]
    treatments = bottom_df[treatment_col].values
    x = np.arange(len(treatments))
    
    ax.bar(x - width, bottom_df["s1_convergence_rate"], width, label="S1 (Retailer)", color="steelblue")
    ax.bar(x, bottom_df["s2_convergence_rate"], width, label="S2 (Supplier)", color="coral")
    if "pair_convergence_rate" in bottom_df.columns:
        ax.bar(x + width, bottom_df["pair_convergence_rate"], width, label="Pair (S1,S2)", color="forestgreen")
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Convergence Rate", fontsize=10)
    ax.set_title(f"Bottom {n_show} Treatments by Convergence Rate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    
    ax = axes[2]
    treatments = top_df[treatment_col].values
    x = np.arange(len(treatments))
    offset = 0.2
    
    plotted_any = False
    if "s1_conv_time_mean" in top_df.columns:
        s1_means = top_df["s1_conv_time_mean"].values
        s1_stds = top_df.get("s1_conv_time_std", pd.Series([np.nan]*len(top_df))).values
        mask = np.isfinite(s1_means)
        if mask.any():
            ax.errorbar(x[mask] - offset, s1_means[mask], yerr=s1_stds[mask], 
                       fmt="o", label="S1", color="steelblue", capsize=3, markersize=4)
            plotted_any = True
    
    if "s2_conv_time_mean" in top_df.columns:
        s2_means = top_df["s2_conv_time_mean"].values
        s2_stds = top_df.get("s2_conv_time_std", pd.Series([np.nan]*len(top_df))).values
        mask = np.isfinite(s2_means)
        if mask.any():
            ax.errorbar(x[mask], s2_means[mask], yerr=s2_stds[mask], 
                       fmt="s", label="S2", color="coral", capsize=3, markersize=4)
            plotted_any = True
    
    if "pair_conv_time_mean" in top_df.columns:
        pair_means = top_df["pair_conv_time_mean"].values
        pair_stds = top_df.get("pair_conv_time_std", pd.Series([np.nan]*len(top_df))).values
        mask = np.isfinite(pair_means)
        if mask.any():
            ax.errorbar(x[mask] + offset, pair_means[mask], yerr=pair_stds[mask], 
                       fmt="^", label="Pair", color="forestgreen", capsize=3, markersize=4)
            plotted_any = True
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Convergence Time (rounds)", fontsize=10)
    ax.set_title(f"Top {n_show} Treatments - Convergence Time", fontsize=12, fontweight="bold")
    if plotted_any:
        ax.legend(fontsize=9)
    
    ax = axes[3]
    treatments = bottom_df[treatment_col].values
    x = np.arange(len(treatments))
    
    plotted_any = False
    if "s1_conv_time_mean" in bottom_df.columns:
        s1_means = bottom_df["s1_conv_time_mean"].values
        s1_stds = bottom_df.get("s1_conv_time_std", pd.Series([np.nan]*len(bottom_df))).values
        mask = np.isfinite(s1_means)
        if mask.any():
            ax.errorbar(x[mask] - offset, s1_means[mask], yerr=s1_stds[mask], 
                       fmt="o", label="S1", color="steelblue", capsize=3, markersize=4)
            plotted_any = True
    
    if "s2_conv_time_mean" in bottom_df.columns:
        s2_means = bottom_df["s2_conv_time_mean"].values
        s2_stds = bottom_df.get("s2_conv_time_std", pd.Series([np.nan]*len(bottom_df))).values
        mask = np.isfinite(s2_means)
        if mask.any():
            ax.errorbar(x[mask], s2_means[mask], yerr=s2_stds[mask], 
                       fmt="s", label="S2", color="coral", capsize=3, markersize=4)
            plotted_any = True
    
    if "pair_conv_time_mean" in bottom_df.columns:
        pair_means = bottom_df["pair_conv_time_mean"].values
        pair_stds = bottom_df.get("pair_conv_time_std", pd.Series([np.nan]*len(bottom_df))).values
        mask = np.isfinite(pair_means)
        if mask.any():
            ax.errorbar(x[mask] + offset, pair_means[mask], yerr=pair_stds[mask], 
                       fmt="^", label="Pair", color="forestgreen", capsize=3, markersize=4)
            plotted_any = True
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Convergence Time (rounds)", fontsize=10)
    ax.set_title(f"Bottom {n_show} Treatments - Convergence Time", fontsize=12, fontweight="bold")
    if plotted_any:
        ax.legend(fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_comparison.png"), dpi=150)
    plt.close(fig)


def plot_ne_classification_comparison(summary_df: pd.DataFrame, output_dir: str):
    """Q3: To what solutions do they converge? (Central vs Nash)"""
    set_style()
    
    n_show = 50
    sorted_df = summary_df.sort_values("both_convergence_rate", ascending=False)
    top_df = sorted_df.head(n_show)
    bottom_df = sorted_df.tail(n_show)
    
    treatment_col = "treatment_full" if "treatment_full" in summary_df.columns else "treatment"
    
    fig, axes = plt.subplots(2, 1, figsize=(24, 10))
    width = 0.25
    
    ax = axes[0]
    treatments = top_df[treatment_col].values
    x = np.arange(len(treatments))
    
    central_rate = top_df.get("converged_to_central_rate", pd.Series([0]*len(top_df))).fillna(0).values
    both_rate = top_df.get("both_convergence_rate", pd.Series([0]*len(top_df))).fillna(0).values
    
    ax.bar(x - width, both_rate, width, label="Both Converged", color="gray", alpha=0.7)
    ax.bar(x + width, central_rate, width, label="Converged to Central", color="green", alpha=0.7)
    
    if "converged_to_ne_rate" in top_df.columns:
        ne_rate = top_df["converged_to_ne_rate"].fillna(0).values
        if not np.all(ne_rate == 0):
            ax.bar(x, ne_rate, width, label="Converged to NE", color="blue", alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(f"Top {n_show} Treatments - Outcome Classification", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    
    ax = axes[1]
    treatments = bottom_df[treatment_col].values
    x = np.arange(len(treatments))
    
    central_rate = bottom_df.get("converged_to_central_rate", pd.Series([0]*len(bottom_df))).fillna(0).values
    both_rate = bottom_df.get("both_convergence_rate", pd.Series([0]*len(bottom_df))).fillna(0).values
    
    ax.bar(x - width, both_rate, width, label="Both Converged", color="gray", alpha=0.7)
    ax.bar(x + width, central_rate, width, label="Converged to Central", color="green", alpha=0.7)
    
    if "converged_to_ne_rate" in bottom_df.columns:
        ne_rate = bottom_df["converged_to_ne_rate"].fillna(0).values
        if not np.all(ne_rate == 0):
            ax.bar(x, ne_rate, width, label="Converged to NE", color="blue", alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(treatments, rotation=90, ha="right", fontsize=6)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(f"Bottom {n_show} Treatments - Outcome Classification", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ne_classification_comparison.png"), dpi=150)
    plt.close(fig)


def plot_mechanism_summary(summary_df: pd.DataFrame, output_dir: str):
    """Q4: Which mechanisms influence convergence?"""
    set_style()
    
    mechanism_cols = []
    
    if "cooperation_mode" in summary_df.columns and summary_df["cooperation_mode"].nunique() > 1:
        mechanism_cols.append(("cooperation_mode", "Cooperation Mode"))
    
    if "prior_knowledge" in summary_df.columns and summary_df["prior_knowledge"].nunique() > 1:
        mechanism_cols.append(("prior_knowledge", "Prior Knowledge"))
    
    if "init_mode" in summary_df.columns and summary_df["init_mode"].nunique() > 1:
        mechanism_cols.append(("init_mode", "Initialization"))
    
    if "s_upper" in summary_df.columns and summary_df["s_upper"].nunique() > 1:
        summary_df = summary_df.copy()
        summary_df["grid_size"] = summary_df["s_upper"].map(
            lambda x: "coarse" if x <= 40 else ("medium" if x <= 60 else "fine")
        )
        mechanism_cols.append(("grid_size", "Grid Size"))
    
    if "agent_retailer" in summary_df.columns and "agent_supplier" in summary_df.columns:
        summary_df = summary_df.copy() if "grid_size" not in summary_df.columns else summary_df
        summary_df["algo_pair"] = summary_df["agent_retailer"] + "/" + summary_df["agent_supplier"]
        if summary_df["algo_pair"].nunique() > 1:
            mechanism_cols.append(("algo_pair", "Algorithm Pair"))
    
    if not mechanism_cols:
        return
    
    n_mechs = len(mechanism_cols)
    ncols = min(3, n_mechs)
    nrows = (n_mechs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_mechs == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (col, title) in enumerate(mechanism_cols):
        ax = axes[idx]
        
        agg_dict = {"both_convergence_rate": "mean"}
        has_pair = "pair_convergence_rate" in summary_df.columns
        if has_pair:
            agg_dict["pair_convergence_rate"] = "mean"
        
        grouped = summary_df.groupby(col).agg(agg_dict).reset_index()
        
        categories = grouped[col].values
        x = np.arange(len(categories))
        width = 0.35
        
        if has_pair:
            ax.bar(x - width/2, grouped["both_convergence_rate"], width, label="Both Conv.", color="steelblue")
            ax.bar(x + width/2, grouped["pair_convergence_rate"], width, label="Pair Conv.", color="forestgreen")
        else:
            ax.bar(x, grouped["both_convergence_rate"], width, label="Both Conv.", color="steelblue")
        
        ax.set_xticks(x)
        fontsize = 6 if col == "algo_pair" else 9
        ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=fontsize)
        ax.set_ylabel("Convergence Rate")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    
    for idx in range(n_mechs, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle("Mechanism Effects on Convergence", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mechanism_effects.png"), dpi=150)
    plt.close(fig)


def plot_final_action_scatter(run_df: pd.DataFrame, summary_df: pd.DataFrame,
                              nash_results: Dict[str, Any], output_dir: str, top_k: int = 30):
    """Q3: Where did they land? Shows top K and bottom K treatments by convergence rate."""
    set_style()
    
    sorted_summary = summary_df.sort_values("both_convergence_rate", ascending=False)
    
    treatment_col = "treatment_full" if "treatment_full" in sorted_summary.columns else "treatment"
    top_treatments = sorted_summary.head(top_k)[treatment_col].values
    bottom_treatments = sorted_summary.tail(top_k)[treatment_col].values
    
    if len(top_treatments) == 0:
        return
    
    ncols = 6
    nrows_top = (len(top_treatments) + ncols - 1) // ncols
    nrows_bottom = (len(bottom_treatments) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows_top + nrows_bottom, ncols, figsize=(3*ncols, 3*(nrows_top + nrows_bottom)))
    axes = np.atleast_2d(axes)
    
    def plot_treatment(ax, treatment):
        if "treatment_full" in run_df.columns:
            tdf = run_df[run_df["treatment_full"] == treatment]
        else:
            tdf = run_df[run_df["treatment"] == treatment]
        s1_modes = tdf["s1_mode"].dropna().values
        s2_modes = tdf["s2_mode"].dropna().values
        
        if len(s1_modes) > 0 and len(s2_modes) > 0:
            ax.scatter(s1_modes, s2_modes, alpha=0.3, s=8, c="blue", label="Final modes")
        
        if "treatment_full" in summary_df.columns:
            srow = summary_df[summary_df["treatment_full"] == treatment]
        else:
            srow = summary_df[summary_df["treatment"] == treatment]
        if len(srow) > 0:
            s1_opt = srow["s1_opt"].values[0]
            s2_opt = srow["s2_opt"].values[0]
            ax.scatter([s1_opt], [s2_opt], marker="*", s=100, c="green", label=f"Central", zorder=5)
        
        if "treatment_full" in tdf.columns and len(tdf) > 0:
            full_name = tdf["treatment_full"].iloc[0]
        else:
            full_name = treatment
        
        if full_name in nash_results:
            ne_set = nash_results[full_name]["ne_set"]
            if ne_set:
                ne_s1, ne_s2 = zip(*ne_set)
                ax.scatter(ne_s1, ne_s2, marker="x", s=60, c="red", label=f"Nash", zorder=4, linewidths=1.2)
        
        conv_rate = srow["both_convergence_rate"].values[0] if len(srow) > 0 else 0
        
        display_name = treatment
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."
        
        ax.set_xlabel("S1", fontsize=8)
        ax.set_ylabel("S2", fontsize=8)
        ax.set_title(f"{display_name}\n(conv={conv_rate:.0%})", fontsize=6.5)
        ax.legend(fontsize=5, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    for idx, treatment in enumerate(top_treatments):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        plot_treatment(ax, treatment)
    
    for idx in range(len(top_treatments), nrows_top * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    for idx, treatment in enumerate(bottom_treatments):
        row, col = idx // ncols, idx % ncols
        ax = axes[nrows_top + row, col]
        plot_treatment(ax, treatment)
    
    for idx in range(len(bottom_treatments), nrows_bottom * ncols):
        row, col = idx // ncols, idx % ncols
        axes[nrows_top + row, col].axis('off')
    
    fig.suptitle(f"Final Action Pairs - Top {top_k} and Bottom {top_k} by Convergence", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Leave more space for suptitle
    fig.savefig(os.path.join(output_dir, "final_action_scatter.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(results: Dict[str, Any], output_dir: str = None):
    """Generate all analysis plots (reduced set answering the 4 key questions)."""
    output_dir = output_dir or results.get("output_dir", "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    summary_df = results["summary_df"]
    run_df = results.get("run_df")
    nash_results = results.get("nash_results", {})
    
    print("Generating plots...")
    
    # Q1+Q2: Convergence overview
    plot_convergence_comparison(summary_df, fig_dir)
    
    # Q3: Solution outcome rates
    plot_ne_classification_comparison(summary_df, fig_dir)
    
    # Q4: Mechanism effects
    plot_mechanism_summary(summary_df, fig_dir)
    
    # Q3: Where did they land?
    if run_df is not None and not run_df.empty:
        plot_final_action_scatter(run_df, summary_df, nash_results, fig_dir)
    
    print(f"Plots saved to {fig_dir}/")
