import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'primary': '#0065BD',
    'secondary': '#64A0C8',
    'tertiary': '#A2AD00',
    'quaternary': '#E37222',
}

ALGO_COLORS = {
    'greedy': '#2E86AB',
    'ucb': '#A23B72',
    'thompson': '#F18F01',
    'exp3': '#C73E1D',
    'etc': '#4ECDC4',
}


def load_data(results_dir='results_full_factorial'):
    runs = pd.read_csv(os.path.join(results_dir, 'runs.csv'))

    runs['agent_retailer'] = runs['treatment'].str.split('_').str[0]
    runs['agent_supplier'] = runs['treatment'].str.split('_').str[1]
    runs['algo_pair'] = runs['agent_retailer'] + '/' + runs['agent_supplier']

    def extract_s_upper(treatment_name):
        try:
            parts = treatment_name.split('_')
            s_part = [p for p in parts if p.startswith('s')][0]
            return int(s_part.split('-')[1])
        except Exception:
            return None

    runs['s_upper'] = runs['treatment'].apply(extract_s_upper)
    runs['grid_size'] = runs['s_upper'].map(
        lambda x: 'coarse' if x <= 40 else ('medium' if x <= 60 else 'fine') if x is not None else None
    )

    return runs


def _savefig(output_dir, name):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300)
    plt.close()


def _clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


def chart_time_distribution(runs, output_dir):
    converged = runs[runs['both_converged'] == True].copy()
    if len(converged) == 0:
        print("No converged runs for time distribution")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    s1_times = converged['s1_conv_time'].dropna()
    s2_times = converged['s2_conv_time'].dropna()

    if len(s1_times) > 0:
        ax.hist(s1_times, bins=30, alpha=0.7, label=f'Retailer (μ={s1_times.mean():.0f})',
                color=COLORS['tertiary'], edgecolor='white', linewidth=0.5)
    if len(s2_times) > 0:
        ax.hist(s2_times, bins=30, alpha=0.7, label=f'Supplier (μ={s2_times.mean():.0f})',
                color=COLORS['primary'], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Convergence Time (rounds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Convergence Time Distribution', fontweight='bold')
    ax.legend(frameon=False)
    _clean_spines(ax)

    _savefig(output_dir, 'time_distribution')
    print("✓ Time Distribution saved")


def chart_solution_quality(runs, output_dir):
    converged = runs[runs['both_converged'] == True].copy()
    if len(converged) == 0:
        print("No converged runs for solution quality")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    conv_rate = runs['both_converged'].mean()
    central_rate = converged['converged_to_central'].mean() if 'converged_to_central' in converged.columns else 0
    ne_rate = converged['converged_to_ne'].mean() if 'converged_to_ne' in converged.columns else 0

    labels = ['Converged', 'To Central\nOptimum', 'To Nash\nEquilibrium']
    rates = [conv_rate, central_rate, ne_rate]
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['secondary']]

    bars = ax.bar(labels, rates, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Rate')
    ax.set_title('Solution Quality', fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.15)
    _clean_spines(ax)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    _savefig(output_dir, 'solution_quality')
    print("✓ Solution Quality saved")


def chart_convergence_by_algo_pair(runs, output_dir):
    algos = ['greedy', 'ucb', 'thompson', 'exp3', 'etc']
    algo_matrix = np.zeros((len(algos), len(algos)))

    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            mask = (runs['agent_retailer'] == a1) & (runs['agent_supplier'] == a2)
            if mask.sum() > 0:
                algo_matrix[i, j] = runs.loc[mask, 'both_converged'].mean()

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(algo_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(algos)))
    ax.set_yticks(np.arange(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_yticklabels([a.upper() for a in algos])
    ax.set_xlabel('Supplier Algorithm')
    ax.set_ylabel('Retailer Algorithm')
    ax.set_title('Convergence Rate by Algorithm Pair', fontweight='bold')

    for i in range(len(algos)):
        for j in range(len(algos)):
            ax.text(j, i, f'{algo_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if algo_matrix[i, j] < 0.5 else 'black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Convergence Rate')

    _savefig(output_dir, 'convergence_by_algo_pair')
    print("✓ Convergence by Algorithm Pair saved")


def chart_regret_by_algo_pair(runs, output_dir):
    algos = ['greedy', 'ucb', 'thompson', 'exp3', 'etc']
    regret_matrix = np.zeros((len(algos), len(algos)))

    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            mask = (runs['agent_retailer'] == a1) & (runs['agent_supplier'] == a2)
            if mask.sum() > 0:
                regret_matrix[i, j] = runs.loc[mask, 'train_total_regret'].mean()

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(regret_matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(np.arange(len(algos)))
    ax.set_yticks(np.arange(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_yticklabels([a.upper() for a in algos])
    ax.set_xlabel('Supplier Algorithm')
    ax.set_ylabel('Retailer Algorithm')
    ax.set_title('Average Total Regret by Algorithm Pair', fontweight='bold')

    for i in range(len(algos)):
        for j in range(len(algos)):
            ax.text(j, i, f'{regret_matrix[i, j] / 1000:.1f}k',
                    ha='center', va='center', fontsize=8,
                    color='white' if regret_matrix[i, j] > regret_matrix.mean() else 'black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Regret')

    _savefig(output_dir, 'regret_by_algo_pair')
    print("✓ Regret by Algorithm Pair saved")


def chart_convergence_by_grid_size(runs, output_dir):
    grid_data = runs.groupby('grid_size').agg({
        's1_converged': 'mean',
        's2_converged': 'mean',
        'both_converged': 'mean',
    })

    grid_order = ['coarse', 'medium', 'fine']
    grid_data = grid_data.reindex(grid_order)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(grid_order))
    width = 0.25

    bars1 = ax.bar(x - width, grid_data['s1_converged'], width, label='Retailer',
                   color=COLORS['primary'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, grid_data['s2_converged'], width, label='Supplier',
                   color=COLORS['secondary'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, grid_data['both_converged'], width, label='Both',
                   color=COLORS['tertiary'], edgecolor='white', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in grid_order])
    ax.set_ylabel('Convergence Rate')
    ax.set_xlabel('Grid Size')
    ax.set_title('Convergence Rate by Grid Size', fontweight='bold')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    _clean_spines(ax)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=8)

    _savefig(output_dir, 'convergence_by_grid_size')
    print("✓ Convergence by Grid Size saved")


def chart_convergence_time_by_grid_size(runs, output_dir):
    converged = runs[runs['both_converged'] == True].copy()
    if len(converged) == 0:
        print("No converged runs for time by grid size")
        return

    grid_order = ['coarse', 'medium', 'fine']

    grid_stats = converged.groupby('grid_size').agg({
        's1_conv_time': ['mean', 'std', 'min', 'max'],
        's2_conv_time': ['mean', 'std', 'min', 'max'],
    })
    grid_stats = grid_stats.reindex(grid_order)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(grid_order))
    width = 0.35

    s1_means = grid_stats['s1_conv_time']['mean']
    s2_means = grid_stats['s2_conv_time']['mean']

    bars1 = ax.bar(x - width / 2, s1_means, width, label='Retailer',
                   color=COLORS['primary'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width / 2, s2_means, width, label='Supplier',
                   color=COLORS['secondary'], edgecolor='white', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in grid_order], fontsize=9)
    ax.set_ylabel('Convergence Time (rounds)')
    ax.set_xlabel('Grid Size')
    ax.set_title('Convergence Time by Grid Size (Both Converged)', fontweight='bold')
    ax.legend(frameon=False)
    _clean_spines(ax)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)

    _savefig(output_dir, 'convergence_time_by_grid_size')
    print("✓ Convergence Time by Grid Size saved")


def chart_both_convergence_by_algo_grid(runs, output_dir):
    algo_conv = runs.groupby('algo_pair')['both_converged'].mean()
    top_algos = algo_conv[algo_conv > 0].nlargest(9).index

    data = runs[runs['algo_pair'].isin(top_algos)]

    pivot = data.groupby(['algo_pair', 'grid_size'])['both_converged'].mean().unstack()
    pivot = pivot.reindex(top_algos)

    grid_order = ['coarse', 'medium', 'fine']
    pivot = pivot[grid_order]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(pivot.index))
    width = 0.25

    ax.bar(x - width, pivot['coarse'], width, label='Coarse',
           color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    ax.bar(x, pivot['medium'], width, label='Medium',
           color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax.bar(x + width, pivot['fine'], width, label='Fine',
           color=COLORS['tertiary'], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Convergence Rate')
    ax.set_xlabel('Algorithm Pair')
    ax.set_title('Both Agent Convergence by Algorithm Pair × Grid Size',
                 fontweight='bold', fontsize=11)
    ax.legend(frameon=False, title='Grid Size')
    ax.set_ylim(0, 1.05)
    _clean_spines(ax)

    _savefig(output_dir, 'both_convergence_by_algo_grid')
    print("✓ Both Convergence by Algo × Grid saved (Top 9 algorithm pairs)")


def chart_convergence_by_prior_init(runs, output_dir):
    if 'prior_knowledge' not in runs.columns or 'init_mode' not in runs.columns:
        print("Missing prior_knowledge or init_mode columns")
        return

    data = runs.groupby(['prior_knowledge', 'init_mode']).agg({
        'both_converged': 'mean',
        'converged_to_ne': lambda x: x.sum() / len(x) if len(x) > 0 else 0,
        'converged_to_central': lambda x: x.sum() / len(x) if len(x) > 0 else 0,
    }).reset_index()

    data['to_central'] = data['converged_to_central']
    data['to_nash_only'] = np.maximum(0, data['converged_to_ne'] - data['converged_to_central'])
    data['other_converged'] = np.maximum(0, data['both_converged'] - data['converged_to_ne'])
    data['not_converged'] = 1 - data['both_converged']

    priors = sorted(data['prior_knowledge'].unique())
    init_modes = sorted(data['init_mode'].unique())

    fig, ax = plt.subplots(figsize=(8, 8))

    x_positions = []
    x_labels = []

    pos = 0
    for prior in priors:
        for init_mode in init_modes:
            x_positions.append(pos)
            prior_label = prior.replace('_', '\n')
            init_label = init_mode.replace('_', '\n')
            x_labels.append(f"{prior_label}\n{init_label}")
            pos += 1
        pos += 0.5

    to_central_vals = []
    to_nash_only_vals = []
    other_conv_vals = []
    not_conv_vals = []

    for prior in priors:
        for init_mode in init_modes:
            subset = data[(data['prior_knowledge'] == prior) & (data['init_mode'] == init_mode)]
            if len(subset) > 0:
                to_central_vals.append(subset['to_central'].iloc[0])
                to_nash_only_vals.append(subset['to_nash_only'].iloc[0])
                other_conv_vals.append(subset['other_converged'].iloc[0])
                not_conv_vals.append(subset['not_converged'].iloc[0])
            else:
                to_central_vals.append(0)
                to_nash_only_vals.append(0)
                other_conv_vals.append(0)
                not_conv_vals.append(1)

    width = 0.7

    ax.bar(x_positions, to_central_vals, width, label='To Central Optimum',
           color=COLORS['tertiary'], edgecolor='white', linewidth=0.8)

    ax.bar(x_positions, to_nash_only_vals, width, bottom=to_central_vals,
           label='To Nash (not Central)', color=COLORS['secondary'],
           edgecolor='white', linewidth=0.8)

    bottom2 = [tc + tn for tc, tn in zip(to_central_vals, to_nash_only_vals)]
    ax.bar(x_positions, other_conv_vals, width, bottom=bottom2,
           label='Other Converged', color=COLORS['primary'],
           edgecolor='white', linewidth=0.8)

    bottom3 = [b + oc for b, oc in zip(bottom2, other_conv_vals)]
    ax.bar(x_positions, not_conv_vals, width, bottom=bottom3,
           label='Not Converged', color='#CCCCCC', edgecolor='white', linewidth=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_xlabel('Prior Knowledge × Initialization Mode', fontsize=11)
    ax.set_title('Convergence Outcomes by Prior Knowledge × Init Mode', fontweight='bold', fontsize=12)
    ax.legend(frameon=True, loc='upper right', fontsize=9, facecolor='white', edgecolor='gray', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    _clean_spines(ax)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    n_inits = len(init_modes)
    for i in range(1, len(priors)):
        sep_pos = i * n_inits + (i - 0.5) * 0.5
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    _savefig(output_dir, 'convergence_by_prior_init')
    print("✓ Convergence by Prior × Init saved (Stacked)")


def chart_convergence_by_cooperation(runs, output_dir):
    if 'cooperation_mode' not in runs.columns:
        print("Missing cooperation_mode column")
        return

    coop_data = runs.groupby('cooperation_mode').agg({
        's1_converged': 'mean',
        's2_converged': 'mean',
        'both_converged': 'mean',
    })

    modes = ['competitive', 'cooperative', 'partial']
    coop_data = coop_data.reindex(modes)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(modes))
    width = 0.25

    bars1 = ax.bar(x - width, coop_data['s1_converged'], width, label='Retailer',
                   color=COLORS['primary'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, coop_data['s2_converged'], width, label='Supplier',
                   color=COLORS['secondary'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, coop_data['both_converged'], width, label='Both',
                   color=COLORS['tertiary'], edgecolor='white', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.set_ylabel('Convergence Rate')
    ax.set_xlabel('Cooperation Mode')
    ax.set_title('Convergence by Cooperation Mode', fontweight='bold')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    _clean_spines(ax)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=8)

    _savefig(output_dir, 'convergence_by_cooperation')
    print("✓ Convergence by Cooperation (overall) saved")


def chart_factor_importance_comparison(runs, output_dir):
    algo_conv = runs.groupby('algo_pair')['both_converged'].mean()
    feasible_algos = algo_conv[algo_conv >= 0.01].index.tolist()
    runs_feasible = runs[runs['algo_pair'].isin(feasible_algos)].copy()

    factors = ['algo_pair', 'cooperation_mode', 'prior_knowledge', 'grid_size', 'init_mode']
    factor_labels = ['Algorithm\nPair', 'Cooperation\nMode', 'Prior\nKnowledge', 'Grid\nSize', 'Init\nMode']

    full_effects, full_pvalues = [], []
    cond_effects, cond_pvalues = [], []

    for factor in factors:
        if factor not in runs.columns:
            full_effects.append(0)
            full_pvalues.append(1.0)
            cond_effects.append(0)
            cond_pvalues.append(1.0)
            continue

        for dataset, effects_list, pvalues_list in [
            (runs, full_effects, full_pvalues),
            (runs_feasible, cond_effects, cond_pvalues),
        ]:
            grouped = dataset.groupby(factor)['both_converged'].mean()
            if len(grouped) >= 2:
                effects_list.append(grouped.max() - grouped.min())
                contingency = pd.crosstab(dataset[factor], dataset['both_converged'])
                if len(contingency) > 1 and len(contingency.columns) > 1:
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                    pvalues_list.append(p_value)
                else:
                    pvalues_list.append(1.0)
            else:
                effects_list.append(0)
                pvalues_list.append(1.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(factors))
    width = 0.35

    bars1 = ax.bar(x - width / 2, full_effects, width,
                   label=f'Full Dataset (n={len(runs):,})',
                   color=COLORS['primary'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width / 2, cond_effects, width,
                   label=f'Feasible Algorithms Only (n={len(runs_feasible):,})',
                   color=COLORS['secondary'], edgecolor='white', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(factor_labels, fontsize=10)
    ax.set_ylabel("Effect Size (Max - Min Convergence Rate)", fontsize=11)
    ax.set_title('Factor Importance: Full Dataset vs Feasible Algorithms',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.set_ylim(0, max(max(full_effects), max(cond_effects)) * 1.25)
    _clean_spines(ax)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val, pval in zip(bars1, full_effects, full_pvalues):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}\n{_sig_stars(pval)}', ha='center', va='bottom', fontsize=8)

    for bar, val, pval in zip(bars2, cond_effects, cond_pvalues):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}\n{_sig_stars(pval)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.text(0.5, -0.15, '*** p<0.001  ** p<0.01  * p<0.05  n.s. not significant',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic')

    _savefig(output_dir, 'factor_importance_comparison')
    print("✓ Factor Importance Comparison saved")


def generate_all_charts(results_dir='results_full_factorial', output_dir='latex_charts'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {results_dir}...")
    runs = load_data(results_dir)
    print(f"Loaded {len(runs):,} runs")

    print("\nGenerating charts...")
    print("-" * 60)

    chart_time_distribution(runs, output_dir)
    chart_solution_quality(runs, output_dir)
    chart_convergence_by_algo_pair(runs, output_dir)
    chart_regret_by_algo_pair(runs, output_dir)
    chart_convergence_by_grid_size(runs, output_dir)
    chart_convergence_time_by_grid_size(runs, output_dir)
    chart_both_convergence_by_algo_grid(runs, output_dir)
    chart_convergence_by_prior_init(runs, output_dir)
    chart_convergence_by_cooperation(runs, output_dir)
    chart_factor_importance_comparison(runs, output_dir)

    print("-" * 60)
    print(f"\n✓ All charts saved to {output_dir}/")
