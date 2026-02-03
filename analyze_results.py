#!/usr/bin/env python3
import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats.contingency import association


# Plot style configuration
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Raspberry
    'tertiary': '#F18F01',     # Orange
    'success': '#C73E1D',      # Rust red
    'neutral': '#6E7E85',      # Slate gray
    'highlight': '#4ECDC4',    # Teal
    'algorithms': {
        'greedy': '#2E86AB',
        'ucb': '#A23B72', 
        'thompson': '#F18F01',
        'exp3': '#C73E1D',
        'etc': '#4ECDC4',
    },
    'cooperation': {
        'competitive': '#E74C3C',
        'cooperative': '#27AE60',
        'partial': '#F39C12',
    }
}

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'grid.color': '#E0E0E0',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all result files from the results directory."""
    print(f"Loading results from {results_dir}...")
    
    results = {}
    
    # Load runs.csv (per-seed data)
    runs_path = os.path.join(results_dir, 'runs.csv')
    if os.path.exists(runs_path):
        results['runs'] = pd.read_csv(runs_path)
        print(f"  - Loaded runs.csv: {len(results['runs']):,} runs")
    else:
        raise FileNotFoundError(f"runs.csv not found in {results_dir}")
    
    # Load summary.csv (treatment-level aggregates)
    summary_path = os.path.join(results_dir, 'summary.csv')
    if os.path.exists(summary_path):
        results['summary'] = pd.read_csv(summary_path)
        print(f"  - Loaded summary.csv: {len(results['summary']):,} treatments")
    
    # Load benchmarks.csv (Nash equilibria and central optimum)
    benchmarks_path = os.path.join(results_dir, 'benchmarks.csv')
    if os.path.exists(benchmarks_path):
        results['benchmarks'] = pd.read_csv(benchmarks_path)
        print(f"  - Loaded benchmarks.csv: {len(results['benchmarks']):,} entries")
    
    # Load metadata.json
    metadata_path = os.path.join(results_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            results['metadata'] = json.load(f)
        print(f"  - Loaded metadata.json")
    
    # Load treatments.jsonl
    treatments_path = os.path.join(results_dir, 'treatments.jsonl')
    if os.path.exists(treatments_path):
        results['treatments'] = []
        with open(treatments_path, 'r') as f:
            for line in f:
                results['treatments'].append(json.loads(line))
        print(f"  - Loaded treatments.jsonl: {len(results['treatments']):,} treatments")
    
    return results


def preprocess_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess and enrich the data with derived columns."""
    runs = results['runs'].copy()
    summary = results['summary'].copy() if 'summary' in results else None
    
    # Extract algorithm from treatment name if not present
    if 'agent_retailer' not in runs.columns:
        runs['agent_retailer'] = runs['treatment'].str.split('_').str[0]
        runs['agent_supplier'] = runs['treatment'].str.split('_').str[1]
    
    if summary is not None and 'agent_retailer' not in summary.columns:
        summary['agent_retailer'] = summary['treatment'].str.split('_').str[0]
        summary['agent_supplier'] = summary['treatment'].str.split('_').str[1]
    
    # Create algorithm pair label
    runs['algo_pair'] = runs['agent_retailer'] + '/' + runs['agent_supplier']
    if summary is not None:
        summary['algo_pair'] = summary['agent_retailer'] + '/' + summary['agent_supplier']
    
    # Create grid size category
    if 's_upper' in runs.columns:
        runs['grid_size'] = runs['s_upper'].map(
            lambda x: 'coarse' if x <= 40 else ('medium' if x <= 60 else 'fine')
        )
        if summary is not None and 's_upper' in summary.columns:
            summary['grid_size'] = summary['s_upper'].map(
                lambda x: 'coarse' if x <= 40 else ('medium' if x <= 60 else 'fine')
            )
    else:
        # Extract s_upper from treatment name (format: algo1_algo2_s0-XX-1)
        def extract_s_upper(treatment_name):
            try:
                parts = treatment_name.split('_')
                s_part = [p for p in parts if p.startswith('s')][0]  # e.g., "s0-40-1"
                s_upper = int(s_part.split('-')[1])
                return s_upper
            except:
                return None
        
        runs['s_upper'] = runs['treatment'].apply(extract_s_upper)
        runs['grid_size'] = runs['s_upper'].map(
            lambda x: 'coarse' if x <= 40 else ('medium' if x <= 60 else 'fine') if x is not None else None
        )
        
        if summary is not None:
            summary['s_upper'] = summary['treatment'].apply(extract_s_upper)
            summary['grid_size'] = summary['s_upper'].map(
                lambda x: 'coarse' if x <= 40 else ('medium' if x <= 60 else 'fine') if x is not None else None
            )
    
    # Ensure proper types
    for col in ['both_converged', 's1_converged', 's2_converged', 'converged_to_central', 'converged_to_ne']:
        if col in runs.columns:
            runs[col] = runs[col].fillna(False).astype(bool)
    
    results['runs'] = runs
    if summary is not None:
        results['summary'] = summary
    
    return results


# Q1: DO AGENTS CONVERGE?

def analyze_convergence_rates(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze overall and segmented convergence rates."""
    runs = results['runs']
    
    analysis = {
        'overall': {},
        'by_agent': {},
        'by_algorithm': {},
        'by_mechanism': {},
    }
    
    # Overall convergence
    analysis['overall'] = {
        's1_convergence_rate': runs['s1_converged'].mean(),
        's2_convergence_rate': runs['s2_converged'].mean(),
        'both_convergence_rate': runs['both_converged'].mean(),
        'total_runs': len(runs),
        'converged_runs': runs['both_converged'].sum(),
    }
    
    # By algorithm pair
    algo_conv = runs.groupby('algo_pair').agg({
        's1_converged': 'mean',
        's2_converged': 'mean',
        'both_converged': ['mean', 'count'],
    }).round(4)
    algo_conv.columns = ['s1_rate', 's2_rate', 'both_rate', 'n_runs']
    analysis['by_algorithm'] = algo_conv.to_dict('index')
    
    # By cooperation mode
    if 'cooperation_mode' in runs.columns:
        coop_conv = runs.groupby('cooperation_mode').agg({
            's1_converged': 'mean',
            's2_converged': 'mean',
            'both_converged': 'mean',
        }).round(4)
        analysis['by_mechanism']['cooperation_mode'] = coop_conv.to_dict('index')
    
    # By prior knowledge
    if 'prior_knowledge' in runs.columns:
        prior_conv = runs.groupby('prior_knowledge').agg({
            's1_converged': 'mean',
            's2_converged': 'mean',
            'both_converged': 'mean',
        }).round(4)
        analysis['by_mechanism']['prior_knowledge'] = prior_conv.to_dict('index')
    
    # By initialization mode
    if 'init_mode' in runs.columns:
        init_conv = runs.groupby('init_mode').agg({
            's1_converged': 'mean',
            's2_converged': 'mean',
            'both_converged': 'mean',
        }).round(4)
        analysis['by_mechanism']['init_mode'] = init_conv.to_dict('index')
    
    # By grid size
    if 'grid_size' in runs.columns:
        grid_conv = runs.groupby('grid_size').agg({
            's1_converged': 'mean',
            's2_converged': 'mean',
            'both_converged': 'mean',
        }).round(4)
        analysis['by_mechanism']['grid_size'] = grid_conv.to_dict('index')
    
    return analysis


# Q2: HOW FAST DO THEY CONVERGE?

def analyze_convergence_speed(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze convergence speed among converged runs."""
    runs = results['runs']
    
    analysis = {
        'overall': {},
        'by_algorithm': {},
        'by_mechanism': {},
    }
    
    # Filter to converged runs only
    converged = runs[runs['both_converged'] == True].copy()
    
    if len(converged) == 0:
        return analysis
    
    # Overall convergence times
    s1_times = converged['s1_conv_time'].dropna()
    s2_times = converged['s2_conv_time'].dropna()
    
    analysis['overall'] = {
        's1_conv_time_mean': s1_times.mean() if len(s1_times) > 0 else None,
        's1_conv_time_std': s1_times.std() if len(s1_times) > 0 else None,
        's1_conv_time_median': s1_times.median() if len(s1_times) > 0 else None,
        's2_conv_time_mean': s2_times.mean() if len(s2_times) > 0 else None,
        's2_conv_time_std': s2_times.std() if len(s2_times) > 0 else None,
        's2_conv_time_median': s2_times.median() if len(s2_times) > 0 else None,
        'n_converged': len(converged),
    }
    
    # By algorithm pair
    for algo_pair in converged['algo_pair'].unique():
        mask = converged['algo_pair'] == algo_pair
        s1_t = converged.loc[mask, 's1_conv_time'].dropna()
        s2_t = converged.loc[mask, 's2_conv_time'].dropna()
        analysis['by_algorithm'][algo_pair] = {
            's1_mean': s1_t.mean() if len(s1_t) > 0 else None,
            's2_mean': s2_t.mean() if len(s2_t) > 0 else None,
            's1_std': s1_t.std() if len(s1_t) > 0 else None,
            's2_std': s2_t.std() if len(s2_t) > 0 else None,
            'n': len(converged[mask]),
        }
    
    # By mechanism
    for mech in ['cooperation_mode', 'prior_knowledge', 'init_mode', 'grid_size']:
        if mech in converged.columns:
            analysis['by_mechanism'][mech] = {}
            for val in converged[mech].unique():
                mask = converged[mech] == val
                s1_t = converged.loc[mask, 's1_conv_time'].dropna()
                s2_t = converged.loc[mask, 's2_conv_time'].dropna()
                analysis['by_mechanism'][mech][val] = {
                    's1_mean': s1_t.mean() if len(s1_t) > 0 else None,
                    's2_mean': s2_t.mean() if len(s2_t) > 0 else None,
                    'n': len(converged[mask]),
                }
    
    return analysis


# Q3: TO WHAT SOLUTIONS?

def analyze_solution_quality(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze what solutions agents converge to."""
    runs = results['runs']
    
    analysis = {
        'overall': {},
        'by_algorithm': {},
        'by_mechanism': {},
        'distance_analysis': {},
    }
    
    # Filter to converged runs
    converged = runs[runs['both_converged'] == True].copy()
    
    if len(converged) == 0:
        return analysis
    
    # Overall solution quality
    to_central = converged['converged_to_central'].sum() if 'converged_to_central' in converged.columns else 0
    to_ne = converged['converged_to_ne'].sum() if 'converged_to_ne' in converged.columns else 0
    
    # Count runs where NE analysis was possible
    ne_analyzed = converged['converged_to_ne'].notna().sum() if 'converged_to_ne' in converged.columns else 0
    
    analysis['overall'] = {
        'n_converged': len(converged),
        'to_central': int(to_central),
        'to_central_rate': to_central / len(converged) if len(converged) > 0 else 0,
        'to_ne': int(to_ne),
        'to_ne_rate': to_ne / ne_analyzed if ne_analyzed > 0 else 0,
        'ne_analyzed': int(ne_analyzed),
    }
    
    # Distance to central optimum
    if 'distance_to_central' in converged.columns:
        dist = converged['distance_to_central'].dropna()
        analysis['distance_analysis']['to_central'] = {
            'mean': dist.mean(),
            'std': dist.std(),
            'median': dist.median(),
            'min': dist.min(),
            'max': dist.max(),
        }
    
    # Distance to Nash
    if 'distance_to_nash' in converged.columns:
        dist = converged['distance_to_nash'].dropna()
        analysis['distance_analysis']['to_nash'] = {
            'mean': dist.mean(),
            'std': dist.std(),
            'median': dist.median(),
            'min': dist.min(),
            'max': dist.max(),
        }
    
    # Deviation incentives
    if 'delta1' in converged.columns and 'delta2' in converged.columns:
        d1 = converged['delta1'].dropna()
        d2 = converged['delta2'].dropna()
        analysis['distance_analysis']['deviation_incentives'] = {
            'delta1_mean': d1.mean() if len(d1) > 0 else None,
            'delta1_std': d1.std() if len(d1) > 0 else None,
            'delta2_mean': d2.mean() if len(d2) > 0 else None,
            'delta2_std': d2.std() if len(d2) > 0 else None,
        }
    
    # By algorithm pair
    for algo_pair in converged['algo_pair'].unique():
        mask = converged['algo_pair'] == algo_pair
        subset = converged[mask]
        ne_mask = subset['converged_to_ne'].notna() if 'converged_to_ne' in subset.columns else pd.Series([False]*len(subset))
        analysis['by_algorithm'][algo_pair] = {
            'n': len(subset),
            'to_central_rate': subset['converged_to_central'].mean() if 'converged_to_central' in subset.columns else 0,
            'to_ne_rate': subset.loc[ne_mask, 'converged_to_ne'].mean() if ne_mask.sum() > 0 else None,
        }
    
    # By mechanism
    for mech in ['cooperation_mode', 'prior_knowledge', 'init_mode', 'grid_size']:
        if mech in converged.columns:
            analysis['by_mechanism'][mech] = {}
            for val in converged[mech].unique():
                mask = converged[mech] == val
                subset = converged[mask]
                ne_mask = subset['converged_to_ne'].notna() if 'converged_to_ne' in subset.columns else pd.Series([False]*len(subset))
                analysis['by_mechanism'][mech][val] = {
                    'n': len(subset),
                    'to_central_rate': subset['converged_to_central'].mean() if 'converged_to_central' in subset.columns else 0,
                    'to_ne_rate': subset.loc[ne_mask, 'converged_to_ne'].mean() if ne_mask.sum() > 0 else None,
                }
    
    return analysis


# ============================================================================
# Q4: WHICH MECHANISMS INFLUENCE CONVERGENCE?
# ============================================================================

def analyze_mechanism_effects(results: Dict[str, Any]) -> Dict[str, Any]:
    """Statistical analysis of mechanism effects on convergence."""
    runs = results['runs']
    
    analysis = {
        'main_effects': {},
        'main_effects_separate': {},  # S1, S2, Both separately
        'effect_sizes': {},
        'interactions': {},
        'stratified': {},
        'rankings': {},
    }
    
    # Main effects - SEPARATE analysis for S1, S2, and Both
    factors = ['cooperation_mode', 'prior_knowledge', 'init_mode', 'grid_size', 
               'agent_retailer', 'agent_supplier', 'algo_pair']
    
    for factor in factors:
        if factor not in runs.columns:
            continue
        
        # Analyze all three outcomes separately
        analysis['main_effects_separate'][factor] = {}
        
        for outcome in ['s1_converged', 's2_converged', 'both_converged']:
            grouped = runs.groupby(factor).agg({
                outcome: ['mean', 'std', 'count'],
                'train_total_regret': ['mean', 'std'],
            })
            grouped.columns = ['conv_mean', 'conv_std', 'n', 'regret_mean', 'regret_std']
            
            # Chi-square test and Cramér's V effect size
            contingency = pd.crosstab(runs[factor], runs[outcome])
            if len(contingency) > 1 and len(contingency.columns) > 1:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                # Cramér's V: effect size for chi-square
                n = contingency.sum().sum()
                min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            else:
                chi2, p_value, cramers_v = 0, 1.0, 0
            
            analysis['main_effects_separate'][factor][outcome] = {
                'levels': grouped.to_dict('index'),
                'chi2': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'significant': p_value < 0.05,
            }
        
        # Keep "both_converged" as main effect for backward compatibility
        analysis['main_effects'][factor] = analysis['main_effects_separate'][factor]['both_converged']
    
    # Effect size summary
    for factor in factors:
        if factor in analysis['main_effects_separate']:
            analysis['effect_sizes'][factor] = {
                outcome: analysis['main_effects_separate'][factor][outcome]['cramers_v']
                for outcome in ['s1_converged', 's2_converged', 'both_converged']
            }
    
    # Stratified analysis: Cooperation effect within each algorithm pair
    if 'cooperation_mode' in runs.columns and 'algo_pair' in runs.columns:
        stratified_coop = []
        for algo in runs['algo_pair'].unique():
            algo_data = runs[runs['algo_pair'] == algo]
            coop_effect = algo_data.groupby('cooperation_mode')['both_converged'].agg(['mean', 'count'])
            coop_effect = coop_effect.reset_index()
            coop_effect['algo_pair'] = algo
            stratified_coop.append(coop_effect)
        
        if stratified_coop:
            analysis['stratified']['coop_by_algo'] = pd.concat(stratified_coop).to_dict('records')
    
    # Stratified: Prior knowledge effect within each grid size
    if 'prior_knowledge' in runs.columns and 'grid_size' in runs.columns:
        stratified_prior = []
        for grid in runs['grid_size'].unique():
            grid_data = runs[runs['grid_size'] == grid]
            prior_effect = grid_data.groupby('prior_knowledge')['both_converged'].agg(['mean', 'count'])
            prior_effect = prior_effect.reset_index()
            prior_effect['grid_size'] = grid
            stratified_prior.append(prior_effect)
        
        if stratified_prior:
            analysis['stratified']['prior_by_grid'] = pd.concat(stratified_prior).to_dict('records')
    
    # Interaction: Algorithm x Cooperation Mode (extended with stats)
    if 'cooperation_mode' in runs.columns and 'algo_pair' in runs.columns:
        interaction = runs.groupby(['algo_pair', 'cooperation_mode'])['both_converged'].agg(['mean', 'count'])
        interaction.columns = ['rate', 'n']
        analysis['interactions']['algo_coop'] = interaction.reset_index().to_dict('records')
    
    # Interaction: Prior Knowledge x Initialization
    if 'prior_knowledge' in runs.columns and 'init_mode' in runs.columns:
        interaction = runs.groupby(['prior_knowledge', 'init_mode'])['both_converged'].agg(['mean', 'count'])
        interaction.columns = ['rate', 'n']
        analysis['interactions']['prior_init'] = interaction.reset_index().to_dict('records')
    
    # NEW: Interaction - Prior x Grid
    if 'prior_knowledge' in runs.columns and 'grid_size' in runs.columns:
        interaction = runs.groupby(['prior_knowledge', 'grid_size'])['both_converged'].agg(['mean', 'count'])
        interaction.columns = ['rate', 'n']
        analysis['interactions']['prior_grid'] = interaction.reset_index().to_dict('records')
    
    # NEW: Interaction - Algo x Grid (is grid size effect algo-specific?)
    if 'algo_pair' in runs.columns and 'grid_size' in runs.columns:
        # Only for top algos to keep it manageable
        top_algos = runs.groupby('algo_pair')['both_converged'].mean().nlargest(10).index
        algo_grid_data = runs[runs['algo_pair'].isin(top_algos)]
        interaction = algo_grid_data.groupby(['algo_pair', 'grid_size'])['both_converged'].agg(['mean', 'count'])
        interaction.columns = ['rate', 'n']
        analysis['interactions']['algo_grid_top10'] = interaction.reset_index().to_dict('records')
    
    # Rankings
    if 'treatment_full' in runs.columns:
        treatment_col = 'treatment_full'
    else:
        treatment_col = 'treatment'
    
    rankings = runs.groupby(treatment_col).agg({
        'both_converged': 'mean',
        'train_total_regret': 'mean',
        'converged_to_ne': lambda x: x.dropna().mean() if x.notna().any() else None,
        'converged_to_central': 'mean',
    }).reset_index()
    rankings.columns = ['treatment', 'conv_rate', 'regret', 'ne_rate', 'central_rate']
    
    # Top/bottom by convergence
    analysis['rankings']['top_by_convergence'] = rankings.nlargest(15, 'conv_rate').to_dict('records')
    analysis['rankings']['bottom_by_convergence'] = rankings.nsmallest(15, 'conv_rate').to_dict('records')
    
    # Top by regret (lowest is best)
    analysis['rankings']['top_by_regret'] = rankings.nsmallest(15, 'regret').to_dict('records')
    
    return analysis


def plot_convergence_overview(results: Dict[str, Any], conv_analysis: Dict, output_dir: str):
    """Create overview plot of convergence rates."""
    runs = results['runs']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Overall convergence rates
    ax = axes[0, 0]
    categories = ['Retailer (S1)', 'Supplier (S2)', 'Both Agents']
    rates = [
        conv_analysis['overall']['s1_convergence_rate'],
        conv_analysis['overall']['s2_convergence_rate'],
        conv_analysis['overall']['both_convergence_rate'],
    ]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    bars = ax.bar(categories, rates, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Overall Convergence Rates', fontweight='bold')
    ax.set_ylim(0, 1.05)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    # 2. Convergence by cooperation mode
    ax = axes[0, 1]
    if 'cooperation_mode' in conv_analysis['by_mechanism']:
        coop_data = conv_analysis['by_mechanism']['cooperation_mode']
        modes = list(coop_data.keys())
        x = np.arange(len(modes))
        width = 0.25
        
        s1_rates = [coop_data[m]['s1_converged'] for m in modes]
        s2_rates = [coop_data[m]['s2_converged'] for m in modes]
        both_rates = [coop_data[m]['both_converged'] for m in modes]
        
        ax.bar(x - width, s1_rates, width, label='S1', color=COLORS['primary'])
        ax.bar(x, s2_rates, width, label='S2', color=COLORS['secondary'])
        ax.bar(x + width, both_rates, width, label='Both', color=COLORS['tertiary'])
        
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Convergence by Cooperation Mode', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.05)
    
    # 3. Algorithm heatmap
    ax = axes[1, 0]
    algos = ['greedy', 'ucb', 'thompson', 'exp3', 'etc']
    algo_matrix = np.zeros((len(algos), len(algos)))
    
    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            pair = f'{a1}/{a2}'
            if pair in conv_analysis['by_algorithm']:
                algo_matrix[i, j] = conv_analysis['by_algorithm'][pair]['both_rate']
    
    im = ax.imshow(algo_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(algos)))
    ax.set_yticks(np.arange(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos], fontsize=9)
    ax.set_yticklabels([a.upper() for a in algos], fontsize=9)
    ax.set_xlabel('Supplier Algorithm')
    ax.set_ylabel('Retailer Algorithm')
    ax.set_title('Convergence Rate by Algorithm Pair', fontweight='bold')
    
    for i in range(len(algos)):
        for j in range(len(algos)):
            text = ax.text(j, i, f'{algo_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if algo_matrix[i, j] < 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, label='Convergence Rate')
    
    # 4. Convergence by other factors
    ax = axes[1, 1]
    factor_effects = []
    factor_names = []
    
    for factor in ['prior_knowledge', 'init_mode', 'grid_size']:
        if factor in conv_analysis['by_mechanism']:
            for level, data in conv_analysis['by_mechanism'][factor].items():
                factor_effects.append(data['both_converged'])
                factor_names.append(f'{factor.replace("_", " ").title()}\n({level})')
    
    if factor_effects:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(factor_effects)))
        bars = ax.barh(range(len(factor_effects)), factor_effects, color=colors)
        ax.set_yticks(range(len(factor_effects)))
        ax.set_yticklabels(factor_names, fontsize=8)
        ax.set_xlabel('Convergence Rate')
        ax.set_title('Convergence by Treatment Factor', fontweight='bold')
        ax.set_xlim(0, 1.05)
        
        for bar, rate in zip(bars, factor_effects):
            ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{rate:.1%}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_speed(results: Dict[str, Any], speed_analysis: Dict, output_dir: str):
    """Create plot of convergence speed analysis."""
    runs = results['runs']
    converged = runs[runs['both_converged'] == True].copy()
    
    if len(converged) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Distribution of convergence times
    ax = axes[0, 0]
    s1_times = converged['s1_conv_time'].dropna()
    s2_times = converged['s2_conv_time'].dropna()
    
    if len(s1_times) > 0 and len(s2_times) > 0:
        ax.hist(s1_times, bins=30, alpha=0.6, label=f'S1 (μ={s1_times.mean():.0f})', 
                color=COLORS['primary'], edgecolor='white')
        ax.hist(s2_times, bins=30, alpha=0.6, label=f'S2 (μ={s2_times.mean():.0f})', 
                color=COLORS['secondary'], edgecolor='white')
        ax.set_xlabel('Convergence Time (rounds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Convergence Times', fontweight='bold')
        ax.legend()
        ax.axvline(s1_times.median(), color=COLORS['primary'], linestyle='--', alpha=0.8)
        ax.axvline(s2_times.median(), color=COLORS['secondary'], linestyle='--', alpha=0.8)
    
    # 2. Convergence time by algorithm
    ax = axes[0, 1]
    algo_pairs = sorted(speed_analysis['by_algorithm'].keys())
    x = np.arange(len(algo_pairs))
    width = 0.35
    
    s1_means = [speed_analysis['by_algorithm'][a].get('s1_mean', 0) or 0 for a in algo_pairs]
    s2_means = [speed_analysis['by_algorithm'][a].get('s2_mean', 0) or 0 for a in algo_pairs]
    
    ax.bar(x - width/2, s1_means, width, label='S1', color=COLORS['primary'])
    ax.bar(x + width/2, s2_means, width, label='S2', color=COLORS['secondary'])
    ax.set_xticks(x)
    ax.set_xticklabels(algo_pairs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Convergence Time (rounds)')
    ax.set_title('Convergence Time by Algorithm Pair', fontweight='bold')
    ax.legend()
    
    # 3. Convergence time by cooperation mode
    ax = axes[1, 0]
    if 'cooperation_mode' in speed_analysis['by_mechanism']:
        coop_data = speed_analysis['by_mechanism']['cooperation_mode']
        modes = list(coop_data.keys())
        x = np.arange(len(modes))
        
        s1_means = [coop_data[m].get('s1_mean', 0) or 0 for m in modes]
        s2_means = [coop_data[m].get('s2_mean', 0) or 0 for m in modes]
        
        ax.bar(x - width/2, s1_means, width, label='S1', color=COLORS['primary'])
        ax.bar(x + width/2, s2_means, width, label='S2', color=COLORS['secondary'])
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.set_ylabel('Convergence Time (rounds)')
        ax.set_title('Convergence Time by Cooperation Mode', fontweight='bold')
        ax.legend()
    
    # 4. Scatter: Convergence time vs Regret
    ax = axes[1, 1]
    valid = converged[converged['s1_conv_time'].notna() & converged['train_total_regret'].notna()].copy()
    if len(valid) > 0:
        # Use cooperation mode for coloring if available
        if 'cooperation_mode' in valid.columns:
            for mode in valid['cooperation_mode'].unique():
                mask = valid['cooperation_mode'] == mode
                ax.scatter(valid.loc[mask, 's1_conv_time'], 
                          valid.loc[mask, 'train_total_regret'],
                          alpha=0.3, s=20, label=mode.capitalize(),
                          color=COLORS['cooperation'].get(mode, COLORS['neutral']))
            ax.legend(title='Cooperation Mode')
        else:
            ax.scatter(valid['s1_conv_time'], valid['train_total_regret'], 
                      alpha=0.3, s=20, color=COLORS['primary'])
        
        ax.set_xlabel('Convergence Time (S1)')
        ax.set_ylabel('Total Regret')
        ax.set_title('Convergence Time vs Regret', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_speed.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_by_grid_size(results: Dict[str, Any], conv_analysis: Dict, speed_analysis: Dict,
                                  output_dir: str):
    """Create selection graphs showing convergence behavior by action grid size."""
    runs = results['runs']
    if 'grid_size' not in runs.columns:
        return

    # Order grid sizes consistently (coarse -> medium -> fine)
    grid_order = ['coarse', 'medium', 'fine']
    runs_grid = runs[runs['grid_size'].isin(grid_order)].copy()
    if len(runs_grid) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Convergence rates by grid size (S1, S2, Both)
    ax = axes[0, 0]
    x = np.arange(len(grid_order))
    width = 0.25

    s1_rates = [conv_analysis['by_mechanism']['grid_size'].get(g, {}).get('s1_converged', 0) for g in grid_order]
    s2_rates = [conv_analysis['by_mechanism']['grid_size'].get(g, {}).get('s2_converged', 0) for g in grid_order]
    both_rates = [conv_analysis['by_mechanism']['grid_size'].get(g, {}).get('both_converged', 0) for g in grid_order]

    ax.bar(x - width, s1_rates, width, label='S1 (Retailer)', color=COLORS['primary'])
    ax.bar(x, s2_rates, width, label='S2 (Supplier)', color=COLORS['secondary'])
    ax.bar(x + width, both_rates, width, label='Both', color=COLORS['tertiary'])

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in grid_order])
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Convergence Rate by Grid Size', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for i, (s1, s2, both) in enumerate(zip(s1_rates, s2_rates, both_rates)):
        for j, (val, offset) in enumerate([(s1, -width), (s2, 0), (both, width)]):
            if val > 0.05:
                ax.text(i + offset, val + 0.02, f'{val:.0%}', ha='center', va='bottom', fontsize=8)

    # 2. Convergence rate by grid size × cooperation mode
    ax = axes[0, 1]
    if 'cooperation_mode' in runs_grid.columns:
        coop_modes = sorted(runs_grid['cooperation_mode'].unique())
        x = np.arange(len(grid_order))
        width = 0.8 / max(len(coop_modes), 1)

        for i, mode in enumerate(coop_modes):
            offset = (i - len(coop_modes) / 2 + 0.5) * width
            group = runs_grid[runs_grid['cooperation_mode'] == mode]
            both_rates_coop = [
                group[group['grid_size'] == g]['both_converged'].mean() if len(group[group['grid_size'] == g]) > 0 else 0
                for g in grid_order
            ]
            color = COLORS['cooperation'].get(mode, COLORS['neutral'])
            bars = ax.bar(x + offset, both_rates_coop, width, label=mode.capitalize(), color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in grid_order])
        ax.set_ylabel('Both-Agent Convergence Rate')
        ax.set_title('Convergence Rate by Grid Size × Cooperation Mode', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, 'No cooperation_mode data', ha='center', va='center', transform=ax.transAxes)

    # 3. Convergence time by grid size (among converged runs)
    ax = axes[1, 0]
    converged = runs_grid[runs_grid['both_converged'] == True]
    if len(converged) > 0 and 's1_conv_time' in converged.columns:
        x = np.arange(len(grid_order))
        width = 0.35

        s1_means = []
        s2_means = []
        s1_stds = []
        s2_stds = []
        for g in grid_order:
            sub = converged[converged['grid_size'] == g]
            s1_t = sub['s1_conv_time'].dropna()
            s2_t = sub['s2_conv_time'].dropna()
            s1_means.append(s1_t.mean() if len(s1_t) > 0 else 0)
            s2_means.append(s2_t.mean() if len(s2_t) > 0 else 0)
            s1_stds.append(s1_t.std() if len(s1_t) > 1 else 0)
            s2_stds.append(s2_t.std() if len(s2_t) > 1 else 0)

        ax.bar(x - width/2, s1_means, width, yerr=s1_stds, capsize=3, label='S1', color=COLORS['primary'])
        ax.bar(x + width/2, s2_means, width, yerr=s2_stds, capsize=3, label='S2', color=COLORS['secondary'])
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in grid_order])
        ax.set_ylabel('Mean Convergence Time (rounds)')
        ax.set_title('Convergence Time by Grid Size (converged runs only)', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No converged runs', ha='center', va='center', transform=ax.transAxes)

    # 4. Heatmap: algorithm pair × grid size for both_convergence_rate
    ax = axes[1, 1]
    algos = ['greedy', 'ucb', 'thompson', 'exp3', 'etc']
    grid_labels = [g.capitalize() for g in grid_order]
    heatmap_data = np.zeros((len(algos) * len(algos), len(grid_order)))

    for gi, grid in enumerate(grid_order):
        grid_runs = runs_grid[runs_grid['grid_size'] == grid]
        for i, a1 in enumerate(algos):
            for j, a2 in enumerate(algos):
                pair = f'{a1}/{a2}'
                row_idx = i * len(algos) + j
                pair_data = grid_runs[grid_runs['algo_pair'] == pair]
                if len(pair_data) > 0:
                    heatmap_data[row_idx, gi] = pair_data['both_converged'].mean()
                else:
                    heatmap_data[row_idx, gi] = np.nan

    # Reshape for 5x5 × 3 grid display: show top algorithm pairs or full matrix
    # Use algo pairs as rows, grid as columns
    pair_labels = [f'{a1}/{a2}' for a1 in algos for a2 in algos]
    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(grid_order)))
    ax.set_xticklabels(grid_labels)
    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Algorithm Pair (Retailer/Supplier)')
    ax.set_title('Both-Agent Convergence Rate by Algorithm Pair × Grid Size', fontweight='bold')

    for i in range(len(pair_labels)):
        for j in range(len(grid_order)):
            val = heatmap_data[i, j]
            text = ax.text(j, i, f'{val:.2f}' if not np.isnan(val) else '-',
                          ha='center', va='center', fontsize=7,
                          color='white' if val < 0.5 and not np.isnan(val) else 'black')

    plt.colorbar(im, ax=ax, label='Convergence Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_by_grid_size.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_outcome_by_prior_init(results: Dict[str, Any], solution_analysis: Dict, output_dir: str):
    """Create graph showing to which optimum converged runs landed, by prior_knowledge and init_mode."""
    runs = results['runs']
    converged = runs[runs['both_converged'] == True].copy()

    if len(converged) == 0:
        return

    if 'prior_knowledge' not in converged.columns or 'init_mode' not in converged.columns:
        return

    if 'converged_to_central' not in converged.columns:
        return

    # Mutually exclusive categories among converged runs:
    # - Central optimum: converged_to_central == True
    # - Nash equilibrium only: converged_to_ne == True and not central
    # - Other stable: neither central nor NE
    converged['outcome'] = 'Other stable'
    converged.loc[converged['converged_to_central'] == True, 'outcome'] = 'Central optimum'
    ne_mask = (converged['converged_to_ne'] == True) & (converged['converged_to_central'] != True)
    converged.loc[ne_mask, 'outcome'] = 'Nash equilibrium (not central)'

    # Build combination labels
    prior_order = sorted(converged['prior_knowledge'].unique(), key=lambda x: (x == 'none', x))
    init_order = sorted(converged['init_mode'].unique(), key=lambda x: (x == 'random', x))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Stacked bar chart: prior_knowledge × init_mode
    ax = axes[0]
    outcome_order = ['Central optimum', 'Nash equilibrium (not central)', 'Other stable']
    outcome_colors = {
        'Central optimum': COLORS['tertiary'],
        'Nash equilibrium (not central)': COLORS['secondary'],
        'Other stable': COLORS['neutral'],
    }

    combinations = [(p, i) for p in prior_order for i in init_order]
    x_labels = [f"{p}\n{i}" for p, i in combinations]
    x = np.arange(len(combinations))
    width = 0.7

    bottom = np.zeros(len(combinations))
    for outcome in outcome_order:
        counts = []
        for p, i in combinations:
            mask = (converged['prior_knowledge'] == p) & (converged['init_mode'] == i) & (converged['outcome'] == outcome)
            counts.append(converged.loc[mask].shape[0])
        totals = [converged[(converged['prior_knowledge'] == p) & (converged['init_mode'] == i)].shape[0] for p, i in combinations]
        # Proportions (as fraction of converged runs in that prior/init group)
        proportions = [c / t if t > 0 else 0 for c, t in zip(counts, totals)]
        ax.bar(x, proportions, width, bottom=bottom, label=outcome, color=outcome_colors[outcome], edgecolor='white', linewidth=0.5)
        bottom += np.array(proportions)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('Proportion of converged runs')
    ax.set_title('Convergence Outcome by Prior Knowledge × Init Mode', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)

    # 2. Grouped bar chart: same data, absolute counts
    ax = axes[1]
    bar_width = 0.25
    for i, outcome in enumerate(outcome_order):
        counts = []
        for p, init in combinations:
            mask = (converged['prior_knowledge'] == p) & (converged['init_mode'] == init) & (converged['outcome'] == outcome)
            counts.append(converged.loc[mask].shape[0])
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, counts, bar_width, label=outcome, color=outcome_colors[outcome], edgecolor='white', linewidth=0.5)
        max_count = max(counts) if counts else 1
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count * 0.02, f'{c}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel('Count of converged runs')
    ax.set_title('Convergence Outcome Counts by Prior Knowledge × Init Mode', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_outcome_by_prior_init.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_solution_quality(results: Dict[str, Any], solution_analysis: Dict, output_dir: str):
    """Create plot of solution quality analysis."""
    runs = results['runs']
    converged = runs[runs['both_converged'] == True].copy()
    
    if len(converged) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Solution outcome pie chart
    ax = axes[0, 0]
    n_total = len(converged)
    n_central = converged['converged_to_central'].sum() if 'converged_to_central' in converged.columns else 0
    n_ne = converged['converged_to_ne'].sum() if 'converged_to_ne' in converged.columns else 0
    n_other = n_total - max(n_central, n_ne)  # Avoid double counting
    
    # For better visualization, show breakdown
    labels = ['Central Optimum', 'Nash Equilibrium', 'Other Stable']
    sizes = [n_central, max(0, n_ne - n_central), n_other]  # NE includes central for some
    colors = [COLORS['tertiary'], COLORS['secondary'], COLORS['neutral']]
    explode = (0.05, 0.05, 0)
    
    # Filter out zero sizes
    valid_idx = [i for i, s in enumerate(sizes) if s > 0]
    if valid_idx:
        ax.pie([sizes[i] for i in valid_idx], 
               labels=[labels[i] for i in valid_idx],
               colors=[colors[i] for i in valid_idx],
               explode=[explode[i] for i in valid_idx],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Solution Outcomes (n={n_total:,} converged)', fontweight='bold')
    
    # 2. Distance to optimum distribution
    ax = axes[0, 1]
    if 'distance_to_central' in converged.columns:
        dist = converged['distance_to_central'].dropna()
        if len(dist) > 0:
            ax.hist(dist, bins=30, color=COLORS['primary'], edgecolor='white', alpha=0.7)
            ax.axvline(dist.mean(), color=COLORS['secondary'], linestyle='--', 
                      label=f'Mean: {dist.mean():.1f}', linewidth=2)
            ax.axvline(dist.median(), color=COLORS['tertiary'], linestyle=':', 
                      label=f'Median: {dist.median():.1f}', linewidth=2)
            ax.set_xlabel('L1 Distance to Central Optimum')
            ax.set_ylabel('Frequency')
            ax.set_title('Distance to Central Optimum', fontweight='bold')
            ax.legend()
    
    # 3. NE/Central rate by algorithm
    ax = axes[1, 0]
    algo_data = solution_analysis['by_algorithm']
    algo_pairs = sorted(algo_data.keys())
    x = np.arange(len(algo_pairs))
    width = 0.35
    
    central_rates = [algo_data[a]['to_central_rate'] for a in algo_pairs]
    ne_rates = [algo_data[a].get('to_ne_rate', 0) or 0 for a in algo_pairs]
    
    ax.bar(x - width/2, central_rates, width, label='Central Opt.', color=COLORS['tertiary'])
    ax.bar(x + width/2, ne_rates, width, label='Nash Eq.', color=COLORS['secondary'])
    ax.set_xticks(x)
    ax.set_xticklabels(algo_pairs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Rate (among converged)')
    ax.set_title('Solution Quality by Algorithm Pair', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(max(central_rates), max(ne_rates)) * 1.2 if central_rates else 1)
    
    # 4. Final action scatter (sample)
    ax = axes[1, 1]
    if 's1_mode' in converged.columns and 's2_mode' in converged.columns:
        sample = converged.sample(min(1000, len(converged)), random_state=42)
        
        # Plot final positions
        ax.scatter(sample['s1_mode'], sample['s2_mode'], alpha=0.2, s=20, 
                  c=COLORS['primary'], label='Final actions')
        
        # Plot central optimum if available
        if 's1_opt' in sample.columns and 's2_opt' in sample.columns:
            s1_opt = sample['s1_opt'].iloc[0]
            s2_opt = sample['s2_opt'].iloc[0]
            ax.scatter([s1_opt], [s2_opt], marker='*', s=300, c=COLORS['tertiary'], 
                      edgecolor='black', zorder=10, label='Central Opt.')
        
        ax.set_xlabel('S1 (Retailer) Final Action')
        ax.set_ylabel('S2 (Supplier) Final Action')
        ax.set_title('Final Action Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'solution_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mechanism_effects(results: Dict[str, Any], mech_analysis: Dict, output_dir: str):
    """Create detailed mechanism effects visualization."""
    runs = results['runs']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Factor importance (effect sizes)
    ax = axes[0, 0]
    factors = []
    effect_sizes = []
    p_values = []
    
    for factor, data in mech_analysis['main_effects'].items():
        if factor in ['agent_retailer', 'agent_supplier']:
            continue  # Skip individual algos, use algo_pair
        levels = list(data['levels'].values())
        if len(levels) >= 2:
            conv_rates = [l['conv_mean'] for l in levels]
            effect = max(conv_rates) - min(conv_rates)
            factors.append(factor.replace('_', ' ').title())
            effect_sizes.append(effect)
            p_values.append(data['p_value'])
    
    if factors:
        colors = [COLORS['success'] if p < 0.05 else COLORS['neutral'] for p in p_values]
        bars = ax.barh(range(len(factors)), effect_sizes, color=colors)
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_xlabel('Effect Size (Max - Min Convergence Rate)')
        ax.set_title('Factor Importance (red = p < 0.05)', fontweight='bold')
        
        for bar, effect, p in zip(bars, effect_sizes, p_values):
            ax.text(effect + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{effect:.2f} (p={p:.3f})', ha='left', va='center', fontsize=8)
    
    # 2. Algorithm x Cooperation interaction
    ax = axes[0, 1]
    if 'algo_coop' in mech_analysis['interactions']:
        data = mech_analysis['interactions']['algo_coop']
        df = pd.DataFrame(data)
        pivot = df.pivot(index='algo_pair', columns='cooperation_mode', values='rate')
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([c.capitalize() for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel('Cooperation Mode')
        ax.set_ylabel('Algorithm Pair')
        ax.set_title('Algorithm × Cooperation Interaction', fontweight='bold')
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                           color='white' if val < 0.5 else 'black')
        
        plt.colorbar(im, ax=ax, label='Convergence Rate')
    
    # 3. Prior x Init interaction
    ax = axes[0, 2]
    if 'prior_init' in mech_analysis['interactions']:
        data = mech_analysis['interactions']['prior_init']
        df = pd.DataFrame(data)
        
        # Bar chart grouped
        x = np.arange(len(df['prior_knowledge'].unique()))
        width = 0.35
        
        for i, init in enumerate(df['init_mode'].unique()):
            mask = df['init_mode'] == init
            rates = df[mask]['rate'].values
            ax.bar(x + (i - 0.5) * width, rates, width, label=init.capitalize())
        
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in df['prior_knowledge'].unique()])
        ax.set_xlabel('Prior Knowledge')
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Prior Knowledge × Initialization', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.05)
    
    # 4. Top treatments
    ax = axes[1, 0]
    top_treatments = mech_analysis['rankings']['top_by_convergence'][:10]
    names = [t['treatment'][:35] + '...' if len(t['treatment']) > 35 else t['treatment'] for t in top_treatments]
    rates = [t['conv_rate'] for t in top_treatments]
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(range(len(names)), rates, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Convergence Rate')
    ax.set_title('Top 10 Treatments by Convergence', fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2,
               f'{rate:.1%}', ha='left', va='center', fontsize=8)
    
    # 5. Bottom treatments
    ax = axes[1, 1]
    bottom_treatments = mech_analysis['rankings']['bottom_by_convergence'][:10]
    names = [t['treatment'][:35] + '...' if len(t['treatment']) > 35 else t['treatment'] for t in bottom_treatments]
    rates = [t['conv_rate'] for t in bottom_treatments]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(range(len(names)), rates, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Convergence Rate')
    ax.set_title('Bottom 10 Treatments by Convergence', fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2,
               f'{rate:.1%}', ha='left', va='center', fontsize=8)
    
    # 6. Regret vs Convergence trade-off
    ax = axes[1, 2]
    if 'summary' in results:
        summary = results['summary']
        ax.scatter(summary['both_convergence_rate'], summary['train_total_regret_mean'],
                  alpha=0.5, s=30, c=COLORS['primary'])
        ax.set_xlabel('Convergence Rate')
        ax.set_ylabel('Total Regret (mean)')
        ax.set_title('Convergence vs Regret Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        corr = summary['both_convergence_rate'].corr(summary['train_total_regret_mean'])
        ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
               ha='right', va='top', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mechanism_effects.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_algorithm_deep_dive(results: Dict[str, Any], output_dir: str):
    """Deep dive into algorithm performance."""
    runs = results['runs']
    
    algos = ['greedy', 'ucb', 'thompson', 'exp3', 'etc']
    n_algos = len(algos)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Retailer algorithm performance
    ax = axes[0, 0]
    retailer_stats = runs.groupby('agent_retailer').agg({
        'both_converged': ['mean', 'std'],
        'train_total_regret': 'mean',
    }).round(4)
    retailer_stats.columns = ['conv_mean', 'conv_std', 'regret']
    
    x = np.arange(len(algos))
    bars = ax.bar(x, [retailer_stats.loc[a, 'conv_mean'] if a in retailer_stats.index else 0 for a in algos],
                  yerr=[retailer_stats.loc[a, 'conv_std'] if a in retailer_stats.index else 0 for a in algos],
                  color=[COLORS['algorithms'][a] for a in algos], capsize=5, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Retailer Algorithm Performance', fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # 2. Supplier algorithm performance
    ax = axes[0, 1]
    supplier_stats = runs.groupby('agent_supplier').agg({
        'both_converged': ['mean', 'std'],
        'train_total_regret': 'mean',
    }).round(4)
    supplier_stats.columns = ['conv_mean', 'conv_std', 'regret']
    
    bars = ax.bar(x, [supplier_stats.loc[a, 'conv_mean'] if a in supplier_stats.index else 0 for a in algos],
                  yerr=[supplier_stats.loc[a, 'conv_std'] if a in supplier_stats.index else 0 for a in algos],
                  color=[COLORS['algorithms'][a] for a in algos], capsize=5, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Supplier Algorithm Performance', fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # 3. Regret heatmap by algorithm pair
    ax = axes[1, 0]
    regret_matrix = np.zeros((n_algos, n_algos))
    
    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            mask = (runs['agent_retailer'] == a1) & (runs['agent_supplier'] == a2)
            if mask.sum() > 0:
                regret_matrix[i, j] = runs.loc[mask, 'train_total_regret'].mean()
    
    im = ax.imshow(regret_matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(np.arange(n_algos))
    ax.set_yticks(np.arange(n_algos))
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_yticklabels([a.upper() for a in algos])
    ax.set_xlabel('Supplier Algorithm')
    ax.set_ylabel('Retailer Algorithm')
    ax.set_title('Average Total Regret by Algorithm Pair', fontweight='bold')
    
    for i in range(n_algos):
        for j in range(n_algos):
            ax.text(j, i, f'{regret_matrix[i, j]/1000:.1f}k',
                   ha='center', va='center', fontsize=8,
                   color='white' if regret_matrix[i, j] > regret_matrix.mean() else 'black')
    
    plt.colorbar(im, ax=ax, label='Total Regret')
    
    # 4. Symmetric vs Asymmetric performance
    ax = axes[1, 1]
    symmetric_conv = []
    asymmetric_conv = []
    
    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            mask = (runs['agent_retailer'] == a1) & (runs['agent_supplier'] == a2)
            if mask.sum() > 0:
                rate = runs.loc[mask, 'both_converged'].mean()
                if a1 == a2:
                    symmetric_conv.append(rate)
                else:
                    asymmetric_conv.append(rate)
    
    categories = ['Symmetric\n(same algo)', 'Asymmetric\n(different algos)']
    means = [np.mean(symmetric_conv), np.mean(asymmetric_conv)]
    stds = [np.std(symmetric_conv), np.std(asymmetric_conv)]
    
    bars = ax.bar(categories, means, yerr=stds, color=[COLORS['primary'], COLORS['secondary']],
                  capsize=10, edgecolor='white', linewidth=2)
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Symmetric vs Asymmetric Algorithm Pairs', fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{mean:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_deep_dive.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(results: Dict[str, Any], analyses: Dict[str, Any], output_dir: str):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid specification
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    conv_analysis = analyses['convergence']
    speed_analysis = analyses['speed']
    solution_analysis = analyses['solution']
    mech_analysis = analyses['mechanism']
    
    # 1. Key metrics summary (top row, spans 2 columns)
    ax = fig.add_subplot(gs[0, :2])
    ax.axis('off')
    
    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    KEY FINDINGS SUMMARY                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Total Runs Analyzed:      {conv_analysis['overall']['total_runs']:>10,}                    ║
    ║  Overall Convergence Rate:        {conv_analysis['overall']['both_convergence_rate']:>6.1%}                    ║
    ║  Converged to Central Opt:        {solution_analysis['overall']['to_central_rate']:>6.1%}                    ║
    ║  Converged to Nash Equilibrium:   {solution_analysis['overall']['to_ne_rate']:>6.1%}                    ║
    ║  Mean S1 Convergence Time:     {speed_analysis['overall'].get('s1_conv_time_mean', 0) or 0:>6.0f} rounds              ║
    ║  Mean S2 Convergence Time:     {speed_analysis['overall'].get('s2_conv_time_mean', 0) or 0:>6.0f} rounds              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='center', horizontalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))
    ax.set_title('Executive Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Convergence by cooperation mode
    ax = fig.add_subplot(gs[0, 2])
    if 'cooperation_mode' in conv_analysis['by_mechanism']:
        coop_data = conv_analysis['by_mechanism']['cooperation_mode']
        modes = list(coop_data.keys())
        rates = [coop_data[m]['both_converged'] for m in modes]
        colors = [COLORS['cooperation'].get(m, COLORS['neutral']) for m in modes]
        
        bars = ax.bar(range(len(modes)), rates, color=colors, edgecolor='white')
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m.capitalize() for m in modes], fontsize=9)
        ax.set_ylabel('Conv. Rate')
        ax.set_title('By Cooperation Mode', fontweight='bold')
        ax.set_ylim(0, 1.05)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.1%}', ha='center', fontsize=9)
    
    # 3. Best/Worst algorithm pairs
    ax = fig.add_subplot(gs[0, 3])
    algo_data = conv_analysis['by_algorithm']
    sorted_algos = sorted(algo_data.items(), key=lambda x: x[1]['both_rate'], reverse=True)
    
    top_3 = sorted_algos[:3]
    bottom_3 = sorted_algos[-3:]
    
    labels = [a[0] for a in top_3 + bottom_3]
    rates = [a[1]['both_rate'] for a in top_3 + bottom_3]
    colors = [COLORS['success']] * 3 + [COLORS['secondary']] * 3
    
    bars = ax.barh(range(len(labels)), rates, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Conv. Rate')
    ax.set_title('Top/Bottom Algo Pairs', fontweight='bold')
    ax.invert_yaxis()
    
    # 4. Algorithm heatmap
    ax = fig.add_subplot(gs[1, :2])
    algos = ['greedy', 'ucb', 'thompson', 'exp3', 'etc']
    algo_matrix = np.zeros((len(algos), len(algos)))
    
    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            pair = f'{a1}/{a2}'
            if pair in algo_data:
                algo_matrix[i, j] = algo_data[pair]['both_rate']
    
    im = ax.imshow(algo_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(algos)))
    ax.set_yticks(np.arange(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_yticklabels([a.upper() for a in algos])
    ax.set_xlabel('Supplier Algorithm')
    ax.set_ylabel('Retailer Algorithm')
    ax.set_title('Convergence Rate Heatmap', fontweight='bold')
    
    for i in range(len(algos)):
        for j in range(len(algos)):
            ax.text(j, i, f'{algo_matrix[i, j]:.0%}',
                   ha='center', va='center', fontsize=9,
                   color='white' if algo_matrix[i, j] < 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 5. Factor effects
    ax = fig.add_subplot(gs[1, 2:])
    factors = []
    effects = []
    significances = []
    
    for factor, data in mech_analysis['main_effects'].items():
        if factor in ['agent_retailer', 'agent_supplier']:
            continue
        levels = list(data['levels'].values())
        if len(levels) >= 2:
            conv_rates = [l['conv_mean'] for l in levels]
            effect = max(conv_rates) - min(conv_rates)
            factors.append(factor.replace('_', '\n'))
            effects.append(effect)
            significances.append(data['p_value'] < 0.05)
    
    if factors:
        colors = [COLORS['success'] if sig else COLORS['neutral'] for sig in significances]
        bars = ax.barh(range(len(factors)), effects, color=colors)
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors, fontsize=9)
        ax.set_xlabel('Effect Size (Max - Min Rate)')
        ax.set_title('Factor Importance (red = significant)', fontweight='bold')
    
    # 6. Convergence time distribution
    ax = fig.add_subplot(gs[2, 0])
    runs = results['runs']
    converged = runs[runs['both_converged'] == True]
    
    if len(converged) > 0:
        s1_times = converged['s1_conv_time'].dropna()
        s2_times = converged['s2_conv_time'].dropna()
        
        if len(s1_times) > 0:
            ax.hist(s1_times, bins=20, alpha=0.6, label='S1', color=COLORS['primary'])
        if len(s2_times) > 0:
            ax.hist(s2_times, bins=20, alpha=0.6, label='S2', color=COLORS['secondary'])
        ax.set_xlabel('Convergence Time')
        ax.set_ylabel('Frequency')
        ax.set_title('Conv. Time Distribution', fontweight='bold')
        ax.legend()
    
    # 7. Solution quality
    ax = fig.add_subplot(gs[2, 1])
    labels = ['Converged', 'To Central', 'To Nash']
    rates = [
        conv_analysis['overall']['both_convergence_rate'],
        solution_analysis['overall']['to_central_rate'],
        solution_analysis['overall']['to_ne_rate'],
    ]
    colors = [COLORS['primary'], COLORS['tertiary'], COLORS['secondary']]
    
    bars = ax.bar(labels, rates, color=colors, edgecolor='white')
    ax.set_ylabel('Rate')
    ax.set_title('Solution Quality', fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2 if rates else 1)
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{rate:.1%}', ha='center', fontsize=10)
    
    # 8. Top treatments table
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    top_5 = mech_analysis['rankings']['top_by_convergence'][:5]
    table_data = [[t['treatment'][:40], f"{t['conv_rate']:.1%}", 
                  f"{t['regret']/1000:.1f}k"] for t in top_5]
    
    table = ax.table(cellText=table_data,
                     colLabels=['Treatment', 'Conv. Rate', 'Regret'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Top 5 Treatments', fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: Dict[str, Any], analyses: Dict[str, Any], output_dir: str):
    """Generate comprehensive text report."""
    
    conv = analyses['convergence']
    speed = analyses['speed']
    solution = analyses['solution']
    mech = analyses['mechanism']
    
    report = []
    report.append("=" * 80)
    report.append("SUPPLY CHAIN COORDINATION EXPERIMENT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Results directory: {results.get('results_dir', 'N/A')}")
    
    # Overview
    report.append("\n" + "=" * 80)
    report.append("1. OVERVIEW")
    report.append("=" * 80)
    report.append(f"\nTotal runs analyzed: {conv['overall']['total_runs']:,}")
    report.append(f"Total treatments: {len(results.get('summary', [])):,}")
    
    if 'metadata' in results:
        meta = results['metadata']
        report.append(f"Seeds per treatment: {meta.get('n_seeds', 'N/A')}")
        report.append(f"Convergence window: {meta.get('conv_window', 'N/A')} rounds")
        report.append(f"Convergence threshold: {meta.get('conv_threshold', 'N/A')}")
    
    # Q1: Convergence
    report.append("\n" + "=" * 80)
    report.append("2. DO AGENTS CONVERGE?")
    report.append("=" * 80)
    
    report.append(f"\n2.1 Overall Convergence Rates:")
    report.append(f"    - Retailer (S1) convergence rate: {conv['overall']['s1_convergence_rate']:.1%}")
    report.append(f"    - Supplier (S2) convergence rate: {conv['overall']['s2_convergence_rate']:.1%}")
    report.append(f"    - Both agents converged: {conv['overall']['both_convergence_rate']:.1%}")
    report.append(f"    - Total converged runs: {conv['overall']['converged_runs']:,} / {conv['overall']['total_runs']:,}")
    
    report.append(f"\n2.2 Convergence by Algorithm Pair:")
    sorted_algos = sorted(conv['by_algorithm'].items(), key=lambda x: x[1]['both_rate'], reverse=True)
    for algo, data in sorted_algos:
        report.append(f"    {algo:20s}: {data['both_rate']:.1%} (n={data['n_runs']:,})")
    
    report.append(f"\n2.3 Convergence by Cooperation Mode:")
    if 'cooperation_mode' in conv['by_mechanism']:
        for mode, data in conv['by_mechanism']['cooperation_mode'].items():
            report.append(f"    {mode:15s}: S1={data['s1_converged']:.1%}, S2={data['s2_converged']:.1%}, Both={data['both_converged']:.1%}")
    
    report.append(f"\n2.4 Convergence by Action Grid Size:")
    if 'grid_size' in conv['by_mechanism']:
        grid_order = ['coarse', 'medium', 'fine']
        for grid_size in grid_order:
            if grid_size in conv['by_mechanism']['grid_size']:
                data = conv['by_mechanism']['grid_size'][grid_size]
                # Get the actual grid bounds from the data
                runs = results['runs']
                grid_mask = runs['grid_size'] == grid_size
                if grid_mask.sum() > 0:
                    s_upper = runs[grid_mask]['s_upper'].iloc[0]
                    n_actions = s_upper + 1  # since s_lower is always 0
                    report.append(f"    {grid_size:10s} (0-{s_upper:2d}, {n_actions} actions): S1={data['s1_converged']:.1%}, S2={data['s2_converged']:.1%}, Both={data['both_converged']:.1%}")
    
    report.append(f"\n2.5 Convergence by Prior Knowledge:")
    if 'prior_knowledge' in conv['by_mechanism']:
        for prior, data in conv['by_mechanism']['prior_knowledge'].items():
            report.append(f"    {prior:15s}: S1={data['s1_converged']:.1%}, S2={data['s2_converged']:.1%}, Both={data['both_converged']:.1%}")
    
    report.append(f"\n2.6 Convergence by Initialization Mode:")
    if 'init_mode' in conv['by_mechanism']:
        for init, data in conv['by_mechanism']['init_mode'].items():
            report.append(f"    {init:15s}: S1={data['s1_converged']:.1%}, S2={data['s2_converged']:.1%}, Both={data['both_converged']:.1%}")
    
    # Q2: Convergence Speed
    report.append("\n" + "=" * 80)
    report.append("3. HOW FAST DO THEY CONVERGE?")
    report.append("=" * 80)
    
    if speed['overall'].get('n_converged', 0) > 0:
        report.append(f"\n3.1 Overall Convergence Times (among {speed['overall']['n_converged']:,} converged runs):")
        report.append(f"    - S1 mean: {speed['overall'].get('s1_conv_time_mean', 0):.1f} rounds (std: {speed['overall'].get('s1_conv_time_std', 0):.1f})")
        report.append(f"    - S2 mean: {speed['overall'].get('s2_conv_time_mean', 0):.1f} rounds (std: {speed['overall'].get('s2_conv_time_std', 0):.1f})")
        report.append(f"    - S1 median: {speed['overall'].get('s1_conv_time_median', 0):.1f} rounds")
        report.append(f"    - S2 median: {speed['overall'].get('s2_conv_time_median', 0):.1f} rounds")
        
        report.append(f"\n3.2 Convergence Time by Cooperation Mode:")
        if 'cooperation_mode' in speed['by_mechanism']:
            for mode, data in speed['by_mechanism']['cooperation_mode'].items():
                s1_t = data.get('s1_mean', 0) or 0
                s2_t = data.get('s2_mean', 0) or 0
                report.append(f"    {mode:15s}: S1={s1_t:.1f}, S2={s2_t:.1f} rounds (n={data['n']:,})")
    else:
        report.append("\n    No converged runs to analyze convergence times.")
    
    # Q3: Solution Quality
    report.append("\n" + "=" * 80)
    report.append("4. TO WHAT SOLUTIONS?")
    report.append("=" * 80)
    
    report.append(f"\n4.1 Solution Outcomes (among {solution['overall']['n_converged']:,} converged runs):")
    report.append(f"    - Converged to Central Optimum: {solution['overall']['to_central']:,} ({solution['overall']['to_central_rate']:.1%})")
    report.append(f"    - Converged to Nash Equilibrium: {solution['overall']['to_ne']:,} ({solution['overall']['to_ne_rate']:.1%})")
    report.append(f"    - Runs with NE analysis: {solution['overall']['ne_analyzed']:,}")
    
    if 'to_central' in solution['distance_analysis']:
        d = solution['distance_analysis']['to_central']
        report.append(f"\n4.2 Distance to Central Optimum:")
        report.append(f"    - Mean: {d['mean']:.2f} (std: {d['std']:.2f})")
        report.append(f"    - Median: {d['median']:.2f}")
        report.append(f"    - Range: [{d['min']:.0f}, {d['max']:.0f}]")
    
    if 'deviation_incentives' in solution['distance_analysis']:
        di = solution['distance_analysis']['deviation_incentives']
        report.append(f"\n4.3 Deviation Incentives (stability measure):")
        report.append(f"    - δ1 (retailer): {di.get('delta1_mean', 0):.3f} ± {di.get('delta1_std', 0):.3f}")
        report.append(f"    - δ2 (supplier): {di.get('delta2_mean', 0):.3f} ± {di.get('delta2_std', 0):.3f}")
    
    # Q4: Mechanism Effects
    report.append("\n" + "=" * 80)
    report.append("5. WHICH MECHANISMS INFLUENCE CONVERGENCE?")
    report.append("=" * 80)
    
    report.append(f"\n5.1 Effect Sizes (Cramér's V) for Both-Agent Convergence:")
    report.append(f"    Factor                  Cramér's V    χ²        p-value   Interpretation")
    report.append(f"    " + "-" * 76)
    
    # Interpretation helper
    def interpret_cramers_v(v):
        if v < 0.1: return "negligible"
        elif v < 0.3: return "small"
        elif v < 0.5: return "medium"
        else: return "large"
    
    for factor, data in mech['main_effects'].items():
        if factor in ['agent_retailer', 'agent_supplier']:
            continue
        v = data.get('cramers_v', 0)
        sig = "***" if data['p_value'] < 0.001 else ("**" if data['p_value'] < 0.01 else ("*" if data['p_value'] < 0.05 else "n.s."))
        interp = interpret_cramers_v(v)
        report.append(f"    {factor:20s}    {v:6.3f}    {data['chi2']:>9.1f}   {data['p_value']:.4f} {sig:4s}  {interp}")
    
    report.append(f"\n5.2 Asymmetry Analysis (S1 vs S2 Convergence):")
    s1_rate = conv['overall']['s1_convergence_rate']
    s2_rate = conv['overall']['s2_convergence_rate']
    asymmetry = s1_rate - s2_rate
    report.append(f"    S1 (Retailer) convergence:  {s1_rate:.1%}")
    report.append(f"    S2 (Supplier) convergence:  {s2_rate:.1%}")
    report.append(f"    Asymmetry (S1 - S2):        {asymmetry:+.1%} {'(S1 converges more)' if asymmetry > 0 else '(S2 converges more)'}")
    
    # Show asymmetry by factor
    report.append(f"\n    Asymmetry by Algorithm Pair (all 25, sorted by Both convergence):")
    report.append(f"    {'Algo Pair':20s}  S1      S2      Asymmetry")
    sorted_algos = sorted(conv['by_algorithm'].items(), key=lambda x: x[1]['both_rate'], reverse=True)
    for algo, data in sorted_algos:
        s1 = data['s1_rate']
        s2 = data['s2_rate']
        asym = s1 - s2
        report.append(f"    {algo:20s}  {s1:.1%}   {s2:.1%}   {asym:+.1%}")
    
    report.append(f"\n5.3 Interaction Effects:")
    
    # Prior x Init interaction
    if 'prior_init' in mech['interactions']:
        report.append(f"\n    5.3.1 Prior Knowledge × Initialization Mode:")
        report.append(f"    {'Prior':15s}  {'Init':10s}  Conv. Rate    n")
        for item in mech['interactions']['prior_init']:
            report.append(f"    {item['prior_knowledge']:15s}  {item['init_mode']:10s}  {item['rate']:>8.1%}    {item['n']:>6,}")
    
    # Prior x Grid interaction
    if 'prior_grid' in mech['interactions']:
        report.append(f"\n    5.3.2 Prior Knowledge × Grid Size:")
        report.append(f"    {'Prior':15s}  {'Grid':10s}  Conv. Rate    n")
        for item in mech['interactions']['prior_grid']:
            report.append(f"    {item['prior_knowledge']:15s}  {item['grid_size']:10s}  {item['rate']:>8.1%}    {item['n']:>6,}")
    
    report.append(f"\n5.4 Stratified Analysis:")
    
    # Cooperation effect by algorithm (show ALL 25 algos)
    if 'coop_by_algo' in mech['stratified']:
        report.append(f"\n    5.4.1 Cooperation Mode Effect within Algorithm Pairs (all 25):")
        report.append(f"    (Does cooperation help all algorithms equally?)")
        report.append(f"")
        
        # Get ALL 25 algos sorted by overall convergence
        all_algos = sorted(conv['by_algorithm'].items(), key=lambda x: x[1]['both_rate'], reverse=True)
        all_algo_names = [a[0] for a in all_algos]
        
        for algo in all_algo_names:
            algo_data = [d for d in mech['stratified']['coop_by_algo'] if d['algo_pair'] == algo]
            if algo_data:
                report.append(f"    {algo:20s}:")
                for d in algo_data:
                    report.append(f"      {d['cooperation_mode']:15s}: {d['mean']:>6.1%} (n={int(d['count']):,})")
    
    # Prior effect by grid size
    if 'prior_by_grid' in mech['stratified']:
        report.append(f"\n    5.4.2 Prior Knowledge Effect within Grid Sizes:")
        report.append(f"    (Does prior help more with larger action spaces?)")
        report.append(f"")
        for grid in ['coarse', 'medium', 'fine']:
            grid_data = [d for d in mech['stratified']['prior_by_grid'] if d['grid_size'] == grid]
            if grid_data:
                report.append(f"    {grid:10s}:")
                for d in grid_data:
                    report.append(f"      {d['prior_knowledge']:15s}: {d['mean']:>6.1%} (n={int(d['count']):,})")
    
    report.append(f"\n5.5 Top 10 Treatments by Convergence Rate:")
    for i, t in enumerate(mech['rankings']['top_by_convergence'][:10], 1):
        report.append(f"    {i:2d}. {t['treatment'][:50]:50s} {t['conv_rate']:.1%}")
    
    report.append(f"\n5.6 Bottom 10 Treatments by Convergence Rate:")
    for i, t in enumerate(mech['rankings']['bottom_by_convergence'][:10], 1):
        report.append(f"    {i:2d}. {t['treatment'][:50]:50s} {t['conv_rate']:.1%}")
    
    report.append(f"\n5.7 Top 10 Treatments by Total Regret (lowest = best):")
    for i, t in enumerate(mech['rankings']['top_by_regret'][:10], 1):
        report.append(f"    {i:2d}. {t['treatment'][:50]:50s} {t['regret']:,.0f}")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'detailed_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {report_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Analyze supply chain coordination experiment results')
    parser.add_argument('--results_dir', type=str, default='results_master',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SUPPLY CHAIN COORDINATION EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    # Load and preprocess data
    results = load_results(args.results_dir)
    results['results_dir'] = args.results_dir
    results = preprocess_data(results)
    
    print("\nRunning analyses...")
    
    # Run all analyses
    analyses = {
        'convergence': analyze_convergence_rates(results),
        'speed': analyze_convergence_speed(results),
        'solution': analyze_solution_quality(results),
        'mechanism': analyze_mechanism_effects(results),
    }
    
    print("  - Convergence analysis complete")
    print("  - Speed analysis complete")
    print("  - Solution quality analysis complete")
    print("  - Mechanism effects analysis complete")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_convergence_overview(results, analyses['convergence'], args.output_dir)
    print("  - Convergence overview saved")
    
    plot_convergence_speed(results, analyses['speed'], args.output_dir)
    print("  - Convergence speed saved")

    plot_convergence_by_grid_size(results, analyses['convergence'], analyses['speed'], args.output_dir)
    print("  - Convergence by grid size saved")

    plot_convergence_outcome_by_prior_init(results, analyses['solution'], args.output_dir)
    print("  - Convergence outcome by prior/init saved")
    
    plot_solution_quality(results, analyses['solution'], args.output_dir)
    print("  - Solution quality saved")
    
    plot_mechanism_effects(results, analyses['mechanism'], args.output_dir)
    print("  - Mechanism effects saved")
    
    plot_algorithm_deep_dive(results, args.output_dir)
    print("  - Algorithm deep dive saved")
    
    plot_summary_dashboard(results, analyses, args.output_dir)
    print("  - Summary dashboard saved")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, analyses, args.output_dir)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Total runs: {analyses['convergence']['overall']['total_runs']:,}")
    print(f"Overall convergence rate: {analyses['convergence']['overall']['both_convergence_rate']:.1%}")
    print(f"Central optimum rate: {analyses['solution']['overall']['to_central_rate']:.1%}")
    print(f"Nash equilibrium rate: {analyses['solution']['overall']['to_ne_rate']:.1%}")
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

