import numpy as np
from collections import Counter
from typing import Dict, Any


def compute_convergence(actions: np.ndarray, window: int = 50, 
                        threshold: float = 0.9) -> Dict[str, Any]:
    """
    Check convergence: in the last W rounds, most frequent action >= threshold.
    Returns dict with converged flag, time, final action, volatility.
    Note: convergence_time is relative to the input array (0-indexed).
    """
    n = len(actions)
    
    if n < window:
        return {
            "converged": False,
            "convergence_time": None,
            "final_action": int(actions[-1]) if n > 0 else None,
            "final_mode": None,
            "volatility": _action_volatility(actions),
        }
    
    last_window = actions[-window:]
    counter = Counter(last_window)
    mode_action, mode_count = counter.most_common(1)[0]
    converged = (mode_count / window) >= threshold
    
    conv_time = None
    if converged:
        for t in range(n - window, -1, -1):
            w = actions[t:t + window]
            c = Counter(w)
            m_act, m_cnt = c.most_common(1)[0]
            if (m_cnt / window) < threshold:
                conv_time = t + 1
                break
        if conv_time is None:
            conv_time = 0
    
    return {
        "converged": converged,
        "convergence_time": conv_time,
        "final_action": int(actions[-1]),
        "final_mode": int(mode_action),
        "volatility": _action_volatility(actions),
    }


def _action_volatility(actions: np.ndarray) -> float:
    """Count action changes / total rounds."""
    if len(actions) <= 1:
        return 0.0
    changes = np.sum(actions[1:] != actions[:-1])
    return float(changes) / (len(actions) - 1)


def compute_regret_metrics(total_costs: np.ndarray, 
                           ctot_opt: float) -> Dict[str, Any]:
    """Compute per-round and cumulative regret vs system optimal."""
    per_round = total_costs - ctot_opt
    cumulative = np.cumsum(per_round)
    
    return {
        "per_round_regret": per_round,
        "cumulative_regret": cumulative,
        "total_regret": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        "mean_regret": float(np.mean(per_round)) if len(per_round) > 0 else 0.0,
    }


def compute_run_metrics(model, ctot_opt: float, train_rounds: int,
                        warmup: int = 0,
                        conv_window: int = 50, 
                        conv_threshold: float = 0.9) -> Dict[str, Any]:
    """
    Compute all metrics from a completed model run.
    Warmup-aware: metrics computed on rounds [warmup, train_rounds).
    Convergence times are reported in original (absolute) round index.
    """
    df = model.datacollector.get_model_vars_dataframe()
    
    total_costs = df["Total Cost"].values
    s1_actions = df["S1"].values.astype(int)
    s2_actions = df["S2"].values.astype(int)
    
    # Slice to post-warmup training data
    start = min(warmup, train_rounds)
    post_warmup_costs = total_costs[start:train_rounds]
    post_warmup_s1 = s1_actions[start:train_rounds]
    post_warmup_s2 = s2_actions[start:train_rounds]
    
    # Robust convergence window
    effective_window = min(conv_window, len(post_warmup_s1))
    
    # Training metrics (post-warmup)
    train_regret = compute_regret_metrics(post_warmup_costs, ctot_opt)
    
    # Skip convergence check if window too small for meaningful results
    if effective_window < 5:
        conv_s1 = {"converged": False, "convergence_time": None, "final_action": None,
                   "final_mode": None, "volatility": _action_volatility(post_warmup_s1)}
        conv_s2 = {"converged": False, "convergence_time": None, "final_action": None,
                   "final_mode": None, "volatility": _action_volatility(post_warmup_s2)}
    else:
        conv_s1 = compute_convergence(post_warmup_s1, effective_window, conv_threshold)
        conv_s2 = compute_convergence(post_warmup_s2, effective_window, conv_threshold)
    both_converged = conv_s1["converged"] and conv_s2["converged"]
    
    # Convert convergence times to absolute round index (add warmup back)
    s1_conv_time_abs = conv_s1["convergence_time"]
    s2_conv_time_abs = conv_s2["convergence_time"]
    if s1_conv_time_abs is not None:
        s1_conv_time_abs += start
    if s2_conv_time_abs is not None:
        s2_conv_time_abs += start
    
    result = {
        # Training costs (post-warmup)
        "train_mean_cost": float(np.mean(post_warmup_costs)) if len(post_warmup_costs) > 0 else 0.0,
        "train_total_regret": train_regret["total_regret"],
        "train_mean_regret": train_regret["mean_regret"],
        
        # Convergence S1 (retailer) - times in absolute round index
        "s1_converged": conv_s1["converged"],
        "s1_conv_time": s1_conv_time_abs,
        "s1_final": conv_s1["final_action"],
        "s1_mode": conv_s1["final_mode"],
        "s1_volatility": conv_s1["volatility"],
        
        # Convergence S2 (supplier) - times in absolute round index
        "s2_converged": conv_s2["converged"],
        "s2_conv_time": s2_conv_time_abs,
        "s2_final": conv_s2["final_action"],
        "s2_mode": conv_s2["final_mode"],
        "s2_volatility": conv_s2["volatility"],
        
        # Joint
        "both_converged": both_converged,
        
        # Cumulative rewards (full simulation, not post-warmup)
        "cum_reward_retailer": float(model.retailer.reward_cum),
        "cum_reward_supplier": float(model.supplier.reward_cum),
        
        # Raw timeseries (post-warmup, for plots)
        "_total_costs": post_warmup_costs,
        "_s1_actions": post_warmup_s1,
        "_s2_actions": post_warmup_s2,
        "_cumulative_regret": train_regret["cumulative_regret"],
    }
    
    return result


def aggregate_metrics(run_metrics_list: list) -> Dict[str, Any]:
    """Aggregate metrics across multiple seeds with mean/std/min/max."""
    n = len(run_metrics_list)
    if n == 0:
        return {}
    
    # Scalars to aggregate with mean/std/min/max
    scalar_keys = [
        "train_mean_cost", "train_total_regret", "train_mean_regret",
        "s1_volatility", "s2_volatility",
        "cum_reward_retailer", "cum_reward_supplier",
    ]
    
    agg = {}
    for key in scalar_keys:
        vals = [m[key] for m in run_metrics_list if m.get(key) is not None]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_min"] = float(np.min(vals))
            agg[f"{key}_max"] = float(np.max(vals))
    
    # Convergence rates (proportion)
    agg["s1_convergence_rate"] = np.mean([m["s1_converged"] for m in run_metrics_list])
    agg["s2_convergence_rate"] = np.mean([m["s2_converged"] for m in run_metrics_list])
    agg["both_convergence_rate"] = np.mean([m["both_converged"] for m in run_metrics_list])
    
    # Convergence times with mean/std/min/max
    s1_times = [m["s1_conv_time"] for m in run_metrics_list if m["s1_conv_time"] is not None]
    s2_times = [m["s2_conv_time"] for m in run_metrics_list if m["s2_conv_time"] is not None]
    
    if s1_times:
        agg["s1_conv_time_mean"] = float(np.mean(s1_times))
        agg["s1_conv_time_std"] = float(np.std(s1_times))
        agg["s1_conv_time_min"] = float(np.min(s1_times))
        agg["s1_conv_time_max"] = float(np.max(s1_times))
    if s2_times:
        agg["s2_conv_time_mean"] = float(np.mean(s2_times))
        agg["s2_conv_time_std"] = float(np.std(s2_times))
        agg["s2_conv_time_min"] = float(np.min(s2_times))
        agg["s2_conv_time_max"] = float(np.max(s2_times))
    
    # Final action distributions
    s1_finals = [m["s1_final"] for m in run_metrics_list if m["s1_final"] is not None]
    s2_finals = [m["s2_final"] for m in run_metrics_list if m["s2_final"] is not None]
    
    if s1_finals:
        agg["s1_final_mode"] = int(Counter(s1_finals).most_common(1)[0][0])
    if s2_finals:
        agg["s2_final_mode"] = int(Counter(s2_finals).most_common(1)[0][0])
    
    agg["n_seeds"] = n
    
    return agg
