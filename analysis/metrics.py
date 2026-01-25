import numpy as np
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional


def _volatility(actions: np.ndarray) -> float:
    if len(actions) <= 1:
        return 0.0
    return float(np.sum(actions[1:] != actions[:-1])) / (len(actions) - 1)


def compute_convergence(actions: np.ndarray, window: int = 50, threshold: float = 0.9) -> Dict[str, Any]:
    n = len(actions)
    if n < window:
        return {"converged": False, "convergence_time": None, "final_action": int(actions[-1]) if n > 0 else None,
                "final_mode": None, "volatility": _volatility(actions)}
    
    last = actions[-window:]
    counter = Counter(last)
    mode_action, mode_count = counter.most_common(1)[0]
    converged = (mode_count / window) >= threshold
    
    conv_time = None
    if converged:
        for t in range(n - window, -1, -1):
            c = Counter(actions[t:t + window])
            if c.most_common(1)[0][1] / window < threshold:
                conv_time = t + 1
                break
        if conv_time is None:
            conv_time = 0
    
    return {"converged": converged, "convergence_time": conv_time, "final_action": int(actions[-1]),
            "final_mode": int(mode_action), "volatility": _volatility(actions)}


def compute_run_metrics(model, ctot_opt: float, train_rounds: int, warmup: int = 0,
                        conv_window: int = 50, conv_threshold: float = 0.9,
                        s1_opt: Optional[int] = None, s2_opt: Optional[int] = None,
                        ne_set: Optional[List[Tuple[int, int]]] = None, config=None) -> Dict[str, Any]:
    df = model.datacollector.get_model_vars_dataframe()
    
    total_costs = df["Total Cost"].values
    s1_actions = df["S1"].values.astype(int)
    s2_actions = df["S2"].values.astype(int)
    
    start = min(warmup, train_rounds)
    costs, s1, s2 = total_costs[start:train_rounds], s1_actions[start:train_rounds], s2_actions[start:train_rounds]
    
    per_round = costs - ctot_opt
    cumulative = np.cumsum(per_round)
    
    eff_window = min(conv_window, len(s1))
    if eff_window < 5:
        conv_s1 = {"converged": False, "convergence_time": None, "final_action": None, "final_mode": None, "volatility": _volatility(s1)}
        conv_s2 = {"converged": False, "convergence_time": None, "final_action": None, "final_mode": None, "volatility": _volatility(s2)}
    else:
        conv_s1 = compute_convergence(s1, eff_window, conv_threshold)
        conv_s2 = compute_convergence(s2, eff_window, conv_threshold)
    
    both = conv_s1["converged"] and conv_s2["converged"]
    s1_mode, s2_mode = conv_s1["final_mode"], conv_s2["final_mode"]
    
    to_central = to_ne = False
    delta1 = delta2 = float('nan')
    dist = None
    dist_nash = None
    
    if both and s1_mode is not None and s2_mode is not None:
        if s1_opt is not None and s2_opt is not None:
            to_central = (s1_mode == s1_opt and s2_mode == s2_opt)
            dist = abs(s1_mode - s1_opt) + abs(s2_mode - s2_opt)
        
        is_rn = config is None or getattr(config, 'utility_mode', 'risk_neutral') == 'risk_neutral'
        if is_rn:
            if ne_set:
                to_ne = (s1_mode, s2_mode) in ne_set
                dist_nash = min([abs(s1_mode - ne[0]) + abs(s2_mode - ne[1]) for ne in ne_set])
            if config:
                from .centralsolver import get_deviation_incentives
                delta1, delta2 = get_deviation_incentives(s1_mode, s2_mode, config)
    
    s1_ct = conv_s1["convergence_time"]
    s2_ct = conv_s2["convergence_time"]
    
    return {
        "train_mean_cost": float(np.mean(costs)) if len(costs) > 0 else 0.0,
        "train_total_regret": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        "train_mean_regret": float(np.mean(per_round)) if len(per_round) > 0 else 0.0,
        "s1_converged": conv_s1["converged"], "s1_conv_time": s1_ct + start if s1_ct is not None else None,
        "s1_final": conv_s1["final_action"], "s1_mode": s1_mode, "s1_volatility": conv_s1["volatility"],
        "s2_converged": conv_s2["converged"], "s2_conv_time": s2_ct + start if s2_ct is not None else None,
        "s2_final": conv_s2["final_action"], "s2_mode": s2_mode, "s2_volatility": conv_s2["volatility"],
        "both_converged": both, "converged_to_central": to_central, "converged_to_ne": to_ne,
        "delta1": delta1, "delta2": delta2, "distance_to_central": dist, "distance_to_nash": dist_nash,
        "cum_reward_retailer": float(model.retailer.reward_cum),
        "cum_reward_supplier": float(model.supplier.reward_cum),
        "_total_costs": costs, "_s1_actions": s1, "_s2_actions": s2, "_cumulative_regret": cumulative,
    }


def aggregate_metrics(run_list: list) -> Dict[str, Any]:
    n = len(run_list)
    if n == 0:
        return {}
    
    agg = {}
    for key in ["train_mean_cost", "train_total_regret", "train_mean_regret", "s1_volatility", "s2_volatility",
                "cum_reward_retailer", "cum_reward_supplier"]:
        vals = [m[key] for m in run_list if m.get(key) is not None]
        if vals:
            agg[f"{key}_mean"], agg[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
            agg[f"{key}_min"], agg[f"{key}_max"] = float(np.min(vals)), float(np.max(vals))
    
    agg["s1_convergence_rate"] = np.mean([m["s1_converged"] for m in run_list])
    agg["s2_convergence_rate"] = np.mean([m["s2_converged"] for m in run_list])
    agg["both_convergence_rate"] = np.mean([m["both_converged"] for m in run_list])
    agg["converged_to_central_rate"] = float(np.mean([m.get("converged_to_central", False) for m in run_list]))
    agg["converged_to_ne_rate"] = float(np.mean([m.get("converged_to_ne", False) for m in run_list]))
    
    for key, name in [("delta1", "delta1"), ("delta2", "delta2")]:
        vals = [m[key] for m in run_list if m.get("both_converged") and not np.isnan(m.get(key, float('nan')))]
        if vals:
            agg[f"{name}_mean"], agg[f"{name}_std"] = float(np.mean(vals)), float(np.std(vals))
    
    dist = [m.get("distance_to_central") for m in run_list if m.get("both_converged") and m.get("distance_to_central") is not None]
    if dist:
        agg["distance_to_central_mean"], agg["distance_to_central_std"] = float(np.mean(dist)), float(np.std(dist))
    
    dist_nash = [m.get("distance_to_nash") for m in run_list if m.get("both_converged") and m.get("distance_to_nash") is not None]
    if dist_nash:
        agg["distance_to_nash_mean"], agg["distance_to_nash_std"] = float(np.mean(dist_nash)), float(np.std(dist_nash))
    
    for role in ["s1", "s2"]:
        times = [m[f"{role}_conv_time"] for m in run_list if m[f"{role}_conv_time"] is not None]
        if times:
            agg[f"{role}_conv_time_mean"], agg[f"{role}_conv_time_std"] = float(np.mean(times)), float(np.std(times))
            agg[f"{role}_conv_time_min"], agg[f"{role}_conv_time_max"] = float(np.min(times)), float(np.max(times))
        
        modes = [m[f"{role}_mode"] for m in run_list if m[f"{role}_mode"] is not None]
        if modes:
            agg[f"{role}_final_mode"] = int(Counter(modes).most_common(1)[0][0])
    
    agg["n_seeds"] = n
    return agg
