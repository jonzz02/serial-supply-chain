import numpy as np
from typing import Tuple, Dict, Any

_benchmark_cache, _payoff_cache, _nash_cache, _prior_cache = {}, {}, {}, {}


def _effective_costs(H1: float, H2: float, cooperation_mode: str, beta: float):
    if cooperation_mode == "cooperative":
        J = H1 + H2
        return J, J
    if cooperation_mode == "partial":
        return H1 + beta * H2, H2 + beta * H1
    return H1, H2  # competitive


def _env_step(rng, I1, I2, B1, B2, U1p, U2p, s1, s2, lam, h1, h2, p_bo, alpha):
    I1 += U1p
    I2 += U2p
    O1, O2 = max(0, s1 - (I1 - B1)), max(0, s2 - (I2 - B2))
    ship = min(I2, B2 + O1)
    I2 -= ship
    B2 = B2 + O1 - ship
    U1, U2 = ship, O2
    D = int(rng.poisson(lam))
    sales = min(I1, B1 + D)
    I1 -= sales
    B1 = B1 + D - sales
    H1 = (h1 + h2) * I1 + alpha * p_bo * B1
    H2 = h2 * (I2 + U1) + (1.0 - alpha) * p_bo * B1
    return I1, I2, B1, B2, U1, U2, float(H1 + H2), float(H1), float(H2)


def _estimate_costs(s1, s2, seed, rounds, warmup, lam, h1, h2, p_bo, alpha):
    rng = np.random.default_rng(seed)
    I1 = I2 = B1 = B2 = U1 = U2 = 0
    sH1 = sH2 = sTot = cnt = 0.0
    for t in range(rounds + warmup):
        I1, I2, B1, B2, U1, U2, _, H1, H2 = _env_step(rng, I1, I2, B1, B2, U1, U2, s1, s2, lam, h1, h2, p_bo, alpha)
        if t >= warmup:
            sH1 += H1
            sH2 += H2
            sTot += H1 + H2
            cnt += 1
    n = max(1, cnt)
    return sH1 / n, sH2 / n, sTot / n


def _estimate_reward(s1, s2, seed, rounds, warmup, lam, h1, h2, p_bo, alpha, role,
                     cooperation_mode: str = "competitive", beta: float = 0.5):
    rng = np.random.default_rng(seed)
    I1 = I2 = B1 = B2 = U1 = U2 = 0
    total = cnt = 0.0
    for t in range(rounds + warmup):
        I1, I2, B1, B2, U1, U2, _, H1, H2 = _env_step(rng, I1, I2, B1, B2, U1, U2, s1, s2, lam, h1, h2, p_bo, alpha)
        if t >= warmup:
            J1, J2 = _effective_costs(H1, H2, cooperation_mode, beta)
            total += (-J1 if role == "retailer" else -J2)
            cnt += 1
    return total / max(1, cnt)


def compute_benchmark(config, seed: int = 42) -> Tuple[int, int, float]:
    key = (config.benchmark_key(), seed)
    if key not in _benchmark_cache:
        actions = config.action_space()
        best_s1 = best_s2 = int(actions[0])
        best_cost = float("inf")
        for s1 in actions:
            for s2 in actions:
                costs = [_estimate_costs(int(s1), int(s2), seed + i, config.benchmark_rounds, config.benchmark_warmup,
                                          config.lam, config.h1, config.h2, config.p_bo, config.alpha)[2]
                         for i in range(config.benchmark_n_seeds)]
                avg = sum(costs) / len(costs)
                if avg < best_cost:
                    best_cost, best_s1, best_s2 = avg, int(s1), int(s2)
        _benchmark_cache[key] = (best_s1, best_s2, best_cost)
    return _benchmark_cache[key]


def compute_payoff_matrices(config, seed: int = 42) -> Dict[str, Any]:
    key = (config.game_key(), seed, config.cooperation_mode, config.cooperation_beta)
    if key in _payoff_cache:
        return _payoff_cache[key]
    
    actions = config.action_space()
    n = len(actions)
    H1 = np.zeros((n, n))
    H2 = np.zeros((n, n))
    Htot = np.zeros((n, n))
    
    for i, s1 in enumerate(actions):
        for j, s2 in enumerate(actions):
            costs = [_estimate_costs(int(s1), int(s2), seed + k * n * n + i * n + j,
                                      config.payoff_rounds, config.payoff_warmup,
                                      config.lam, config.h1, config.h2, config.p_bo, config.alpha)
                     for k in range(config.payoff_n_seeds)]
            H1[i, j] = np.mean([c[0] for c in costs])
            H2[i, j] = np.mean([c[1] for c in costs])
            Htot[i, j] = np.mean([c[2] for c in costs])
    
    mode = config.cooperation_mode
    beta = config.cooperation_beta
    if mode == "cooperative":
        J1 = Htot.copy()
        J2 = Htot.copy()
    elif mode == "partial":
        J1 = H1 + beta * H2
        J2 = H2 + beta * H1
    else:
        J1 = H1.copy()
        J2 = H2.copy()
    
    result = {"actions": actions, "H1": H1, "H2": H2, "Htot": Htot, "J1": J1, "J2": J2}
    _payoff_cache[key] = result
    return result


def compute_best_responses(payoff: Dict[str, Any], tau: float = 0.5) -> Dict[str, Any]:
    """Compute best response correspondences with absolute tolerance tau (cost units)."""
    J1, J2, actions = payoff["J1"], payoff["J2"], payoff["actions"]
    n = len(actions)
    BR1 = [np.flatnonzero(J1[:, j] <= J1[:, j].min() + tau).tolist() for j in range(n)]
    BR2 = [np.flatnonzero(J2[i, :] <= J2[i, :].min() + tau).tolist() for i in range(n)]
    return {"BR1": BR1, "BR2": BR2, "actions": actions}


def compute_pure_nash(config, seed: int = 42) -> Dict[str, Any]:
    key = (config.game_key(), seed, config.cooperation_mode, config.cooperation_beta)
    if key in _nash_cache:
        return _nash_cache[key]
    
    payoff = compute_payoff_matrices(config, seed)
    br = compute_best_responses(payoff)
    actions = payoff["actions"]
    n = len(actions)
    
    ne = [(int(actions[i]), int(actions[j])) for i in range(n) for j in range(n) if i in br["BR1"][j] and j in br["BR2"][i]]
    result = {"ne_set": ne, "ne_count": len(ne), "payoff": payoff, "best_responses": br}
    _nash_cache[key] = result
    return result


def get_deviation_incentives(s1_mode: int, s2_mode: int, config, seed: int = 42) -> Tuple[float, float]:
    payoff = compute_payoff_matrices(config, seed)
    actions = list(payoff["actions"])
    J1, J2 = payoff["J1"], payoff["J2"]
    if s1_mode not in actions or s2_mode not in actions:
        return float('nan'), float('nan')
    i, j = actions.index(s1_mode), actions.index(s2_mode)
    return float(J1[i, j] - J1[:, j].min()), float(J2[i, j] - J2[i, :].min())


def compute_prior_rewards(config, s1_opt: int, s2_opt: int, seed_offset: int = 1000) -> Tuple[float, float]:
    r1 = _estimate_reward(s1_opt, s2_opt, seed_offset, 200, 50,
                          config.lam, config.h1, config.h2, config.p_bo, config.alpha,
                          "retailer", config.cooperation_mode, config.cooperation_beta)
    r2 = _estimate_reward(s1_opt, s2_opt, seed_offset + 1, 200, 50,
                          config.lam, config.h1, config.h2, config.p_bo, config.alpha,
                          "supplier", config.cooperation_mode, config.cooperation_beta)
    return r1, r2


def compute_prior_knowledge_rewards(config, role: str, assumed_other: str = "central", seed: int = 2000) -> np.ndarray:
    key = (config.benchmark_key(), role, assumed_other, seed, config.cooperation_mode, config.cooperation_beta)
    if key in _prior_cache:
        return _prior_cache[key]
    
    actions = config.action_space()
    
    if assumed_other == "central":
        s1_opt, s2_opt, _ = compute_benchmark(config)
        other = s2_opt if role == "retailer" else s1_opt
    else:
        other = int(assumed_other)
    
    rounds = max(50, config.benchmark_rounds // 10)
    warmup = max(10, config.benchmark_warmup // 10)
    
    priors = np.array([
        _estimate_reward(
            int(s) if role == "retailer" else other,
            other if role == "retailer" else int(s),
            seed + idx, rounds, warmup,
            config.lam, config.h1, config.h2, config.p_bo, config.alpha,
            role, config.cooperation_mode, config.cooperation_beta
        )
        for idx, s in enumerate(actions)
    ])
    
    _prior_cache[key] = priors
    return priors


def clear_all_caches():
    _benchmark_cache.clear()
    _payoff_cache.clear()
    _nash_cache.clear()
    _prior_cache.clear()
