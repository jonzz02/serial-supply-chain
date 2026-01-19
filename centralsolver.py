import numpy as np


def _env_step_once(rng, I1, I2, B1, B2, U1_prev, U2_prev,
                   s1_loc, s2_loc, lam, h1, h2, p_bo, alpha):
    """Single period inventory transition."""
    I1 += U1_prev
    I2 += U2_prev

    IP1 = I1 - B1
    IP2 = I2 - B2

    O1 = max(0, int(s1_loc) - int(IP1))
    O2 = max(0, int(s2_loc) - int(IP2))

    ship = min(I2, B2 + O1)
    I2 -= ship
    B2 = B2 + O1 - ship
    U1 = ship
    U2 = O2

    D = int(rng.poisson(lam))
    sales = min(I1, B1 + D)
    I1 -= sales
    B1 = B1 + D - sales

    H1 = (h1 + h2) * I1 + alpha * p_bo * B1
    H2 = h2 * (I2 + U1) + (1.0 - alpha) * p_bo * B1
    total_cost = float(H1 + H2)

    return I1, I2, B1, B2, U1, U2, total_cost, float(H1), float(H2)


def estimate_avg_total_cost(*, s1_loc, s2_loc, seed, rounds, warmup, 
                            lam, h1, h2, p_bo, alpha):
    """Estimate long-run average cost for fixed (s1, s2) via Monte Carlo."""
    rng = np.random.default_rng(seed)
    I1 = I2 = B1 = B2 = U1_prev = U2_prev = 0
    total = 0.0
    count = 0

    for t in range(rounds + warmup):
        I1, I2, B1, B2, U1_prev, U2_prev, c, _, _ = _env_step_once(
            rng, I1, I2, B1, B2, U1_prev, U2_prev,
            s1_loc, s2_loc, lam, h1, h2, p_bo, alpha
        )
        if t >= warmup:
            total += c
            count += 1

    return total / max(1, count)


def estimate_agent_reward(*, s1_loc, s2_loc, seed, rounds, warmup,
                          lam, h1, h2, p_bo, alpha, role: str):
    """
    Estimate average reward (negative cost) for a specific agent role
    at fixed (s1, s2). Used for benchmark initialization priors.
    """
    rng = np.random.default_rng(seed)
    I1 = I2 = B1 = B2 = U1_prev = U2_prev = 0
    total_r = 0.0
    count = 0

    for t in range(rounds + warmup):
        I1, I2, B1, B2, U1_prev, U2_prev, _, H1, H2 = _env_step_once(
            rng, I1, I2, B1, B2, U1_prev, U2_prev,
            s1_loc, s2_loc, lam, h1, h2, p_bo, alpha
        )
        if t >= warmup:
            r = -H1 if role == "retailer" else -H2
            total_r += r
            count += 1

    return total_r / max(1, count)


def compute_supply_optimum_local(*, s_lower, s_upper, s_step=1, base_seed,
                                  rounds, warmup, n_seeds, lam, h1, h2, p_bo, alpha):
    """
    Enumerate (s1, s2) grid and return (s1_opt, s2_opt, ctot_opt).
    Averages cost over n_seeds for robustness.
    """
    best_s1, best_s2 = s_lower, s_lower
    best_cost = float("inf")

    for s1 in range(s_lower, s_upper + 1, s_step):
        for s2 in range(s_lower, s_upper + 1, s_step):
            costs = []
            for i in range(n_seeds):
                c = estimate_avg_total_cost(
                    s1_loc=s1, s2_loc=s2, seed=base_seed + i,
                    rounds=rounds, warmup=warmup,
                    lam=lam, h1=h1, h2=h2, p_bo=p_bo, alpha=alpha,
                )
                costs.append(c)
            avg_cost = sum(costs) / len(costs)
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_s1, best_s2 = s1, s2

    return int(best_s1), int(best_s2), float(best_cost)


# Cache for benchmarks
_benchmark_cache = {}


def compute_benchmark(config, benchmark_seed: int = 42):
    """
    Compute benchmark for a given config, with caching.
    Uses config.benchmark_rounds/warmup/n_seeds for estimation.
    Returns (s1_opt, s2_opt, ctot_opt).
    """
    key = config.config_key()
    
    if key not in _benchmark_cache:
        s1_opt, s2_opt, ctot_opt = compute_supply_optimum_local(
            s_lower=config.s_lower,
            s_upper=config.s_upper,
            s_step=config.s_step,
            base_seed=benchmark_seed,
            rounds=config.benchmark_rounds,
            warmup=config.benchmark_warmup,
            n_seeds=config.benchmark_n_seeds,
            lam=config.lam,
            h1=config.h1,
            h2=config.h2,
            p_bo=config.p_bo,
            alpha=config.alpha,
        )
        _benchmark_cache[key] = (s1_opt, s2_opt, ctot_opt)
    
    return _benchmark_cache[key]


def compute_prior_rewards(config, s1_opt: int, s2_opt: int, seed_offset: int = 1000):
    """
    Estimate prior rewards for benchmark initialization.
    Returns (r1_prior, r2_prior) for retailer and supplier.
    Short rollout (~200 steps) at optimal policy for speed.
    """
    r1 = estimate_agent_reward(
        s1_loc=s1_opt, s2_loc=s2_opt, seed=seed_offset,
        rounds=200, warmup=50,
        lam=config.lam, h1=config.h1, h2=config.h2,
        p_bo=config.p_bo, alpha=config.alpha, role="retailer"
    )
    r2 = estimate_agent_reward(
        s1_loc=s1_opt, s2_loc=s2_opt, seed=seed_offset + 1,
        rounds=200, warmup=50,
        lam=config.lam, h1=config.h1, h2=config.h2,
        p_bo=config.p_bo, alpha=config.alpha, role="supplier"
    )
    return r1, r2


def clear_benchmark_cache():
    """Clear cached benchmarks."""
    _benchmark_cache.clear()
