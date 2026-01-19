from config import DEFAULT_CONFIG

# Export defaults for backwards compatibility
H1 = DEFAULT_CONFIG.h1
H2 = DEFAULT_CONFIG.h2
P_BO = DEFAULT_CONFIG.p_bo
ALPHA = DEFAULT_CONFIG.alpha
LAM = DEFAULT_CONFIG.lam
S_LOWER = DEFAULT_CONFIG.s_lower
S_UPPER = DEFAULT_CONFIG.s_upper
EPS_START = DEFAULT_CONFIG.eps_start
EPS_END = DEFAULT_CONFIG.eps_end
ROUNDS = DEFAULT_CONFIG.rounds
WARMUP = DEFAULT_CONFIG.warmup
SEED = 42

def action_space():
    return DEFAULT_CONFIG.action_space()

def epsilon_at(t, rounds):
    return DEFAULT_CONFIG.epsilon_at(t)

def sample_demand(rng):
    return int(rng.poisson(DEFAULT_CONFIG.lam))

# Benchmark values - computed lazily on first access
_benchmark_cache = {}

def get_benchmark(config=None):
    """Get benchmark for config, computing if needed."""
    from centralsolver import compute_benchmark
    cfg = config or DEFAULT_CONFIG
    return compute_benchmark(cfg)

# For backwards compatibility, compute on first access
S1_OPT_LOC = None
S2_OPT_LOC = None
CTOT_OPT = None

def _init_benchmark():
    global S1_OPT_LOC, S2_OPT_LOC, CTOT_OPT
    if S1_OPT_LOC is None:
        S1_OPT_LOC, S2_OPT_LOC, CTOT_OPT = get_benchmark()
