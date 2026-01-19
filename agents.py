import numpy as np
from mesa import Agent


class GreedyAgent(Agent):
    """ε-greedy bandit agent. Uses Mesa's self.random for reproducible exploration."""

    def __init__(self, model, role: str):
        super().__init__(model)
        self.role = role  # "retailer" or "supplier"
        
        cfg = model.config
        self.action_space = cfg.action_space()
        self.n_actions = len(self.action_space)
        
        # Bandit stats
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)
        
        self._config = cfg
        self.eps = cfg.epsilon_at(0)
        
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        """ε-greedy: random with prob ε, else argmax."""
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)
        else:
            action_idx = int(np.argmax(self.average_reward))
        
        self.action_idx = action_idx
        self.action = int(self.action_space[action_idx])

    def update_belief(self):
        """Incremental mean update + decay epsilon."""
        a_idx = self.action_idx
        r = self.model.rewards[self]
        
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n
        
        self.eps = self._config.epsilon_at(self.model.t + 1)


class UcbAgent(Agent):
    """UCB1 bandit agent. Uses Mesa's self.random for tie-breaking and untried arm selection."""

    def __init__(self, model, role: str):
        super().__init__(model)
        self.role = role
        
        cfg = model.config
        self.action_space = cfg.action_space()
        self.n_actions = len(self.action_space)
        
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)
        self.total_plays = 0
        
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        """UCB1: explore untried arms randomly, then use UCB formula."""
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            # Randomize which untried arm to try (not always first)
            action_idx = int(self.random.choice(untried))
        else:
            # UCB1: avg + sqrt(2 * log(total_plays + 1) / counts)
            log_term = np.log(self.total_plays + 1)
            confidence = np.sqrt((2.0 * log_term) / self.counts)
            ucb = self.average_reward + confidence
            best = np.flatnonzero(ucb == ucb.max())
            action_idx = int(self.random.choice(best))
        
        self.action_idx = action_idx
        self.action = int(self.action_space[action_idx])

    def update_belief(self):
        """Incremental mean update."""
        a_idx = self.action_idx
        r = float(self.model.rewards[self])
        
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.total_plays += 1
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n


class ThompsonAgent(Agent):
    """Thompson Sampling with Normal-Inverse-Gamma conjugate prior."""

    def __init__(self, model, role: str):
        super().__init__(model)
        self.role = role
        
        cfg = model.config
        self.action_space = cfg.action_space()
        self.n_actions = len(self.action_space)
        
        # NIG prior params: (mu, kappa, alpha, beta) per arm
        self.mu = np.zeros(self.n_actions)
        self.kappa = np.ones(self.n_actions)
        self.alpha_param = np.ones(self.n_actions)
        self.beta_param = np.ones(self.n_actions)
        
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        """Thompson sampling: sample from posterior and pick best arm."""
        rng = self.model.algo_rng
        sampled_means = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            precision = rng.gamma(self.alpha_param[a], 1.0 / self.beta_param[a])
            variance = 1.0 / (precision + 1e-10)
            std = np.sqrt(variance / self.kappa[a])
            sampled_means[a] = rng.normal(self.mu[a], std)
        action_idx = int(np.argmax(sampled_means))
        
        self.action_idx = action_idx
        self.action = int(self.action_space[action_idx])

    def update_belief(self):
        """Bayesian update for Normal-Inverse-Gamma."""
        a_idx = self.action_idx
        r = float(self.model.rewards[self])
        
        kappa_old = self.kappa[a_idx]
        mu_old = self.mu[a_idx]
        alpha_old = self.alpha_param[a_idx]
        beta_old = self.beta_param[a_idx]
        
        kappa_new = kappa_old + 1
        mu_new = (kappa_old * mu_old + r) / kappa_new
        alpha_new = alpha_old + 0.5
        beta_new = beta_old + 0.5 * kappa_old * (r - mu_old)**2 / kappa_new
        
        self.kappa[a_idx] = kappa_new
        self.mu[a_idx] = mu_new
        self.alpha_param[a_idx] = alpha_new
        self.beta_param[a_idx] = beta_new


# Agent factory
AGENT_CLASSES = {
    "greedy": GreedyAgent,
    "ucb": UcbAgent,
    "thompson": ThompsonAgent,
}


def create_agent(model, agent_type: str, role: str):
    """Create a single agent of the specified type."""
    cls = AGENT_CLASSES.get(agent_type.lower())
    if cls is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return cls(model, role)


def initialize_agent_benchmark(agent, opt_action: int, prior_reward: float, prior_strength: int):
    """
    Warm-start an agent's belief with a prior at the benchmark action.
    Sets pseudo-counts and prior mean for the optimal arm.
    """
    action_space = list(agent.action_space)
    if opt_action not in action_space:
        return  # benchmark action not in grid
    
    idx = action_space.index(opt_action)
    
    if isinstance(agent, (GreedyAgent, UcbAgent)):
        agent.counts[idx] = prior_strength
        agent.average_reward[idx] = prior_reward
        if isinstance(agent, UcbAgent):
            agent.total_plays = prior_strength
    elif isinstance(agent, ThompsonAgent):
        # Set informative prior at benchmark arm
        agent.mu[idx] = prior_reward
        agent.kappa[idx] = prior_strength
        agent.alpha_param[idx] = prior_strength / 2
        agent.beta_param[idx] = prior_strength / 2
