import numpy as np
from mesa import Agent


class BanditAgent(Agent):
    def __init__(self, model, role: str):
        super().__init__(model)
        self.role = role
        self._config = model.config
        self.action_space = self._config.action_space()
        self.n_actions = len(self.action_space)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions)
        self.M2 = np.zeros(self.n_actions)
        self.action_idx = self.action = None
        self.reward = self.reward_cum = 0.0

    def _get_scores(self):
        return self.average_reward

    def _update_stats(self, a_idx, r):
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        old_mean = self.average_reward[a_idx]
        self.average_reward[a_idx] += (r - old_mean) / n
        self.M2[a_idx] += (r - old_mean) * (self.average_reward[a_idx] - old_mean)

    def select_action(self):
        raise NotImplementedError

    def update_belief(self):
        self._update_stats(self.action_idx, float(self.model.rewards[self]))


class GreedyAgent(BanditAgent):
    def __init__(self, model, role: str):
        super().__init__(model, role)
        self.eps = self._config.epsilon_at(0)

    def select_action(self):
        if self.model.rng.random() < self.eps:
            self.action_idx = int(self.model.rng.integers(0, self.n_actions))
        else:
            scores = self._get_scores()
            best = np.flatnonzero(scores == scores.max())
            self.action_idx = int(self.model.rng.choice(best))
        self.action = int(self.action_space[self.action_idx])

    def update_belief(self):
        super().update_belief()
        self.eps = self._config.epsilon_at(self.model.steps)


class UcbAgent(BanditAgent):
    def __init__(self, model, role: str):
        super().__init__(model, role)
        self.total_plays = 0

    def select_action(self):
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            self.action_idx = int(self.model.rng.choice(untried))
        else:
            confidence = np.sqrt(2.0 * np.log(self.total_plays + 1) / self.counts)
            ucb = self._get_scores() + confidence
            best = np.flatnonzero(ucb == ucb.max())
            self.action_idx = int(self.model.rng.choice(best))
        self.action = int(self.action_space[self.action_idx])

    def update_belief(self):
        super().update_belief()
        self.total_plays += 1


class ThompsonAgent(Agent):
    def __init__(self, model, role: str):
        super().__init__(model)
        self.role = role
        self.action_space = model.config.action_space()
        self.n_actions = len(self.action_space)
        self.mu = np.zeros(self.n_actions)
        self.kappa = np.ones(self.n_actions)
        self.alpha_param = np.ones(self.n_actions)
        self.beta_param = np.ones(self.n_actions)
        self.action_idx = self.action = None
        self.reward = self.reward_cum = 0.0

    def select_action(self):
        sampled = np.array([
            self.model.rng.normal(self.mu[a], np.sqrt(1.0 / (self.model.rng.gamma(self.alpha_param[a], 1.0 / self.beta_param[a]) + 1e-10) / self.kappa[a]))
            for a in range(self.n_actions)
        ])
        self.action_idx = int(np.argmax(sampled))
        self.action = int(self.action_space[self.action_idx])

    def update_belief(self):
        a, r = self.action_idx, float(self.model.rewards[self])
        kappa_old, mu_old = self.kappa[a], self.mu[a]
        self.kappa[a] = kappa_old + 1
        self.mu[a] = (kappa_old * mu_old + r) / self.kappa[a]
        self.alpha_param[a] += 0.5
        self.beta_param[a] += 0.5 * kappa_old * (r - mu_old)**2 / self.kappa[a]


class Exp3Agent(Agent):
    def __init__(self, model, role: str):
        super().__init__(model)
        self.role = role
        self.action_space = model.config.action_space()
        self.n_actions = len(self.action_space)
        self.gamma = model.config.exp3_gamma
        self.weights = np.ones(self.n_actions)
        self._probs = np.ones(self.n_actions) / self.n_actions
        self.action_idx = self.action = None
        self.reward = self.reward_cum = 0.0
        self.r_min = float("inf")
        self.r_max = float("-inf")
        self._freeze_t = min(50, getattr(model.config, "rounds", 50))

    def select_action(self):
        w_sum = self.weights.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            self.weights[:] = 1.0
            w_sum = self.weights.sum()
        self._probs = (1 - self.gamma) * (self.weights / w_sum) + self.gamma / self.n_actions
        self.action_idx = int(self.model.rng.choice(self.n_actions, p=self._probs))
        self.action = int(self.action_space[self.action_idx])

    def update_belief(self):
        r = float(self.model.rewards[self])
        if self.model.steps < self._freeze_t:
            self.r_min = min(self.r_min, r)
            self.r_max = max(self.r_max, r)
        denom = self.r_max - self.r_min
        if denom > 1e-12:
            r_norm = (r - self.r_min) / denom
        else:
            r_norm = 0.5
        p = self._probs[self.action_idx]
        if p < 1e-12:
            p = 1e-12
        self.weights[self.action_idx] *= np.exp(self.gamma * r_norm / (p * self.n_actions))
        if self.weights.max() > 1e10:
            self.weights /= self.weights.max()


class EtcAgent(BanditAgent):
    def __init__(self, model, role: str):
        super().__init__(model, role)
        self.explore_rounds = self._config.etc_explore_rounds
        self.committed_arm = None
        self._explore_order = list(range(self.n_actions))
        model.rng.shuffle(self._explore_order)
        self._explore_idx = 0

    def select_action(self):
        if self.committed_arm is not None:
            self.action_idx = self.committed_arm
        else:
            while self._explore_idx < len(self._explore_order):
                arm = self._explore_order[self._explore_idx]
                if self.counts[arm] < self.explore_rounds:
                    self.action_idx = arm
                    break
                self._explore_idx += 1
            else:
                scores = self._get_scores()
                best = np.flatnonzero(scores == scores.max())
                self.committed_arm = int(self.model.rng.choice(best))
                self.action_idx = self.committed_arm
        self.action = int(self.action_space[self.action_idx])


AGENT_CLASSES = {"greedy": GreedyAgent, "ucb": UcbAgent, "thompson": ThompsonAgent, "exp3": Exp3Agent, "etc": EtcAgent}


def create_agent(model, agent_type: str, role: str):
    cls = AGENT_CLASSES.get(agent_type.lower())
    if cls is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return cls(model, role)


def initialize_agent_benchmark(agent, opt_action: int, prior_reward: float, prior_strength: int):
    actions = list(agent.action_space)
    if opt_action not in actions:
        return
    idx = actions.index(opt_action)
    
    if isinstance(agent, (GreedyAgent, UcbAgent, EtcAgent)):
        agent.counts[idx] = prior_strength
        agent.average_reward[idx] = prior_reward
        if isinstance(agent, UcbAgent):
            agent.total_plays = prior_strength
    elif isinstance(agent, ThompsonAgent):
        agent.mu[idx] = prior_reward
        agent.kappa[idx] = prior_strength
        agent.alpha_param[idx] = prior_strength / 2
        agent.beta_param[idx] = prior_strength / 2
    elif isinstance(agent, Exp3Agent):
        agent.weights[idx] = np.exp(prior_strength)


def initialize_agent_prior_knowledge(agent, prior_means: np.ndarray, prior_strength: int = 3):
    n = agent.n_actions
    if len(prior_means) != n:
        return
    
    if isinstance(agent, (GreedyAgent, UcbAgent, EtcAgent)):
        agent.counts = np.full(n, prior_strength, dtype=int)
        agent.average_reward = prior_means.copy()
        if isinstance(agent, UcbAgent):
            agent.total_plays = n * prior_strength
    elif isinstance(agent, ThompsonAgent):
        agent.mu = prior_means.copy()
        agent.kappa = np.full(n, float(prior_strength))
        agent.alpha_param = np.full(n, prior_strength / 2.0)
        agent.beta_param = np.full(n, prior_strength / 2.0)
    elif isinstance(agent, Exp3Agent):
        normalized = prior_means - prior_means.min() + 1.0
        agent.weights = normalized / normalized.sum() * n

