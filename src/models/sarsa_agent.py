"""
SARSA Agent — On-Policy Tabular RL
=====================================
SARSA (State-Action-Reward-State-Action) is an on-policy TD algorithm.
Unlike Q-learning (off-policy), SARSA updates using the *actual* next action
chosen by the policy, not the greedy max. This makes it more conservative
and safer in stochastic environments.

Update rule:
  Q(s, a) <- Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]

where a' is the ACTUAL next action (not argmax), making it on-policy.

Why SARSA vs Q-Learning?
  Q-learning is more aggressive (optimistic) — it assumes the best future
  action. SARSA is more cautious — it accounts for the agent's own
  exploration noise. For safety-critical scenarios like wildfire containment,
  this conservatism can be beneficial.
"""

import os
import pickle
from collections import defaultdict

import numpy as np


class SARSAAgent:
    """
    Tabular SARSA agent with epsilon-greedy exploration.

    On-policy: updates using the action actually taken next,
    not the theoretical maximum — safer in stochastic environments.
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Sparse Q-table
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy action selection (same as Q-learning)."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            valid_q = {a: q_values[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        SARSA on-policy update:
          Q(s, a) <- Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]

        Note: next_action is the action the agent ACTUALLY chose (not argmax),
        which is the key difference from Q-learning.
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]

        self.q_table[state][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        """Exponential epsilon decay after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self):
        """Return the number of unique states in the Q-table."""
        return len(self.q_table)

    def save(self, filepath):
        """Serialise agent state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """Deserialise agent state from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size), data["q_table"]
        )
        self.epsilon = data["epsilon"]
