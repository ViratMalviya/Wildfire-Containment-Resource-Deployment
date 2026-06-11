"""
Double Q-Learning Agent — Reducing Maximisation Bias
======================================================
Standard Q-learning suffers from *maximisation bias*: when selecting the
target value, it uses max_a Q(s', a). In stochastic environments this
systematically overestimates Q-values, leading to suboptimal policies.

Double Q-Learning (van Hasselt, 2010) maintains TWO Q-tables:
  - Table A selects the action: a* = argmax_a Q_A(s', a)
  - Table B evaluates it:       value = Q_B(s', a*)

This decoupling removes the positive bias because the action selector and
value estimator are independent estimates.

Update rule (alternating randomly between A and B):
  If updating A:
    a* = argmax_a Q_A(s', a)
    target = r + γ * Q_B(s', a*)
    Q_A(s, a) <- Q_A(s, a) + α * [target - Q_A(s, a)]
  (symmetrical for B)

Resume talking point:
  "Implemented Double Q-Learning to address overestimation bias in vanilla
   Q-learning, a known issue in RL particularly in stochastic environments
   like wildfire spread."
"""

import os
import pickle
from collections import defaultdict

import numpy as np


class DoubleQLearningAgent:
    """
    Double Q-Learning agent to reduce maximisation bias.

    Maintains two Q-tables (A and B) and alternates between them:
    one selects the greedy action, the other evaluates its value.
    This decoupling prevents systematic overestimation of Q-values.
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

        # Two independent Q-tables
        self.q_a = defaultdict(lambda: np.zeros(action_size))
        self.q_b = defaultdict(lambda: np.zeros(action_size))

    def _combined_q(self, state):
        """Average of both tables — used for action selection."""
        return (self.q_a[state] + self.q_b[state]) / 2.0

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy using the combined (averaged) Q-values."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            combined = self._combined_q(state)
            valid_q = {a: combined[a] for a in valid_actions}
            return max(valid_q, key=valid_q.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Double Q-learning update.
        Randomly choose which table to update each step.
        """
        if np.random.random() < 0.5:
            # Update Q_A using Q_B for evaluation
            if done:
                target = reward
            else:
                a_star = int(np.argmax(self.q_a[next_state]))
                target = reward + self.gamma * self.q_b[next_state][a_star]
            self.q_a[state][action] += self.lr * (target - self.q_a[state][action])
        else:
            # Update Q_B using Q_A for evaluation
            if done:
                target = reward
            else:
                b_star = int(np.argmax(self.q_b[next_state]))
                target = reward + self.gamma * self.q_a[next_state][b_star]
            self.q_b[state][action] += self.lr * (target - self.q_b[state][action])

    def decay_epsilon(self):
        """Exponential epsilon decay after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self):
        """Return number of unique states seen (union of both tables)."""
        return len(set(list(self.q_a.keys()) + list(self.q_b.keys())))

    def save(self, filepath):
        """Serialise agent to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "q_a": dict(self.q_a),
            "q_b": dict(self.q_b),
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """Deserialise agent from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_a = defaultdict(
            lambda: np.zeros(self.action_size), data["q_a"]
        )
        self.q_b = defaultdict(
            lambda: np.zeros(self.action_size), data["q_b"]
        )
        self.epsilon = data["epsilon"]
