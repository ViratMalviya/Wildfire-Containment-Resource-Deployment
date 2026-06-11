"""
Deep Q-Network (DQN) Agent — CNN Architecture
===============================================
Addresses the fundamental limitation of tabular Q-learning:

PROBLEM — State Space Explosion:
  A 10x10 grid with 5 cell states has 5^100 ≈ 7.8×10^69 possible raw
  configurations. A tabular agent will almost NEVER visit the same raw
  state twice, so the Q-table cannot generalise — it memorises noise.

SOLUTION — Function Approximation with a CNN:
  A Convolutional Neural Network learns a *function* Q(s, a; θ) that maps
  any grid state to Q-values. The CNN learns spatial fire patterns and
  generalises across shifted/similar fire configurations because convolution
  is translation-equivariant.

Architecture:
  Input  : (5, 10, 10) one-hot grid — 5 channels for 5 cell types
  Conv1  : (5→32, kernel=3, pad=1) + BatchNorm + ReLU
  Conv2  : (32→64, kernel=3, pad=1) + BatchNorm + ReLU
  Conv3  : (64→64, kernel=3, pad=1) + BatchNorm + ReLU
  Flatten: (6400,)
  FC1    : (6400→256) + ReLU + Dropout(0.2)
  FC2    : (256→64) + ReLU
  FC3    : (64→4) — Q-value for each sector action

Key DQN Techniques:
  1. Experience Replay — 10k buffer, random mini-batch sampling
  2. Target Network — frozen copy synced every 500 steps
  3. Gradient Clipping — max norm 1.0 for training stability
  4. Huber Loss — robust to outlier rewards

Resume talking points:
  "Replaced tabular Q-learning (5^100 states, zero generalisation) with a
   CNN DQN in PyTorch: one-hot encoded 10×10 grid input, experience replay,
   and a frozen target network updated every 500 steps."
"""

import os
from collections import deque

import numpy as np

# Graceful import — DQNAgent raises a clear error if torch is missing
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────
EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3
FIREBREAK = 4
NUM_CELL_STATES = 5
GRID_SIZE = 10


# ─── Replay Buffer (no PyTorch dependency) ────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size circular buffer of (s, a, r, s', done) transitions.
    Random sampling breaks temporal correlations for stable NN training.
    """

    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent with CNN, experience replay, and target network.

    Accepts:
      - A compact state tuple (8 ints from WildfireEnv._get_state)
      - A raw (10, 10) numpy grid array

    The compact tuple is reconstructed into an approximate grid for the CNN.
    """

    def __init__(
        self,
        state_size=8,
        action_size=4,
        hidden_dim=128,           # kept for API compatibility — unused by CNN
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=10_000,
        target_update_freq=500,
        grid_size=GRID_SIZE,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DQNAgent. "
                "Install with: pip install torch"
            )

        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grid_size = grid_size
        self._step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build policy and target networks
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def _build_network(self):
        """Build the WildfireCNN as an nn.Sequential."""
        conv_out_size = 64 * self.grid_size * self.grid_size

        return nn.Sequential(
            # Convolutional feature extractor
            nn.Conv2d(NUM_CELL_STATES, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Flatten + fully-connected head
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.action_size),
        )

    def _encode_state(self, state):
        """
        Convert state to a (1, 5, 10, 10) float32 tensor.
        Handles both raw (10,10) grid arrays and compact 8-int tuples.
        """
        if isinstance(state, np.ndarray) and state.shape == (self.grid_size, self.grid_size):
            grid = state
        else:
            grid = self._compact_to_grid(state)

        one_hot = np.zeros(
            (NUM_CELL_STATES, self.grid_size, self.grid_size), dtype=np.float32
        )
        for cell_val in range(NUM_CELL_STATES):
            one_hot[cell_val] = (grid == cell_val).astype(np.float32)

        return torch.FloatTensor(one_hot).unsqueeze(0).to(self.device)

    def _compact_to_grid(self, state):
        """
        Reconstruct an approximate 10×10 grid from the compact 8-int sector tuple.
        Each sector (5×5 quadrant) is filled based on (burn_bin, tree_bin).
        """
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.int8) * TREE
        sector_size = self.grid_size // 2
        sectors = [
            (0, sector_size, 0, sector_size),
            (0, sector_size, sector_size, self.grid_size),
            (sector_size, self.grid_size, 0, sector_size),
            (sector_size, self.grid_size, sector_size, self.grid_size),
        ]
        state_list = list(state)
        for i, (r0, r1, c0, c1) in enumerate(sectors):
            burn_bin = state_list[i * 2] if i * 2 < len(state_list) else 0
            tree_bin = state_list[i * 2 + 1] if i * 2 + 1 < len(state_list) else 2
            if burn_bin == 2:
                grid[r0:r1, c0:c1] = BURNING
            elif burn_bin == 1:
                mid_r, mid_c = (r0 + r1) // 2, (c0 + c1) // 2
                grid[mid_r, mid_c] = BURNING
            else:
                if tree_bin == 0:
                    grid[r0:r1, c0:c1] = BURNED
                elif tree_bin == 1:
                    half = (r1 - r0) // 2
                    grid[r0:r0 + half, c0:c1] = BURNED
        return grid

    def choose_action(self, state, valid_actions=None):
        """Epsilon-greedy using CNN Q-values."""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(self._encode_state(state)).cpu().numpy()[0]
        self.policy_net.train()

        valid_q = {a: q_values[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Store transition, then train on a random mini-batch once the buffer
        has enough samples.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        state_t = torch.cat([self._encode_state(s) for s in states], dim=0)
        next_t = torch.cat([self._encode_state(s) for s in next_states], dim=0)
        action_t = torch.LongTensor(actions).to(self.device)
        reward_t = torch.FloatTensor(rewards).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)

        # Current Q(s, a)
        current_q = self.policy_net(state_t).gather(
            1, action_t.unsqueeze(1)
        ).squeeze(1)

        # Target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_t).max(1)[0]
            target_q = reward_t + self.gamma * max_next_q * (1 - done_t)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Exponential epsilon decay after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self):
        """DQN has no Q-table; return total trainable parameter count."""
        return sum(p.numel() for p in self.policy_net.parameters())

    def save(self, filepath):
        """Save model weights and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "hyperparams": {
                "lr": self.lr,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "action_size": self.action_size,
                "grid_size": self.grid_size,
            },
        }, filepath)
        print(f"[DQN] Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights from disk."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self._step_count = ckpt.get("step_count", 0)
        self.policy_net.eval()
        print(f"[DQN] Model loaded from {filepath} (ε={self.epsilon:.4f})")
