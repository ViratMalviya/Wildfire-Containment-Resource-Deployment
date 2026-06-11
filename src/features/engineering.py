"""
Feature Engineering — Wildfire State Representation
=====================================================
Transforms raw environment state (compact tuple or grid array) into
richer, domain-informed feature vectors.

Two classes:
  1. FeatureExtractor — derives semantic domain features from the compact
     8-integer sector state tuple. Useful for interpretable ML, feature
     importance analysis, and augmenting tabular agents.

  2. StateEncoder — converts the raw 10×10 grid into tensor formats
     suitable for deep learning (CNN one-hot input or flat normalised).

Resume talking point:
  "Implemented a feature engineering layer that derives domain-specific
   metrics (fire pressure, containment ratio, resource urgency) from raw
   simulator state — enabling both tabular and deep RL approaches to
   benefit from expert-crafted spatial features."
"""

import numpy as np


# ─── Cell state constants ──────────────────────────────────────────────────
EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3
FIREBREAK = 4
NUM_CELL_STATES = 5


class FeatureExtractor:
    """
    Extracts semantic features from the compact 8-integer sector state tuple.

    The compact state encodes (burn_bin, tree_bin) per sector:
      - burn_bin: 0=none, 1=1-3 cells, 2=4+ cells
      - tree_bin: 0=<30% trees, 1=30-60%, 2=>60% trees

    This extractor derives higher-level metrics useful for understanding
    and characterising the fire situation.
    """

    def __init__(self, grid_size=10, num_sectors=4):
        self.grid_size = grid_size
        self.num_sectors = num_sectors

    def extract(self, state, step=0):
        """
        Extract a rich feature dictionary and flat feature vector from
        the compact sector state tuple.

        Args:
            state: tuple of 8 ints — (burn_bin, tree_bin) × 4 sectors
            step:  current timestep (for temporal features)

        Returns:
            features (dict): named domain features
            vector (np.ndarray): flat numerical vector for ML use
        """
        state_list = list(state)

        # Per-sector burn and tree bins
        burn_bins = [state_list[i * 2] for i in range(self.num_sectors)]
        tree_bins = [state_list[i * 2 + 1] for i in range(self.num_sectors)]

        # ── Domain Features ─────────────────────────────────────────────

        # Global fire pressure: weighted sum of burn bins (0-2 per sector)
        global_fire_pressure = sum(burn_bins) / (self.num_sectors * 2.0)

        # Containment ratio: proportion of sectors with no fire
        sectors_clear = sum(1 for b in burn_bins if b == 0)
        containment_ratio = sectors_clear / self.num_sectors

        # Forest health: normalised average tree bin
        avg_tree_density = sum(tree_bins) / (self.num_sectors * 2.0)

        # Active fire count: sectors with any burning (bin > 0)
        active_fire_sectors = sum(1 for b in burn_bins if b > 0)

        # Critical sectors: burning AND low trees (danger of total loss)
        critical_sectors = sum(
            1 for b, t in zip(burn_bins, tree_bins) if b > 0 and t == 0
        )

        # Resource urgency: are any sectors at maximum fire (bin=2)?
        max_fire_sectors = sum(1 for b in burn_bins if b == 2)

        # Fire spread risk: sectors with fire adjacent to high tree density
        spread_risk = sum(
            1 for b, t in zip(burn_bins, tree_bins) if b > 0 and t == 2
        )

        # Temporal decay feature: normalised step counter
        step_normalised = min(step / 50.0, 1.0)

        # Per-sector combined urgency scores (burn × remaining_fuel)
        sector_urgency = [
            b * (2 - t) for b, t in zip(burn_bins, tree_bins)
        ]
        max_urgency = max(sector_urgency) if sector_urgency else 0.0
        urgency_sector = int(np.argmax(sector_urgency))  # which sector is worst

        features = {
            "global_fire_pressure": round(global_fire_pressure, 4),
            "containment_ratio": round(containment_ratio, 4),
            "avg_tree_density": round(avg_tree_density, 4),
            "active_fire_sectors": active_fire_sectors,
            "critical_sectors": critical_sectors,
            "max_fire_sectors": max_fire_sectors,
            "spread_risk": spread_risk,
            "step_normalised": round(step_normalised, 4),
            "max_urgency_score": max_urgency,
            "urgency_sector": urgency_sector,
            "sector_burn_bins": burn_bins,
            "sector_tree_bins": tree_bins,
            "sector_urgency_scores": sector_urgency,
        }

        # Flat vector for ML compatibility (excludes list-type features)
        vector = np.array([
            global_fire_pressure,
            containment_ratio,
            avg_tree_density,
            active_fire_sectors / self.num_sectors,
            critical_sectors / self.num_sectors,
            max_fire_sectors / self.num_sectors,
            spread_risk / self.num_sectors,
            step_normalised,
            max_urgency / 4.0,           # normalised (max possible = 2*2)
            urgency_sector / self.num_sectors,
        ] + [b / 2.0 for b in burn_bins]    # normalised burn bins
          + [t / 2.0 for t in tree_bins],   # normalised tree bins
            dtype=np.float32,
        )

        return features, vector


class StateEncoder:
    """
    Encodes the raw 10×10 grid into formats suitable for deep learning.

    Two encodings:
      1. one_hot_grid: (5, 10, 10) tensor — each cell state is a channel.
         This is the standard input format for the CNN DQN.

      2. flat_normalised: (100,) vector — grid flattened and normalised
         to [0, 1]. Simpler format for MLP-based agents.
    """

    def __init__(self, grid_size=10, num_cell_states=NUM_CELL_STATES):
        self.grid_size = grid_size
        self.num_cell_states = num_cell_states

    def one_hot_grid(self, grid):
        """
        Convert a (10, 10) integer grid to a (5, 10, 10) one-hot array.

        Each channel c contains 1.0 where grid == c, 0.0 elsewhere.
        This is the input format for the WildfireCNN.

        Args:
            grid: np.ndarray of shape (10, 10) with values in {0,1,2,3,4}

        Returns:
            np.ndarray of shape (5, 10, 10) dtype float32
        """
        one_hot = np.zeros(
            (self.num_cell_states, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        for cell_val in range(self.num_cell_states):
            one_hot[cell_val] = (grid == cell_val).astype(np.float32)
        return one_hot

    def flat_normalised(self, grid):
        """
        Flatten the (10, 10) grid and normalise to [0, 1].

        cell_val / (num_cell_states - 1) maps {0,1,2,3,4} → {0, 0.25, 0.5, 0.75, 1.0}

        Args:
            grid: np.ndarray of shape (10, 10) with values in {0,1,2,3,4}

        Returns:
            np.ndarray of shape (100,) dtype float32
        """
        return (grid.flatten().astype(np.float32) / (self.num_cell_states - 1))

    def augment(self, grid):
        """
        Data augmentation: return all 4 rotations and 4 reflections of the grid.
        Useful for DQN training to improve sample efficiency.

        Returns:
            list of 8 np.ndarray, each shape (10, 10)
        """
        augmented = []
        current = grid.copy()
        for _ in range(4):
            augmented.append(current.copy())
            augmented.append(np.fliplr(current).copy())
            current = np.rot90(current)
        return augmented
