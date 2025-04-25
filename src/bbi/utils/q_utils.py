"""Utility functions for Q-learning operations."""

from itertools import product
from typing import List, Tuple

import numpy as np


def q_bounding_box(
    Q: np.ndarray, low_state: Tuple[int, ...], up_state: Tuple[int, ...]
) -> Tuple[float, float, List[int]]:
    """Compute min and max greedy Q-values in a bounding box.

    Args:
        Q (np.ndarray): Q-value table.
        low_state (Tuple[int, ...]): Lower bound of the box.
        up_state (Tuple[int, ...]): Upper bound of the box.

    Returns:
        Tuple[float, float, List[int]]: Minimum greedy Q-value, maximum greedy Q-value, and union of best actions across the box.
    """
    dims = len(low_state)
    bounds = [
        (np.minimum(low_state[i], up_state[i]), np.maximum(low_state[i], up_state[i])) for i in range(dims)
    ]
    ranges = [range(lower, upper + 1) for lower, upper in bounds]
    low_actions, up_actions = [], []
    q_vals = []

    for state in product(*ranges):
        q = Q[state]
        max_q = np.max(q)
        best_actions = np.flatnonzero(q == max_q)
        q_vals.append(max_q)
        low_actions.extend(best_actions)
        up_actions.extend(best_actions)

    return min(q_vals), max(q_vals), list(set(low_actions + up_actions))
