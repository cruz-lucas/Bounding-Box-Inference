"""Utility classes."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Prediction:
    """Dataclass to store model predictions."""
    obs: Tuple[int, ...]
    reward: float
    lower_obs: Optional[Tuple[int, ...]] = None
    upper_obs: Optional[Tuple[int, ...]] = None
    lower_reward: Optional[float] = None
    upper_reward: Optional[float] = None
    prev_status: Optional[int] = None
