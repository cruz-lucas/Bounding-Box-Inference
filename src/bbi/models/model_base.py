"""Abstract base class for predictive models in the BBI environment."""

from typing import List, Optional, Tuple

import numpy as np
from goright.env import GoRight

from bbi.utils import Prediction


class ModelBase(GoRight):
    """Base class for predictive models that augment Q-learning with planning."""

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        show_status_ind: bool = True,
        show_prev_status_ind: bool = False,
    ) -> None:
        """Initialize the GoRight environment with status indexing.

        Args:
            num_prize_indicators (int): Number of prize indicators in the state.
            env_length (int): Total number of positions in the grid.
            status_intensities (List[int]): Discrete intensity values.
            seed (Optional[int]): RNG seed.
            render_mode (Optional[str]): Optional Gym render mode.
            show_status_ind (bool): Whether to include status indicator in the observation.
            show_prev_status_ind (bool): Whether to include previous status indicator.
        """
        if status_intensities is None:
            status_intensities = [0, 5, 10]
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=False,
            seed=seed,
            render_mode=render_mode,
            show_status_ind=show_status_ind,
            show_prev_status_ind=show_prev_status_ind,
        )

        self.idx_to_status = dict(zip(np.arange(len(status_intensities), dtype=int), status_intensities, strict=False))
        self.status_to_idx = dict(zip(status_intensities, np.arange(len(status_intensities), dtype=int), strict=False))

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        **kwargs,
    ) -> Prediction:
        """Abstract method for predicting next state and reward.

        Must be implemented by subclasses.

        Args:
            obs (Tuple[int, ...]): Discrete observation.
            action (int): Chosen action.

        Returns:
            Prediction: Prediction object with fields for expected/uncertainty bounds.

        Raises:
            NotImplementedError: Always, unless subclass overrides.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(
        self,
        obs: Tuple[int, ...],
        action: int,
        next_obs: Tuple[int, ...],
        reward: np.float32,
    ) -> None:
        """Optional update method for stateful models.

        Args:
            obs (Tuple[int, ...]): Current observation.
            action (int): Executed action.
            next_obs (Tuple[int, ...]): Resulting observation.
            reward (np.float32): Received reward.

        Raises:
            NotImplementedError: Unless implemented by subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
