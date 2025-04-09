"""Base class for models."""

from typing import List, Optional, Tuple

import numpy as np
from goright.env import GoRight

from bbi.utils import Prediction


class ModelBase(GoRight):
    """_summary_.

    Args:
        GoRight (_type_): _description_
    """
    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        show_status_ind: bool = True,
        show_prev_status_ind: bool = False
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int): Number of prize indicators.
            env_length (int): Length of the grid.
            status_intensities (List[int]): Possible status intensities.
            has_state_offset (bool): Whether to add noise to observations.
            seed (Optional[int]): Seed for reproducibility.
            render_mode (Optional[int]): Render mode.
            show_status_ind (bool, optional): Flag to include the current status in the observation. Defaults to True.
            show_prev_status_ind (bool, optional): Flag to include the previous status in the observation. Defaults to False.
        """
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=False,
            seed=seed,
            render_mode=render_mode,
            show_status_ind=show_status_ind,
            show_prev_status_ind=show_prev_status_ind
        )

        self.idx_to_status = dict(zip(np.arange(len(status_intensities), dtype=int), status_intensities))
        self.status_to_idx = dict(zip(status_intensities, np.arange(len(status_intensities), dtype=int)))

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        **kwargs,
    ) -> Prediction:
        """_summary_.

        Args:
            self (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(
        self,
        obs: Tuple[int, ...],
        action: int,
        next_obs: Tuple[int, ...],
        reward: np.float32,
    ) -> None:
        """_summary_.

        Args:
            obs (Tuple[int, ...]): _description_
            action (int): _description_
            next_obs (Tuple[int, ...]): _description_
            reward (np.float32): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
