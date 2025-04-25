"""Module with perfect model."""

from typing import List, Optional, Tuple

import numpy as np

from bbi.models.model_base import ModelBase
from bbi.utils import Prediction


class PerfectModel(ModelBase):
    """Perfect model class."""

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        show_status_ind: bool = True,
        show_prev_status_ind: bool = False,
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
            show_prev_status_ind=show_prev_status_ind,
        )

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        prev_status: int | None = None,
        **kwargs,
    ) -> Prediction:
        """Predict the next state and reward given an observation and action, and also compute lower and upper bounds by using the minimum and maximum status intensities, respectively.

        This method sets the environment’s state based on the observation,
        calls step() to simulate the transition, then repeats the process with the
        status indicator forced to its minimum and maximum values. The environment’s
        internal state is restored at the end.

        Args:
            obs: A dictionary with keys 'position', 'status_indicator', and 'prize_indicators'
                representing the current observation.
            action: The action to be executed (e.g. 0 for LEFT, 1 for RIGHT).

        Returns:
            A tuple containing:
            - expected_obs: Expected next observation.
            - expected_reward: Expected reward.
            - previous_status: The current status indicator, used as previous status next time step.
        """
        pos = obs[0]

        if self.show_status_ind and self.show_prev_status_ind:
            status = self.idx_to_status[obs[2]] if obs[2] in self.idx_to_status.keys() else obs[2]
            prev_status = (
                self.idx_to_status[obs[1]] if obs[1] in self.idx_to_status.keys() else obs[1]
            )

        elif self.show_status_ind:
            status = self.idx_to_status[obs[1]] if obs[1] in self.idx_to_status.keys() else obs[1]
            prev_status = None

        elif self.show_prev_status_ind:
            prev_status = (
                self.idx_to_status[obs[1]] if obs[1] in self.idx_to_status.keys() else obs[1]
            )
            status = self._np_random.choice(self.status_intensities)

        prize = np.array(obs[-2:])

        if not isinstance(prev_status, int):
            raise ValueError(
                f"Previous status must be int for the Perfect model prediction. Got: {prev_status}"
            )

        self.state.set_state(
            position=pos,
            previous_status_indicator=prev_status,
            current_status_indicator=status,
            prize_indicators=np.array(prize),
        )
        exp_obs, exp_reward, _, _, _ = self.step(action)
        previous_status = self.state.previous_status_indicator

        if "prev_status_indicator" in exp_obs.keys():
            exp_obs["prev_status_indicator"] = self.status_to_idx[exp_obs["prev_status_indicator"]]

        if "status_indicator" in exp_obs.keys():
            exp_obs["status_indicator"] = self.status_to_idx[exp_obs["status_indicator"]]

        exp_obs = [int(val) for val in np.hstack(list(exp_obs.values()))]

        return Prediction(obs=tuple(exp_obs), reward=exp_reward, prev_status=previous_status)

    def update(
        self,
        obs: Tuple[int, ...],
        action: int,
        next_obs: Tuple[int, ...],
        reward: np.float32,
    ) -> None:
        """_summary_.

        Returns:
            _type_: _description_
        """
        return None
