"""Module with perfect model."""

from typing import List, Optional, Tuple

import numpy as np

from bbi.models.model_base import ModelBase


class PerfectModel(ModelBase):
    """Perfect model class.

    Args:
        ModelBase (_type_): _description_
    """

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int): Number of prize indicators.
            env_length (int): Length of the grid.
            status_intensities (List[int]): Possible status intensities.
            has_state_offset (bool): Whether to add noise to observations.
            seed (Optional[int]): Seed for reproducibility.
            render_mode (Optional[int]): Render mode.
        """
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=False,
            seed=seed,
            render_mode=render_mode,
        )

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        prev_status: int | None = None,
        **kwargs,
    ) -> (
        Tuple[Tuple[int, ...], float]
        | Tuple[Tuple[int, ...], float, Tuple[int, ...], float, Tuple[int, ...], float]
    ):
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
            - lower_obs: Next observation when the status is forced to the minimum.
            - lower_reward: Reward for the lower bound.
            - upper_obs: Next observation when the status is forced to the maximum.
            - upper_reward: Reward for the upper bound.
        """
        pos = obs[0]
        status = obs[1]
        prize = np.array(obs[2:])

        if status == 1:
            status = 5
        elif status == 2:
            status = 10

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

        _, exp_reward, _, _, _ = self.step(action)
        state = self.state.get_state()
        exp_obs = state[self.state.mask]
        previous_status = state[~self.state.mask]

        assert len(previous_status) == 1
        previous_status = int(previous_status[0])

        exp_obs[1] = np.argwhere(self.status_intensities == exp_obs[1])
        exp_obs = [int(i) for i in exp_obs]

        return (tuple(exp_obs), exp_reward, previous_status)

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
