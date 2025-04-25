"""Model that returns deterministic expected predictions."""

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bbi.models.model_base import ModelBase
from bbi.utils import Prediction


class ExpectationModel(ModelBase):
    """Model that returns deterministic expected predictions for transitions."""

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
        """Initialize expectation model with deterministic status logic.

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
            seed=seed,
            render_mode=render_mode,
            show_status_ind=show_status_ind,
            show_prev_status_ind=show_prev_status_ind,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment state and override status computation.

        Args:
            seed (Optional[int]): RNG seed.
            options (Optional[Dict[str, Any]], optional): _description_. Defaults to None.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: _description_
        """
        obs, info = super().reset(seed=seed)
        self.state.current_status_indicator = self._compute_next_status()
        return obs, info

    def _compute_next_status(self, previous_status: int = 0, current_status: int = 0) -> int:
        """Return fixed expected status.

        Args:
            previous_status (int, optional): _description_. Defaults to 0.
            current_status (int, optional): _description_. Defaults to 0.

        Returns:
            int: Deterministic status value (e.g. 5).
        """
        return 5

    def _compute_expected_next_prize_indicators(self, next_position: float) -> np.ndarray:
        """Compute next prize indicator vector from position.

        Args:
            next_position (float): Agent's next location on grid.

        Returns:
            np.ndarray: Expected prize indicator values.
        """
        prize_indicators = np.array(self.state.prize_indicators)
        if int(next_position) == self.env_length - 1:
            if int(self.state.position) == self.env_length - 2:
                return np.ones_like(prize_indicators, dtype=float) / 3.0
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        state_bb: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        action_bb: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Prediction:
        """Generate predicted state and reward with optional bounds.

        Args:
            obs (Tuple[int, ...]): Current discrete state.
            action (int): Action to simulate.
            state_bb (Optional[Tuple]): Bounding-box (low, high) for uncertainty planning.
            action_bb (Optional[Tuple]): Action bounds for bounding-box Q-estimation.

        Returns:
            Prediction: Expected, lower, and upper prediction results.
        """
        pos = obs[0]

        # Decode status
        if self.show_status_ind and self.show_prev_status_ind:
            status = self.idx_to_status.get(obs[2], obs[2])
            prev_status = self.idx_to_status.get(obs[1], obs[1])
        elif self.show_status_ind:
            status = self.idx_to_status.get(obs[1], obs[1])
            prev_status = 0  # doesn't matter
        elif self.show_prev_status_ind:
            prev_status = self.idx_to_status.get(obs[1], obs[1])
            status = self._np_random.choice(self.status_intensities)
        else:
            prev_status = 0  # doesn't matter
            status = 0

        prize = np.array(obs[-2:])
        self.state.set_state(
            position=pos,
            current_status_indicator=status,
            prize_indicators=prize,
            previous_status_indicator=prev_status
        )

        self._compute_next_status = lambda *args, **kwargs: 5
        exp_obs, exp_reward, _, _, _ = self.step(action)

        if state_bb is not None and action_bb is not None:
            state_ranges = [range(low, high + 1) for low, high in zip(*state_bb, strict=False)]
            state_candidates = []
            reward_candidates = []

            for s in product(*state_ranges):
                for a in action_bb:
                    pos = s[0]
                    status = s[1] if self.show_status_ind else self._np_random.choice(self.status_intensities)
                    prize = np.array(s[-2:])
                    self.state.set_state(
                        position=pos,
                        current_status_indicator=status,
                        prize_indicators=prize,
                        previous_status_indicator=prev_status
                    )

                    self._compute_next_status = lambda *args, **kwargs: 0
                    lower_obs, lower_reward, *_ = self.step(a)

                    self.state.set_state(
                        position=pos,
                        current_status_indicator=status,
                        prize_indicators=prize,
                        previous_status_indicator=prev_status
                    )

                    self._compute_next_status = lambda *args, **kwargs: 10
                    upper_obs, upper_reward, *_ = self.step(a)

                    state_candidates.append(np.hstack(list(lower_obs.values())))
                    state_candidates.append(np.hstack(list(upper_obs.values())))
                    reward_candidates.extend([lower_reward, upper_reward])

            lower_obs = list(map(int, np.min(state_candidates, axis=0)))
            upper_obs = list(map(int, np.max(state_candidates, axis=0)))
            lower_reward = min(reward_candidates)
            upper_reward = max(reward_candidates)

            exp_obs = list(map(int, np.hstack(list(exp_obs.values()))))

            if self.show_status_ind and not self.show_prev_status_ind:
                lower_obs[1] = self.status_to_idx[lower_obs[1]]
                upper_obs[1] = self.status_to_idx[upper_obs[1]]
                exp_obs[1] = self.status_to_idx[exp_obs[1]]

            return Prediction(
                obs=tuple(exp_obs),
                reward=exp_reward,
                lower_obs=tuple(lower_obs),
                upper_obs=tuple(upper_obs),
                lower_reward=lower_reward,
                upper_reward=upper_reward)

        if "prev_status_indicator" in exp_obs:
            exp_obs["prev_status_indicator"] = self.status_to_idx[exp_obs["prev_status_indicator"]]
        if "status_indicator" in exp_obs:
            exp_obs["status_indicator"] = self.status_to_idx[exp_obs["status_indicator"]]

        return Prediction(obs=tuple(map(int, np.hstack(list(exp_obs.values())))), reward=exp_reward)

    def update(
        self,
        obs: Tuple[int, ...],
        action: int,
        next_obs: Tuple[int, ...],
        reward: np.float32,
    ) -> None:
        """No-op update for stateless models."""
        return None
