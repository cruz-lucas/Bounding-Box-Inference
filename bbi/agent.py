"""Implementation of the learning agent."""

import logging
from collections.abc import Iterable
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from bbi.models import PerfectModel
from bbi.models.model_base import ModelBase

logger = logging.getLogger(__name__)


def softmin(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Compute a softmin distribution over an array x using temperature tau.

    Args:
        x: Array of values.
        tau: Temperature parameter (default: 1.0).

    Returns:
        A probability distribution over x computed via softmin.
    """
    scaled = np.exp(-np.array(x) / tau)
    return scaled / np.sum(scaled)


class BoundingBoxPlanningAgent:
    """A planning agent that implements a bounding-box inference method for selective planning.

    This agent uses tabular Q-learning augmented with model-based value expansion.
    In each update, it first computes a 1–step TD target from the real transition.
    Then, for additional planning steps (up to a given horizon), it queries a predictive
    model that returns not only an expected next state and reward but also lower and upper
    bounds for both. For each rollout step, the agent computes a target using the expected
    prediction and also derives an uncertainty measure (the difference between the upper and
    lower targets). These targets are then weighted via a softmin function over uncertainties
    and combined to update Q-values.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        step_size: float,
        discount_rate: float,
        epsilon: float,
        horizon: int,
        initial_value: float = 0.0,
        seed: Optional[int] = None,
        uncertainty_type: str = "unselective",
    ) -> None:
        """Class for the learning agent.

        Args:
            state_shape (Tuple[int, ...]): Shape of the state space (for storing Q in a numpy array).
            num_actions (int): Number of possible actions.
            step_size (float): Learning rate for Q–updates.
            discount_rate (float): Discount factor (gamma).
            epsilon (float): Epsilon for epsilon–greedy exploration.
            horizon (int): Maximum planning horizon H.
            initial_value (float, optional): Value to initialize the q-values. Defaults to 0.0.
            seed (Optional[int], optional): Optional random seed. Defaults to None.
            uncertainty_type (str, optional): Uncertainty estimation method. Defaults to "unselective".
        """
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.horizon = horizon
        self.uncertainty_type = uncertainty_type
        self.rng = np.random.default_rng(seed)

        self.Q = np.full(state_shape + (num_actions,), fill_value=initial_value)

        self.td_errors = []

    def act(self, obs: Dict[str, np.ndarray] | Tuple[int, ...], eps: float | None = None) -> int:
        """Select an action using an epsilon–greedy strategy based on Q–values.

        Args:
            state: Current state as a discrete tuple.

        Returns:
            The chosen action as an integer.
        """
        if eps is None:
            eps = self.epsilon
        if self.rng.random() < eps:
            chosen = int(self.rng.integers(self.num_actions))
            logger.debug(f"Random action chosen: {chosen}")
            return chosen
        else:
            state = ()
            if isinstance(obs, dict):
                state = self._round_obs(obs)
            if isinstance(obs, tuple):
                state = obs
            q_values = self.Q[state]
            best_actions = np.flatnonzero(q_values == q_values.max())
            chosen = int(self.rng.choice(best_actions))
            logger.debug(f"Greedy action chosen from Q[{state}] = {q_values}: {chosen}")
            return chosen

    def update(
        self,
        obs: Dict[str, Union[np.float32, np.ndarray]],
        action: int,
        next_obs: Dict[str, Union[np.float32, np.ndarray]],
        reward: float,
        model: ModelBase,
        prev_status: int | None = None,
    ) -> None:
        """Perform a bounding-box inference update of Q(s, a).

        The update procedure is as follows:
          1. Compute the 1–step TD target from the observed (real) transition.
          2. For each planning step h = 1 ... H–1:
             a. Use the predictive model to obtain an imaginary transition from the current planning
                state. The model returns:
                - an expected next state and reward,
                - a lower bound state and reward,
                - an upper bound state and reward.
             b. Compute:
                - the expected target: r_pred + γ * maxₐ Q(s_pred, a),
                - lower target:    r_lower + γ * (minₛ∈box maxₐ Q(s, a))
                - upper target:    r_upper + γ * (maxₛ∈box maxₐ Q(s, a))
             c. Define the uncertainty as (upper target – lower target) and record the expected target.
          3. Combine the targets from all steps using a softmin weighting over uncertainties.
          4. Update Q(s, a) with a gradient step toward the weighted target.

        Args:
            obs: Current observation (a dict with keys "position", "status_indicator", and "prize_indicators").
            action: Action taken in the current state.
            next_obs: Observed next observation.
            reward: Observed reward after taking the action.
            model: A predictive model providing a method
                   predict(state: Tuple[int, ...], action: int) ->
                     (expected_state, expected_reward,
                      lower_state, lower_reward,
                      upper_state, upper_reward).

        Returns:
            None.
        """
        gamma = self.discount_rate

        s = self._round_obs(obs)
        s_prime = self._round_obs(next_obs)

        target_real = reward + gamma * np.max(self.Q[s_prime])
        targets = [target_real]
        uncertainties = [0.0]
        disc_reward = reward
        low_disc_reward = reward
        high_disc_reward = reward

        logger.debug(f"Real transition: s={s}, action={action}, s'={s_prime}, reward={reward}, q(s)={self.Q[s]}, q(s')={self.Q[s_prime]}")
        logger.debug(f"1-step TD target: {target_real}")

        pred_state = s_prime
        planning_action = self.act(s_prime, eps=0.0)

        state_bb = (pred_state, pred_state)
        action_bb = [planning_action]

        for h in range(1, self.horizon):
            if self.uncertainty_type == "bbi":
                prediction = model.predict(
                    obs=pred_state,
                    action=planning_action,
                    state_bb=state_bb,
                    action_bb=action_bb,
                    prev_status=None,
                )
                if prediction.lower_obs is None or prediction.upper_obs is None:
                    raise ValueError(
                        f"Prediction must have lower and upper observations with BBI planning. Got: {prediction}"
                    )

                pred_state = prediction.obs
                pred_reward = prediction.reward
                low_state = prediction.lower_obs
                low_reward = prediction.lower_reward
                up_state = prediction.upper_obs
                up_reward = prediction.upper_reward

                state_bb = (low_state, up_state)

                logger.debug(
                    f"Planning step {h}: pred_state={pred_state}, pred_reward={pred_reward}, q(pred_state)={self.Q[pred_state]}, "
                    f"low_state={low_state}, low_reward={low_reward}, "
                    f"up_state={up_state}, up_reward={up_reward}"
                )

                disc_reward += gamma * pred_reward
                low_disc_reward += gamma * low_reward
                high_disc_reward += gamma * up_reward
                gamma *= self.discount_rate
                q_pred = np.max(self.Q[pred_state])
                target_pred = disc_reward + gamma * q_pred

                q_low, q_up, action_bb = self.q_bounding_box(low_state, up_state)
                target_low = low_disc_reward + gamma * q_low
                target_up = high_disc_reward + gamma * q_up
                uncertainty = target_up - target_low

                logger.debug(
                    f"Planning step {h}: pred_target={target_pred}, target_low={target_low}, "
                    f"target_up={target_up}, uncertainty={uncertainty}"
                )

            elif self.uncertainty_type == "unselective":
                if isinstance(model, PerfectModel):
                    if prev_status is None:
                        raise ValueError(f"Planning with perfect model must receive the previous status. Got: {prev_status}")
                else:
                    if prev_status is not None:
                        raise ValueError(f"Planning with any other model rather than the perfect model must NOT receive the previous status. Got: {prev_status}. Expected: None")

                prediction = model.predict(
                    obs=pred_state,
                    action=planning_action,
                    state_bb=None,
                    action_bb=None,
                    prev_status=prev_status,  # None if not using the perfect model
                )

                pred_state = prediction.reward
                pred_reward = prediction.reward
                prev_status = prediction.prev_status  # None if not using the perfect model

                disc_reward += gamma * pred_reward
                gamma *= self.discount_rate
                q_pred = np.max(self.Q[pred_state])
                target_pred = disc_reward + gamma * q_pred

                uncertainty = 0

                logger.debug(
                    f"Planning step {h}: pred_state={pred_state}, action={planning_action}, pred_reward={pred_reward},"
                    f"pred_target={target_pred}, q(pred_state)={self.Q[pred_state]}"
                )

            else:
                raise ValueError(
                    f"uncertainty_type must be one of: unselective, bbi. Got: {self.uncertainty_type}."
                )

            targets.append(target_pred)
            uncertainties.append(uncertainty)

            planning_action = self.act(pred_state, eps=0.0)

        weights = softmin(np.array(uncertainties))
        final_target = np.dot(weights, np.array(targets))

        td_error = final_target - self.Q[s][action]

        self.Q[s][action] += self.step_size * td_error

        logger.debug(
            f"Update for state {s}, action {action}: "
            f"targets={targets}, uncertainties={uncertainties}, weights={weights}, "
            f"final_target={final_target}, TD error={td_error}, updated Q={self.Q[s][action]}"
        )

        self.td_errors.append(td_error)

        return td_error

    def q_bounding_box(
        self, low_state: Tuple[int, ...], up_state: Tuple[int, ...]
    ) -> Tuple[float, float, List[int]]:
        """Compute the minimum and maximum greedy Q-values over all states in the bounding box.

        A state s lies in the bounding box defined by low_state and up_state if and only if
          low_state[i] <= s[i] <= up_state[i] for all dimensions i.

        Args:
            low_state: Lower bound state as a tuple.
            up_state: Upper bound state as a tuple.

        Returns:
            A tuple (q_low, q_up) where q_low is the minimum and q_up is the maximum
            greedy Q-value among all states in the bounding box.
        """
        dims = len(low_state)
        low_candidates = []
        up_candidates = []
        ranges = [range(low_state[i], up_state[i] + 1) for i in range(dims)]
        for s in product(*ranges):
            q_vals = self.Q[s]
            max_q = np.max(q_vals)
            best_actions = np.flatnonzero(q_vals == max_q)
            low_candidates.append((max_q, best_actions))
            up_candidates.append((max_q, best_actions))

        q_low: float = min(low_candidates, key=lambda x: x[0])[0]
        low_actions = [i for (q, a) in low_candidates if q == q_low for i in a]

        q_up: float = max(up_candidates, key=lambda x: x[0])[0]
        up_actions = [i for (q, a) in up_candidates if q == q_up for i in a]

        action_bb = list(set(low_actions + up_actions))
        return (q_low, q_up, action_bb)

    def _round_obs(self, obs: Dict[str, np.ndarray]) -> Tuple[int, ...]:
        """Convert a continuous observation into a discrete (tabular) state.

        The conversion is performed as follows:
          - The "position" is rounded to the nearest integer.
          - The "status_indicator" is discretized into three bins:
                <= 2.5 -> 0,
                <= 7.5 -> 1,
                > 7.5  -> 2.
          - Each element in "prize_indicators" is thresholded at 0.5.

        Args:
            obs: A dictionary with keys "position", "status_indicator", and "prize_indicators".

        Returns:
            A tuple representing the discrete state.
        """
        pos = int(np.round(obs["position"]).item())

        prize_indicators = obs["prize_indicators"]
        if not isinstance(prize_indicators, Iterable):
            raise ValueError(
                f"Prize indicators in the observation object must be iterable. Got: {type(prize_indicators)}"
            )

        prize_indicators = [
            int(prize_indicators[i] >= 0.5) for i in range(len(prize_indicators))
        ]

        if "status_indicator" in obs.keys():
            status_val = float(obs["status_indicator"].item())
            if status_val <= 2.5:
                intensity_value = 0
            elif status_val <= 7.5:
                intensity_value = 1
            else:
                intensity_value = 2

            return (pos, intensity_value, *prize_indicators)
        return (pos, *prize_indicators)
