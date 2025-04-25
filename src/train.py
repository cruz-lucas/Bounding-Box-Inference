"""Training script for the BBI agent in the GoRight environment.

Supports multiple models, seed-parallel training, and integrated
logging via MLflow (offline) and JSONL for HPC compatibility.
"""

import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import gin
import gymnasium as gym
import mlflow
import numpy as np
import structlog

from bbi.agents.agent import BoundingBoxPlanningAgent
from bbi.logging.jsonl_logger import JSONLLogger
from bbi.models import ModelBase
from bbi.utils import EpisodeMetrics, TrainingConfig
from bbi.utils.mlflow_utils import setup_mlflow
from bbi.utils.model_registry import load_model

logger = structlog.get_logger()


def run_episode(
    env: gym.Env,
    agent: BoundingBoxPlanningAgent,
    model: ModelBase,
    config: TrainingConfig,
    episode_seed: int,
    training: bool = True,
) -> Tuple[EpisodeMetrics, Dict]:
    """Run a single training or evaluation episode.

    Args:
        env (gym.Env): _description_
        agent (BoundingBoxPlanningAgent): _description_
        model (_type_): _description_
        config (TrainingConfig): _description_
        episode_seed (int): _description_
        training (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[EpisodeMetrics, Dict]: _description_
    """
    metrics = EpisodeMetrics()
    obs, _ = env.reset(seed=episode_seed)

    epsilon = 1.0 if training else 0.0
    prev_status = None

    for step in range(config.n_steps):
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        td_error = (
            agent.update(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=float(reward),
                model=model,
                prev_status=prev_status,
            )
            if training
            else None
        )

        metrics.update(reward=float(reward), discount=config.discount, step=step, td_error=td_error)

        obs = next_obs
        if terminated or truncated:
            break

    return metrics, {}


@gin.configurable
def run_training(
    config: TrainingConfig, max_workers: Optional[int] = None, n_seeds: int = 1, start_seed: int = 0
) -> None:
    """Run seed-parallel training with MLflow + JSONL logging.

    Args:
        config (TrainingConfig): Base training configuration.
        max_workers (Optional[int], optional): Max parallel processes to use. Defaults to None.
        n_seeds (int, optional): Number of independent training seeds to run. Defaults to 1.
        start_seed (int, optional): Starting value for seed range. Defaults to 0.
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    logger.info("starting_multiprocess_training", max_workers=max_workers, n_seeds=n_seeds)

    configs = [
        TrainingConfig(**{**config.to_dict(), "seed": seed})
        for seed in range(start_seed, start_seed + n_seeds)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {executor.submit(train_single_seed, cfg): cfg.seed for cfg in configs}

        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                future.result()
                logger.info("seed_training_completed", seed=seed)
            except Exception as e:
                logger.error("seed_training_failed", seed=seed, error=str(e))


def train_single_seed(config: TrainingConfig) -> None:
    """Train a single seed instance of the agent.

    Args:
        config (TrainingConfig): _description_
    """
    setup_mlflow(config)
    rng = np.random.default_rng(seed=config.seed)

    log_path = Path("logs") / f"{config.run_group}.jsonl"
    jsonl_logger = JSONLLogger(log_path)

    try:
        logger.info("training_started", seed=config.seed)
        env = gym.make(
            id=config.environment_id,
            show_status_ind=config.show_status_ind,
            show_prev_status_ind=config.show_prev_status_ind,
            render_mode='human'
        )

        agent = BoundingBoxPlanningAgent(
            state_shape=config.obs_shape,
            num_actions=2,
            step_size=config.step_size,
            discount_rate=config.discount,
            epsilon=1.0,
            horizon=config.max_horizon,
            initial_value=config.initial_value,
            seed=int(rng.integers(low=0, high=int(1e10))),
            uncertainty_type=config.uncertainty_type,
        )

        model = load_model(
            config.model_type,
            num_prize_indicators=len(config.obs_shape[2:]),
            env_length=config.obs_shape[0],
            status_intensities=config.status_intensities,
            seed=int(rng.integers(low=0, high=int(1e10))),
            show_status_ind=config.show_status_ind,
            show_prev_status_ind=config.show_prev_status_ind,
        )
        model.reset()

        for episode in range(config.n_episodes):
            train_metrics, _ = run_episode(
                env,
                agent,
                model,
                config,
                episode_seed=int(rng.integers(low=0, high=int(1e10))),
                training=True,
            )
            eval_metrics, _ = run_episode(
                env,
                agent,
                model,
                config,
                episode_seed=int(rng.integers(low=0, high=int(1e10))),
                training=False,
            )

            step = (episode + 1) * config.n_steps
            metrics = {
                **train_metrics.to_dict(prefix="train/"),
                **eval_metrics.to_dict(prefix="eval/"),
                "episode": episode + 1,
                "env_step": step,
                "seed": config.seed,
            }

            mlflow.log_metrics(metrics, step=step)
            jsonl_logger.log(metrics)

            logger.info("episode_completed", episode=episode + 1, step=step)

        logger.info("training_completed", seed=config.seed)
        mlflow.end_run()

    except Exception as e:
        logger.error(
            "training_failed", seed=config.seed, error=str(e), traceback=traceback.format_exc()
        )
        raise
