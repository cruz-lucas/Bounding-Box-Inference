"""Training script."""

import argparse
import logging
import multiprocessing as mp
import traceback
from typing import Dict, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import gin
import goright.env
import gymnasium as gym
from concurrent.futures import ProcessPoolExecutor, as_completed
import structlog
from structlog import get_logger
from structlog.stdlib import LoggerFactory

import goright
from bbi.agent import BoundingBoxPlanningAgent
from bbi.models import ExpectationModel, ModelBase, PerfectModel, SamplingModel
from bbi.config.training_config import TrainingConfig, EpisodeMetrics, setup_wandb


MODEL_CLS = {
    "expectation": ExpectationModel,
    "sampling": SamplingModel,
    "perfect": PerfectModel,
    "none": ModelBase
}

logging.basicConfig(
    format="%(message)s",
    # stream=sys.stdout,
    level=logging.DEBUG,
    filename='output.log'
)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    # cache_logger_on_first_use=True,
)

logger = get_logger()

def run_episode(
    env: gym.Env,
    agent: BoundingBoxPlanningAgent,
    model: ModelBase,
    config: TrainingConfig,
    episode_seed: int,
    training: bool = True,
) -> Tuple[EpisodeMetrics, Dict]:
    """Run a single training or evaluation episode."""
    metrics = EpisodeMetrics()
    obs, info = env.reset(seed=episode_seed)
    
    prev_status = None
    epsilon = 1.0 if training else 0.0
    
    for step in range(config.n_steps):
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if training:
            if config.model_type == "perfect":
                env_unwrapped = env.unwrapped
                if not isinstance(env_unwrapped, (ModelBase, goright.env.GoRight)):
                    raise ValueError(f"Environment must inherit ModelBase. Got: {type(env_unwrapped)}")
                env_state = env_unwrapped.state.get_state()
                prev_status = int(env_state[1])
            
            td_error = agent.update(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=float(reward),
                model=model,
                prev_status=prev_status,
            )
        else:
            td_error = None
            
        metrics.update(
            reward=float(reward),
            discount=config.discount,
            step=step,
            td_error=td_error
        )
        
        obs = next_obs
        if terminated or truncated:
            break
            
    return metrics, info

@gin.configurable
def train_agent(config: TrainingConfig) -> None:
    """Train agent with given configuration."""
    run = setup_wandb(config)
    writer = SummaryWriter(log_dir=f'logs/{config.run_group}/seed_{config.seed}')

    rng = np.random.default_rng(seed=config.seed)

    try:
        logger.info("starting_training")
        
        env = gym.make(id=config.environment_id)
        agent = BoundingBoxPlanningAgent(
            state_shape=config.obs_shape,
            num_actions=2,
            step_size=config.step_size,
            discount_rate=config.discount,
            epsilon=1.0,
            horizon=config.max_horizon,
            initial_value=config.initial_value,
            seed=int(rng.integers(0, 1e10)),
            uncertainty_type=config.uncertainty_type,
        )
        
        if config.model_type not in MODEL_CLS:
            raise ValueError(f"model_type must be one of: {MODEL_CLS.keys()}. Got: {config.model_type}")
            
        model = MODEL_CLS[config.model_type](
            num_prize_indicators=len(config.obs_shape[2:]),
            env_length=config.obs_shape[0],
            status_intensities=config.status_intensities,
            seed=int(rng.integers(0, 1e10)),
        )
        model.reset()
        
        for episode in range(config.n_episodes):
            train_metrics, _ = run_episode(
                env=env,
                agent=agent,
                model=model,
                config=config,
                episode_seed=int(rng.integers(0, 1e10)),
                training=True
            )
            
            eval_metrics, _ = run_episode(
                env=env,
                agent=agent,
                model=model,
                config=config,
                episode_seed=int(rng.integers(0, 1e10)),
                training=False
            )
            
            metrics = {
                **train_metrics.to_dict(prefix="train/"),
                **eval_metrics.to_dict(prefix="evaluation/"),
                # "q_values": agent.Q,
                "episode": episode+1,
                "env_step": (episode+1) * config.n_steps
            }

            writer.add_scalars(
                'train',
                train_metrics.to_dict(),
                (episode+1) * config.n_steps,
            )
            writer.add_scalars(
                'eval',
                eval_metrics.to_dict(),
                (episode+1) * config.n_steps,
            )

            if config.debug:
                for dim1 in range(3):
                    for dim2 in range(2):
                        for dim3 in range(2):
                            slice_2d = agent.Q[:, dim1, dim2, dim3, :]
                            
                            max_value = 30
                            norm_img = (slice_2d + max_value) / (2*max_value)

                            writer.add_image(
                                f"q_values/status_{dim1}_prize1_{dim2}_prize2_{dim3}",
                                norm_img,
                                global_step=(episode+1) * config.n_steps,
                                dataformats='WH',

                            )
                            
                            # plt.close(fig)
            
            wandb.log(metrics, step=(episode+1) * config.n_steps)
            # q_values_filename = "q_values.npz"
            # np.savez_compressed(q_values_filename, q_values=agent.Q)
            # artifact = wandb.Artifact("q_values", type="model")
            # artifact.add_file(q_values_filename)
            # wandb.log_artifact(artifact)
            # os.remove(q_values_filename)
            logger.info("episode_completed", f"Episode: {episode}")

        writer.add_hparams(
            config.to_dict(),
            eval_metrics.to_dict(),
            run_name=f"{config.run_group}_seed_{config.seed}",
        )
            
        logger.info("training_completed")
        run.finish()
        
    except Exception as e:
        logger.error(
            "training_failed",
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise

def run_seeds(
    n_seeds: int = 100,
    start_seed: int = 0,
    config_file: str = "bbi/config/goright_bbi.gin",
    max_workers: Optional[int] = None
) -> None:
    """Run training across multiple seeds using process pool."""
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    logger.info("starting_multiprocess_training", max_workers=max_workers)

    gin.parse_config_file(config_file)
    base_config = TrainingConfig(
        seed=0,
    )
    
    configs = [
        TrainingConfig(**{**base_config.to_dict(), "seed": seed})
        for seed in range(start_seed, start_seed + n_seeds)
    ]
    
    # for config in configs:
    #     train_agent(config)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {
            executor.submit(train_agent, config): config.seed 
            for config in configs
        }
        
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            future.result()
            logger.info("seed_training_completed", seed=seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train learning agent with GoRight environment.")
    parser.add_argument("--config_file", type=str, default="goright_perfect", help="Path to the config gin file")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--start_seed", type=int, default=0, help="Initial seed")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    run_seeds(
        n_seeds=args.n_seeds,
        start_seed=args.start_seed,
        config_file=f"bbi/config/{args.config_file}.gin",
        max_workers=args.max_workers
    )