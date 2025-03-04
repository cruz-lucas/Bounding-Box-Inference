import gin
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import wandb
import numpy as np
import wandb.util

@gin.configurable
@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    seed: int
    n_episodes: int = 600
    n_steps: int = 500
    step_size: float = 0.1
    initial_value: float = 0.0
    discount: float = 0.9
    max_horizon: int = 5
    tau: float = 1.0
    environment_id: str = "GoRight-v0"
    obs_shape: Tuple[int, int, int, int] = (11, 3, 2, 2)
    status_intensities: List[int] = field(default_factory=lambda: [0, 5, 10])
    model_type: str = "expectation"
    uncertainty_type: str = "unselective"
    experiment_name: str = "BBI_Training"
    run_group: str = "default"
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

@dataclass
class EpisodeMetrics:
    """Metrics collected during a training/evaluation episode."""
    total_reward: float = 0.0
    discounted_return: float = 0.0
    episode_length: int = 0
    td_errors: List[float] = field(default_factory=list)
    
    def update(self, reward: float, discount: float, step: int, td_error: Optional[float] = None):
        """Update metrics with new step information."""
        self.total_reward += reward
        self.discounted_return += discount ** step * reward
        self.episode_length += 1
        if td_error is not None:
            self.td_errors.append(td_error)
    
    def to_dict(self, prefix: str = "") -> Dict:
        """Convert metrics to dictionary with optional prefix."""
        metrics = {
            "total_reward": self.total_reward,
            "discounted_return": self.discounted_return,
            "episode_length": self.episode_length,
        }
        if self.td_errors:
            metrics["avg_td_error"] = np.mean(self.td_errors)
            metrics["max_td_error"] = np.max(self.td_errors)
            
        return {f"{prefix}{k}": v for k, v in metrics.items()}


def setup_wandb(config: TrainingConfig) -> str:
    """Setup wandn experiment and start run."""
    run = wandb.init(
        project=config.experiment_name,
        name=f"{config.experiment_name}_seed_{config.seed}",
        config=config.to_dict(),
        reinit=True,
        group=config.run_group,
        dir=f"./wandb_{config.run_group}_seed_{config.seed}",
        notes=config.notes,
        id=f"{config.run_group}_seed_{config.seed}_{wandb.util.generate_id()}",
    )
    return run

class TrainingError(Exception):
    """Custom exception for training errors that can be pickled."""
    def __init__(self, seed: int, error_msg: str = "", traceback_str: str = ""):
        self.seed = seed
        self.error_msg = error_msg
        self.traceback_str = traceback_str
        super().__init__(self.error_msg)

