"""Utility for setting up MLflow experiment tracking.

This module configures MLflow in offline mode for tracking training runs,
making it suitable for offline environments like Compute Canada.
"""

import mlflow

from bbi.utils import TrainingConfig


def setup_mlflow(config: TrainingConfig) -> None:
    """Initialize MLflow tracking for a training run.

    Sets tracking URI, experiment, and starts a new run. Logs all hyperparameters.

    Args:
        config: Training configuration object.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config.experiment_name)
    mlflow.start_run(run_name=f"{config.run_group}_seed_{config.seed}")
    mlflow.log_params(config.to_dict())
