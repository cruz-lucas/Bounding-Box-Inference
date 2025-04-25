"""Command-line entry point for training and evaluation.

Parses arguments and dispatches to training logic using Gin-configured settings.
"""

import argparse

import gin

from bbi.utils import TrainingConfig
from train import run_training


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="goright_bbi",
        help="Base name of the Gin config file (without extension)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Maximum number of parallel workers"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run training from CLI."""
    args = parse_args()
    gin.parse_config_file(f"configs/{args.config_file}.gin")
    base_config = TrainingConfig(seed=0)
    run_training(base_config, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
