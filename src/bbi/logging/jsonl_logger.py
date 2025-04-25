"""Module for file-count–aware JSONL logging.

This logger appends training metrics in JSON Lines format, minimizing
file count on disk—a requirement for HPC environments like Compute Canada.
"""

import json
from pathlib import Path


class JSONLLogger:
    """Logger that writes metrics to a single JSONL file."""

    def __init__(self, path: Path) -> None:
        """Initialize the logger and create the directory if necessary.

        Args:
            path: Path to the JSONL file for logging.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a")

    def log(self, metrics: dict) -> None:
        """Append a dictionary of metrics as a JSON line.

        Args:
            metrics: Dictionary containing scalar metrics to log.
        """
        self.file.write(json.dumps(metrics) + "\n")
        self.file.flush()
