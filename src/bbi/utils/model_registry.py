"""Registry of available predictive models for dynamic loading.

This module enables string-based model selection from configs or CLI.
"""

from bbi.models.expectation import ExpectationModel
from bbi.models.model_base import ModelBase
from bbi.models.perfect import PerfectModel
from bbi.models.sampling import SamplingModel

MODEL_REGISTRY = {
    "expectation": ExpectationModel,
    "sampling": SamplingModel,
    "perfect": PerfectModel,
    "none": ModelBase,
}


def load_model(name: str, **kwargs) -> ModelBase:
    """Load model class from the registry.

    Args:
        name (str): Name of the model to load (must be in MODEL_REGISTRY).
        **kwargs: Keyword arguments for the model constructor.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](**kwargs)
