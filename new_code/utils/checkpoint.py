"""
Checkpoint utilities — save and load Keras models.
"""

import os

from new_code.utils.logger import get_logger

log = get_logger(__name__)


def save_checkpoint(model, path: str) -> None:
    """Save a Keras model to *path*, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(path: str):
    """Load and return a Keras model from *path*."""
    from tensorflow.keras.models import load_model as _load

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    model = _load(path)
    log.info("Checkpoint loaded ← %s", path)
    return model
