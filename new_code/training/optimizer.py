"""
Optimizer factory.
"""

from tensorflow.keras.optimizers import Adam

from new_code.utils.config import CONFIG


def get_optimizer():
    """Return the optimizer used for training."""
    return Adam(learning_rate=CONFIG["learning_rate"])
