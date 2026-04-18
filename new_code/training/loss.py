"""
Loss function wrapper.
"""

from tensorflow.keras.losses import CategoricalCrossentropy


def get_loss():
    """Return the loss function used for training."""
    return CategoricalCrossentropy(label_smoothing=0.0)
