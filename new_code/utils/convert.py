"""
Model conversion: Keras (.h5) → TensorFlow Lite (.tflite).
"""

import os

import tensorflow as tf

from new_code.utils.config import get_model_path, get_tflite_path
from new_code.utils.logger import get_logger

log = get_logger(__name__)


def convert_to_tflite(
    model_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Convert a saved Keras model to TensorFlow Lite format.

    Args:
        model_path: Path to the .h5 model file.
        output_path: Path for the output .tflite file.

    Returns:
        Path to the saved .tflite file.
    """
    model_path = model_path or get_model_path()
    output_path = output_path or get_tflite_path()

    log.info("Loading Keras model: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantize for smaller size
    tflite_model = converter.convert()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info("TFLite model saved → %s (%.2f MB)", output_path, size_mb)
    return output_path
