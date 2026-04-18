"""
Model architecture for sign language frame-level classification.
Uses MobileNetV2 (ImageNet weights) with optional fine-tuning of top layers
and a trainable Dense classification head.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from new_code.utils.config import CONFIG
from new_code.utils.logger import get_logger

log = get_logger(__name__)


def build_model(
    num_classes: int | None = None,
    input_shape: tuple[int, int, int] | None = None,
    fine_tune_layers: int | None = None,
) -> Sequential:
    """
    Build and compile the sign-language classification model.

    Args:
        num_classes: Number of output classes (default from config).
        input_shape: Input image shape (H, W, C) (default from config).
        fine_tune_layers: Number of top base-model layers to unfreeze.
                          None = use config value. 0 = freeze all.

    Returns:
        Compiled Keras Sequential model.
    """
    num_classes = num_classes or CONFIG["num_classes"]
    input_shape = input_shape or tuple(CONFIG["input_shape"])
    if fine_tune_layers is None:
        fine_tune_layers = CONFIG.get("fine_tune_layers", 30)

    log.info(
        "Building model — classes=%d  input=%s  fine_tune_layers=%d",
        num_classes, input_shape, fine_tune_layers,
    )

    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=CONFIG["base_model_weights"],
    )

    # Freeze all layers first, then unfreeze the top N
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    log.info("Base model: %d/%d layers trainable", trainable_count, len(base.layers))

    model = Sequential([
        base,
        Flatten(),
        BatchNormalization(),
        Dense(
            CONFIG["dense_units"],
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        Dropout(CONFIG["dropout_rate"]),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",  # optimizer is set properly in training
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log.info("Model compiled — total params: %s", f"{model.count_params():,}")
    return model
