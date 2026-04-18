"""
Training loop with validation, early stopping, and checkpoint saving.
"""

import os

from tensorflow.keras.callbacks import EarlyStopping

from new_code.data.dataset import SignLanguageDataset
from new_code.models.model import build_model
from new_code.training.loss import get_loss
from new_code.training.optimizer import get_optimizer
from new_code.utils.checkpoint import save_checkpoint
from new_code.utils.config import CONFIG, get_model_path
from new_code.utils.logger import get_logger

log = get_logger(__name__)


def train(dataset_dir: str | None = None) -> dict:
    """
    Full training pipeline.

    1. Load data generators.
    2. Build model.
    3. Compile with optimizer + loss.
    4. Fit with early stopping.
    5. Save best model.

    Args:
        dataset_dir: Override path to dataset (default from config).

    Returns:
        Keras History.history dict.
    """
    # ── Data ──────────────────────────────────────────────────
    ds = SignLanguageDataset(dataset_dir)
    train_gen = ds.train_generator()
    val_gen = ds.val_generator()
    log.info(
        "Dataset loaded — %d train / %d val samples, %d classes",
        train_gen.samples,
        val_gen.samples,
        train_gen.num_classes,
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(num_classes=train_gen.num_classes)
    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss(),
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )

    # ── Training ──────────────────────────────────────────────
    log.info("Starting training — %d epochs, batch_size=%d", CONFIG["epochs"], CONFIG["batch_size"])
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=[early_stop],
        verbose=1,
    )

    # ── Save ──────────────────────────────────────────────────
    model_path = get_model_path()
    save_checkpoint(model, model_path)
    log.info("Training complete ✓")

    return history.history
