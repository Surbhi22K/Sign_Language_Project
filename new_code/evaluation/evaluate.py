"""
Full evaluation pipeline — run metrics on a trained model.
"""

import numpy as np

from new_code.data.dataset import SignLanguageDataset
from new_code.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_loss_curve,
    save_results,
)
from new_code.utils.checkpoint import load_checkpoint
from new_code.utils.config import get_model_path
from new_code.utils.logger import get_logger

log = get_logger(__name__)


def evaluate(
    model_path: str | None = None,
    dataset_dir: str | None = None,
    history: dict | None = None,
) -> dict:
    """
    Evaluate a trained model on the validation set.

    Args:
        model_path: Path to saved model (default from config).
        dataset_dir: Path to dataset directory (default from config).
        history: Training history dict for plotting loss curves.

    Returns:
        Combined results dict.
    """
    model_path = model_path or get_model_path()
    log.info("Loading model for evaluation: %s", model_path)
    model = load_checkpoint(model_path)

    ds = SignLanguageDataset(dataset_dir)
    val_gen = ds.val_generator()
    labels = [k for k, _ in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]

    # Predict
    log.info("Running predictions on validation set (%d samples)…", val_gen.samples)
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    # Frame-level metrics
    results = compute_metrics(y_true, y_pred, labels)

    # Plots
    plot_confusion_matrix(y_true, y_pred, labels)

    if history:
        plot_loss_curve(history)

    # Save
    save_results(results)
    log.info("Evaluation complete ✓")
    return results
