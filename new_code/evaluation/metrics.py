"""
Evaluation metrics — frame-level and sequence-level.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from new_code.utils.config import CONFIG, get_results_path
from new_code.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Frame-level metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> dict:
    """
    Compute frame-level classification metrics.

    Args:
        y_true: Ground-truth class indices (N,).
        y_pred: Predicted class indices (N,).
        labels: Optional list of class label strings.

    Returns:
        Dict with accuracy, precision, recall, f1, and classification report.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Build explicit label indices so target_names always matches
    if labels is not None:
        label_indices = list(range(len(labels)))
        report = classification_report(
            y_true, y_pred,
            labels=label_indices,
            target_names=labels,
            zero_division=0,
        )
    else:
        report = classification_report(y_true, y_pred, zero_division=0)

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "classification_report": report,
    }
    log.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f", acc, prec, rec, f1)
    return results


# ─────────────────────────────────────────────────────────────
# Sequence-level metrics
# ─────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def sequence_accuracy(y_true_seqs: list[str], y_pred_seqs: list[str]) -> float:
    """Fraction of sequences that match exactly."""
    if not y_true_seqs:
        return 0.0
    return sum(t == p for t, p in zip(y_true_seqs, y_pred_seqs)) / len(y_true_seqs)


def word_accuracy_rate(y_true_seqs: list[str], y_pred_seqs: list[str]) -> float:
    """Average word-level accuracy across all sequences."""
    if not y_true_seqs:
        return 0.0
    scores = []
    for t, p in zip(y_true_seqs, y_pred_seqs):
        t_words = t.split()
        p_words = p.split()
        if not t_words:
            scores.append(1.0 if not p_words else 0.0)
            continue
        correct = sum(tw == pw for tw, pw in zip(t_words, p_words))
        scores.append(correct / max(len(t_words), len(p_words)))
    return float(np.mean(scores))


def compute_sequence_metrics(y_true_seqs: list[str], y_pred_seqs: list[str]) -> dict:
    """
    Compute sequence-level metrics.

    Returns dict with sequence_accuracy, word_accuracy_rate, and avg edit distance.
    """
    seq_acc = sequence_accuracy(y_true_seqs, y_pred_seqs)
    word_acc = word_accuracy_rate(y_true_seqs, y_pred_seqs)
    edit_dists = [levenshtein_distance(t, p) for t, p in zip(y_true_seqs, y_pred_seqs)]
    avg_edit = float(np.mean(edit_dists)) if edit_dists else 0.0

    results = {
        "sequence_accuracy": seq_acc,
        "word_accuracy_rate": word_acc,
        "avg_edit_distance": avg_edit,
    }
    log.info("SeqAcc=%.4f  WordAcc=%.4f  AvgEditDist=%.2f", seq_acc, word_acc, avg_edit)
    return results


# ─────────────────────────────────────────────────────────────
# Persistence & Plotting
# ─────────────────────────────────────────────────────────────

def save_results(results: dict, path: str | None = None) -> None:
    """Write results dictionary to a JSON file."""
    path = path or get_results_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved → %s", path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Generate and save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = save_path or os.path.join(CONFIG["reports_dir"], "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved → %s", save_path)


def plot_loss_curve(history: dict, save_path: str | None = None) -> None:
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history["accuracy"], label="Train Acc")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()

    save_path = save_path or os.path.join(CONFIG["reports_dir"], "training_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Training curves saved → %s", save_path)
