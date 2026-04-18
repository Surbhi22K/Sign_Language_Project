"""
Centralized configuration for the Sign Language Decoding System.
All paths, hyperparameters, and constants are defined here.
"""

import os

# Project root (two levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────
    "dataset_dir": os.path.join(PROJECT_ROOT, "dataset"),
    "video_dir": os.path.join(PROJECT_ROOT, "video"),
    "saved_models_dir": os.path.join(PROJECT_ROOT, "saved_models"),
    "reports_dir": os.path.join(PROJECT_ROOT, "reports"),
    "model_filename": "sign_model.h5",
    "tflite_filename": "sign_model.tflite",
    "results_filename": "results.json",
    "test_video": "rama.webm",

    # ── Image / Input ─────────────────────────────────────────
    "image_size": (64, 64),
    "input_shape": (64, 64, 3),
    "color_mode": "rgb",

    # ── Model ─────────────────────────────────────────────────
    "num_classes": 24,
    "base_model_weights": "imagenet",
    "freeze_base": True,
    "fine_tune_layers": 30,
    "dense_units": 128,
    "dropout_rate": 0.5,

    # ── Training ──────────────────────────────────────────────
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 5e-4,
    "validation_split": 0.2,
    "early_stopping_patience": 100,
    "shuffle_train": True,

    # ── Sequence Decoder ──────────────────────────────────────
    "sliding_window_size": 5,
    "frame_interval_sec": 1.0,

    # ── Mobile / TFLite ───────────────────────────────────────
    "frame_capture_interval_ms": 250,
    "max_model_size_mb": 20,
}


def get_model_path() -> str:
    """Return absolute path to the saved model checkpoint."""
    return os.path.join(CONFIG["saved_models_dir"], CONFIG["model_filename"])


def get_tflite_path() -> str:
    """Return absolute path to the converted TFLite model."""
    return os.path.join(CONFIG["saved_models_dir"], CONFIG["tflite_filename"])


def get_test_video_path() -> str:
    """Return absolute path to the test video."""
    return os.path.join(CONFIG["video_dir"], CONFIG["test_video"])


def get_results_path() -> str:
    """Return absolute path to the results JSON file."""
    return os.path.join(CONFIG["reports_dir"], CONFIG["results_filename"])
