"""
Image transforms for the Sign Language Decoding System.
"""

import cv2
import numpy as np


def resize_and_normalize(
    frame: np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Resize a BGR/RGB frame and scale pixel values to [0, 1].

    Args:
        frame: Input image (H, W, C) in uint8.
        size: Target (width, height).

    Returns:
        Normalized float32 array of shape (*size, C).
    """
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def augment(frame: np.ndarray) -> np.ndarray:
    """
    Apply simple data-augmentation (random horizontal flip + brightness jitter).
    Operates on a float32 frame in [0, 1].
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        frame = np.fliplr(frame)

    # Random brightness shift ±10 %
    factor = 1.0 + np.random.uniform(-0.1, 0.1)
    frame = np.clip(frame * factor, 0.0, 1.0)

    return frame
