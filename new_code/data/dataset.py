"""
Dataset and video-frame loading utilities.
"""

import os
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from new_code.data.transforms import resize_and_normalize
from new_code.utils.config import CONFIG
from new_code.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Video Frame Loader
# ─────────────────────────────────────────────────────────────
class VideoFrameLoader:
    """
    Read a video file and yield pre-processed frames.

    Usage:
        loader = VideoFrameLoader("video/signv.mp4")
        for frame in loader:
            # frame shape: (64, 64, 3), dtype float32 in [0,1]
            ...
    """

    def __init__(self, video_path: str, size: tuple[int, int] | None = None):
        self.video_path = video_path
        self.size = size or tuple(CONFIG["image_size"])

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        log.info("Reading video: %s", self.video_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            yield resize_and_normalize(frame, self.size)

        cap.release()
        log.info("Extracted %d frames from %s", frame_count, self.video_path)

    def to_array(self) -> np.ndarray:
        """Return all frames as a single numpy array of shape (T, H, W, C)."""
        frames = list(self)
        return np.stack(frames, axis=0)


# ─────────────────────────────────────────────────────────────
# Image-Directory Dataset (Keras generator)
# ─────────────────────────────────────────────────────────────
def augment_blur_grayscale(image):
    """
    Custom preprocessing function.
    Randomly applies Gaussian blur and/or Grayscale conversion to prevent
    the model from relying on sharp edges or specific colors.
    """
    img = image.copy()
    
    # 30% chance of grayscale
    if np.random.rand() < 0.3:
        # Convert to grayscale and back to 3 channels to preserve shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
    # 30% chance of blur
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
    return img

class SignLanguageDataset:
    """
    Wraps Keras ImageDataGenerator for directory-based datasets.

    Expects directory layout:
        dataset_dir/
            A/
                img1.jpg
                img2.jpg
            B/
                ...

    Provides train and validation generators.
    """

    def __init__(self, dataset_dir: str | None = None):
        self.dataset_dir = dataset_dir or CONFIG["dataset_dir"]
        self.image_size = CONFIG["image_size"]
        self.batch_size = CONFIG["batch_size"]
        self.val_split = CONFIG["validation_split"]

        # Training generator with orientation-safe augmentations
        self._train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=self.val_split,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=(0.8, 1.2),
            preprocessing_function=augment_blur_grayscale,
            fill_mode="nearest",
        )

        # Validation generator — only rescale, no augmentation
        self._val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=self.val_split,
        )

    def train_generator(self):
        """Return a Keras DirectoryIterator for training."""
        return self._train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=CONFIG["shuffle_train"],
            color_mode=CONFIG["color_mode"],
        )

    def val_generator(self):
        """Return a Keras DirectoryIterator for validation."""
        return self._val_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            color_mode=CONFIG["color_mode"],
        )

    def class_labels(self) -> list[str]:
        """Return ordered list of class label strings."""
        gen = self.train_generator()
        # class_indices is {label: index}
        return [k for k, _ in sorted(gen.class_indices.items(), key=lambda x: x[1])]
