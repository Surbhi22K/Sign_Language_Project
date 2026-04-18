"""
Frame-level inference on video files with hand detection + cropping.
"""

import cv2
import numpy as np

from new_code.data.transforms import resize_and_normalize
from new_code.inference.hand_detector import HandDetector
from new_code.inference.sequence_decoder import decode_sequence
from new_code.utils.checkpoint import load_checkpoint
from new_code.utils.config import CONFIG, get_model_path, get_test_video_path
from new_code.utils.logger import get_logger

log = get_logger(__name__)


def infer_video(
    model_path: str | None = None,
    video_path: str | None = None,
    labels: list[str] | None = None,
    confidence_threshold: float = 0.95,
) -> str:
    """
    Run end-to-end inference on a video file with hand detection.

    Pipeline:
        1. Load trained model.
        2. Open video, initialize hand detector.
        3. For each frame:
           a. Detect and crop hand region (MediaPipe).
           b. Resize + normalize the crop.
           c. Run model prediction.
        4. Decode the sequence of predictions.

    Args:
        model_path: Path to saved model checkpoint.
        video_path: Path to input video file.
        labels: Ordered list of class label strings.

    Returns:
        Decoded output string.
    """
    model_path = model_path or get_model_path()
    video_path = video_path or get_test_video_path()

    # Load model
    model = load_checkpoint(model_path)

    # Default labels (A–Y, excluding J which requires motion)
    if labels is None:
        labels = [chr(c) for c in range(ord("A"), ord("Z")) if chr(c) != "J"]
        labels = labels[: model.output_shape[-1]]

    log.info("Running inference — model=%s  video=%s", model_path, video_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video has %d frames", total_frames)

    image_size = tuple(CONFIG["image_size"])

    # Per-frame prediction with hand detection
    raw_predictions = []

    with HandDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        padding=0.25,
    ) as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Step 1: Detect and crop hand
            hand_crop = detector.crop_hand(frame)
            # Step 2: Resize + normalize
            processed = resize_and_normalize(hand_crop, image_size)
            # Step 3: Predict
            img = np.expand_dims(processed, axis=0)  # (1, H, W, C)
            pred = model.predict(img, verbose=0)
            class_idx = int(np.argmax(pred, axis=1)[0])
            confidence = float(np.max(pred))
            
            if confidence >= confidence_threshold:
                letter = labels[class_idx] if class_idx < len(labels) else "?"
                raw_predictions.append(letter)
                print(f"Frame {frame_idx:04d} -> Confident: {letter} ({confidence*100:.1f}%)")

        stats = detector.get_stats()

    cap.release()

    log.info(
        "Processed %d frames — hand detected in %d (%.1f%%)",
        stats["total_frames"], stats["detected_frames"], stats["detection_rate"],
    )
    log.info("Raw predictions (first 20): %s", raw_predictions[:20])

    # Step 4: Decode
    decoded = decode_sequence(raw_predictions)
    log.info("✅ Final output: %s", decoded)

    return decoded
