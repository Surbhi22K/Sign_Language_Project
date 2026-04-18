"""
Hand detection and cropping using MediaPipe Hands.
Extracts the hand region from a video frame for more accurate classification.
"""

import cv2
import numpy as np
import mediapipe as mp

from new_code.utils.logger import get_logger

log = get_logger(__name__)


class HandDetector:
    """
    Detect and crop hand regions from frames using MediaPipe Hands.

    Usage:
        detector = HandDetector()
        cropped = detector.crop_hand(frame)  # returns cropped hand or full frame
        detector.close()
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        padding: float = 0.2,
    ):
        self.padding = padding
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detected_count = 0
        self._total_count = 0

    def crop_hand(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect hand in the frame and return a cropped region around it.

        If no hand is detected, returns the original frame.

        Args:
            frame: BGR image (H, W, 3), uint8 or float32.

        Returns:
            Cropped hand region (or full frame if no hand found).
        """
        self._total_count += 1
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        if frame.dtype == np.float32:
            rgb = (frame * 255).astype(np.uint8)
        else:
            rgb = frame.copy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            self._detected_count += 1
            hand = results.multi_hand_landmarks[0]
            
            # --- Get bounding box with padding & elbow extension ---
            x_coords = [lm.x for lm in hand.landmark]
            y_coords = [lm.y for lm in hand.landmark]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            
            px_x1 = int(x_min * w)
            px_x2 = int(x_max * w)
            px_y1 = int(y_min * h)
            px_y2 = int(y_max * h)

            # Estimate elbow position: wrist (0) to middle finger base (9)
            wrist = hand.landmark[0]
            middle_mcp = hand.landmark[9]
            
            # Vector pointing towards elbow
            dir_x = wrist.x - middle_mcp.x
            dir_y = wrist.y - middle_mcp.y
            
            # Elbow is roughly 2.5x the distance from wrist as the hand length
            elbow_x = int((wrist.x + dir_x * 2.5) * w)
            elbow_y = int((wrist.y + dir_y * 2.5) * h)
            
            # Include elbow in the bounding box
            px_x1 = min(px_x1, elbow_x)
            px_x2 = max(px_x2, elbow_x)
            px_y1 = min(px_y1, elbow_y)
            px_y2 = max(px_y2, elbow_y)

            # Add 75 pixels padding outside the hand/elbow
            pad = 75
            px_x1 -= pad
            px_y1 -= pad
            px_x2 += pad
            px_y2 += pad

            # Ensure square crop (to prevent distortion when resizing)
            box_w = px_x2 - px_x1
            box_h = px_y2 - px_y1
            
            if box_w > box_h:
                diff = box_w - box_h
                px_y1 -= diff // 2
                px_y2 += diff - (diff // 2)
            elif box_h > box_w:
                diff = box_h - box_w
                px_x1 -= diff // 2
                px_x2 += diff - (diff // 2)
                
            # Clamp to frame borders
            px_x1 = max(0, px_x1)
            px_y1 = max(0, px_y1)
            px_x2 = min(w, px_x2)
            px_y2 = min(h, px_y2)

            # Ensure valid crop
            if px_x2 > px_x1 + 10 and px_y2 > px_y1 + 10:
                return frame[px_y1:px_y2, px_x1:px_x2]

        # No hand detected — return full frame
        return frame

    def get_stats(self) -> dict:
        """Return detection statistics."""
        rate = (self._detected_count / self._total_count * 100) if self._total_count else 0
        return {
            "total_frames": self._total_count,
            "detected_frames": self._detected_count,
            "detection_rate": round(rate, 1),
        }

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
