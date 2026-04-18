"""
Test script — End-to-end video inference.

Usage:
    python test/test_video.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_code.inference.frame_infer import infer_video
from new_code.utils.config import get_model_path, get_test_video_path


def main():
    model_path = get_model_path()
    video_path = get_test_video_path()

    print(f"Model : {model_path}")
    print(f"Video : {video_path}")
    print("-" * 50)

    result = infer_video(model_path=model_path, video_path=video_path)

    print("=" * 50)
    print(f"DECODED OUTPUT: {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()
