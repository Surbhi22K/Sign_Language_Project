"""
Sign Language Decoding System — CLI Entry Point.

Usage:
    python main.py train          Train the model
    python main.py evaluate       Evaluate the trained model
    python main.py infer          Run inference on a test video
    python main.py convert        Convert model to TFLite
    python main.py all            Run full pipeline (train → evaluate → infer → convert)
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_code.utils.logger import get_logger

log = get_logger("main")


def cmd_train(args):
    """Train the model."""
    from new_code.training.train import train

    log.info("=" * 60)
    log.info("  TRAINING")
    log.info("=" * 60)
    history = train(dataset_dir=args.dataset)
    return history


def cmd_evaluate(args):
    """Evaluate the trained model."""
    from new_code.evaluation.evaluate import evaluate

    log.info("=" * 60)
    log.info("  EVALUATION")
    log.info("=" * 60)
    evaluate(model_path=args.model, dataset_dir=args.dataset)


def cmd_infer(args):
    """Run inference on a video."""
    from new_code.inference.frame_infer import infer_video

    log.info("=" * 60)
    log.info("  INFERENCE")
    log.info("=" * 60)
    result = infer_video(model_path=args.model, video_path=args.video)
    print(f"\n{'=' * 60}")
    print(f"  DECODED OUTPUT: {result}")
    print(f"{'=' * 60}\n")


def cmd_convert(args):
    """Convert model to TFLite."""
    from new_code.utils.convert import convert_to_tflite

    log.info("=" * 60)
    log.info("  MODEL CONVERSION → TFLite")
    log.info("=" * 60)
    convert_to_tflite(model_path=args.model)


def cmd_all(args):
    """Run the full pipeline."""
    from run_full_pipeline import run_all
    run_all()


def main():
    parser = argparse.ArgumentParser(
        description="Sign Language Decoding System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # Shared args
    def add_common(sub):
        sub.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
        sub.add_argument("--dataset", type=str, default=None, help="Path to dataset directory")
        sub.add_argument("--video", type=str, default=None, help="Path to input video")

    # train
    p_train = subparsers.add_parser("train", help="Train the model")
    add_common(p_train)
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model")
    add_common(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    # infer
    p_infer = subparsers.add_parser("infer", help="Run video inference")
    add_common(p_infer)
    p_infer.set_defaults(func=cmd_infer)

    # convert
    p_conv = subparsers.add_parser("convert", help="Convert model to TFLite")
    add_common(p_conv)
    p_conv.set_defaults(func=cmd_convert)

    # all
    p_all = subparsers.add_parser("all", help="Run full pipeline")
    add_common(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
