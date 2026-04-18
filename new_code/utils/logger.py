"""
Logging utility for the Sign Language Decoding System.
Provides a pre-configured logger with console + optional file output.
"""

import logging
import os
import sys


def get_logger(name: str = "signlang", log_file: str | None = None) -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: Logger name (module name recommended).
        log_file: Optional path to a log file.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
