"""
Sequence Decoder — converts raw per-frame predictions into final words/sentences.

Key operations:
    1. Sliding-window majority voting (smoothing).
    2. Collapse consecutive repeated labels.
    3. Assemble into final decoded string.
"""

from collections import Counter
from textblob import TextBlob

from new_code.utils.config import CONFIG
from new_code.utils.logger import get_logger

log = get_logger(__name__)


def majority_vote(predictions: list[str], window_size: int | None = None) -> list[str]:
    """
    Apply sliding-window majority voting to smooth noisy frame predictions.

    Args:
        predictions: List of predicted label strings, one per frame.
        window_size: Size of the sliding window (default from config).

    Returns:
        Smoothed list of labels (same length as input).
    """
    window_size = window_size or CONFIG["sliding_window_size"]

    if len(predictions) <= window_size:
        # Not enough frames for sliding window — return most common overall
        if not predictions:
            return []
        most_common = Counter(predictions).most_common(1)[0][0]
        return [most_common] * len(predictions)

    smoothed = []
    half_w = window_size // 2

    for i in range(len(predictions)):
        start = max(0, i - half_w)
        end = min(len(predictions), i + half_w + 1)
        window = predictions[start:end]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)

    return smoothed


def collapse_repeats(sequence: list[str]) -> list[str]:
    """
    Remove consecutive duplicate labels.

    Example:
        ['A', 'A', 'A', 'B', 'B', 'C'] → ['A', 'B', 'C']
    """
    if not sequence:
        return []

    collapsed = [sequence[0]]
    for label in sequence[1:]:
        if label != collapsed[-1]:
            collapsed.append(label)

    return collapsed


def decode_sequence(
    raw_predictions: list[str],
    window_size: int | None = None,
) -> str:
    """
    Full decoding pipeline: smooth → collapse → join.

    Args:
        raw_predictions: List of per-frame predicted label strings.
        window_size: Majority-vote window size.

    Returns:
        Final decoded string (space-separated labels).
    """
    if not raw_predictions:
        return ""

    # Step 1: Smooth via majority voting
    smoothed = majority_vote(raw_predictions, window_size)

    # Step 2: Collapse consecutive repeats
    collapsed = collapse_repeats(smoothed)

    # Step 3: Join into a single string
    raw_text = "".join(collapsed)
    
    # Step 4: NLP spelling correction using TextBlob
    # We pass the concatenated letters "MYNAM" to attempt correction to "MY NAME" or "MYNAME"
    if raw_text:
        decoded = str(TextBlob(raw_text).correct())
    else:
        decoded = ""

    log.info(
        "Decoded %d raw frames → %d smoothed → %d unique → '%s' (Corrected: '%s')",
        len(raw_predictions), len(smoothed), len(collapsed), raw_text, decoded,
    )
    return decoded
