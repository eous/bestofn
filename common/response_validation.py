#!/usr/bin/env python3
"""
Response Validation Utilities.

Provides response length limits and truncation to prevent:
- Memory issues from extremely long responses
- Data corruption in parquet storage
- Processing timeouts from oversized content

Also includes empty response detection.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Response length limits (characters)
MAX_ANSWER_LENGTH = 500_000  # 500K - generous limit for answers
MAX_ANALYSIS_LENGTH = 2_000_000  # 2M - very generous for reasoning traces
MAX_RAW_TEXT_LENGTH = 2_000_000  # 2M - same as analysis

# Truncation marker
TRUNCATION_MARKER = "\n[TRUNCATED]"


def truncate_if_needed(
    text: Optional[str],
    max_length: int,
    label: str = "text",
) -> Optional[str]:
    """
    Truncate text if it exceeds max_length.

    Args:
        text: Text to potentially truncate
        max_length: Maximum allowed length in characters
        label: Human-readable label for logging (e.g., "final_answer", "analysis")

    Returns:
        Original text if within limit, or truncated text with marker

    Example:
        answer = truncate_if_needed(answer, MAX_ANSWER_LENGTH, "final_answer")
    """
    if not text:
        return text

    if len(text) > max_length:
        logger.warning(
            f"Truncating {label} from {len(text)} to {max_length} chars"
        )
        return text[:max_length] + TRUNCATION_MARKER

    return text


def truncate_response_fields(
    final_answer: Optional[str] = None,
    analysis: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> tuple:
    """
    Truncate all response fields to their respective limits.

    Args:
        final_answer: The final answer text
        analysis: The analysis/reasoning text
        raw_text: The raw response text

    Returns:
        Tuple of (final_answer, analysis, raw_text) with truncation applied
    """
    return (
        truncate_if_needed(final_answer, MAX_ANSWER_LENGTH, "final_answer"),
        truncate_if_needed(analysis, MAX_ANALYSIS_LENGTH, "analysis"),
        truncate_if_needed(raw_text, MAX_RAW_TEXT_LENGTH, "raw_text"),
    )


def is_empty_response(
    answer: str,
    reasoning: Optional[str] = None,
) -> bool:
    """
    Check if a response is empty or truncated.

    A response is considered empty if the answer field has no content.
    Short answers like "4" or "Yes" are valid - only truly empty answers
    are flagged.

    Args:
        answer: The answer text to check
        reasoning: Optional reasoning text (for logging context)

    Returns:
        True if the response is empty
    """
    answer_length = len(answer.strip()) if answer else 0
    reasoning_length = len(reasoning.strip()) if reasoning else 0

    is_empty = answer_length == 0

    if is_empty:
        logger.warning(
            f"Empty response detected: answer_length={answer_length}, "
            f"reasoning_length={reasoning_length}"
        )

    return is_empty


def was_truncated(text: Optional[str]) -> bool:
    """
    Check if text was previously truncated.

    Args:
        text: Text to check

    Returns:
        True if text ends with truncation marker
    """
    if not text:
        return False
    return text.endswith(TRUNCATION_MARKER)
