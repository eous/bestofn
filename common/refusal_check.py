#!/usr/bin/env python3
"""
Refusal Detection Utilities.

Provides a helper function to check for refusals in model responses.
Uses a two-pass approach: checks the answer first, then the full response
if no refusal is detected in the answer.
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers import RefusalClassifier
    from common.schema import RefusalDetection

logger = logging.getLogger(__name__)


def check_refusal(
    answer: str,
    full_text: Optional[str],
    classifier: 'RefusalClassifier',
) -> dict:
    """
    Check for refusals in model response using two-pass approach.

    First checks the answer text, then checks the full response if no
    refusal is found. Returns the result with highest confidence.

    Args:
        answer: The final answer text
        full_text: The full response text (raw_text or analysis + answer)
        classifier: RefusalClassifier instance from verifiers module

    Returns:
        Dict with keys:
        - is_refusal: bool
        - confidence: float
        - refusal_type: Optional[str]
        - matched_patterns: List[str]

    Example:
        result = check_refusal(answer, raw_text, refusal_classifier)
        refusal_detection = RefusalDetection(**result)
    """
    # Default result
    result = {
        "is_refusal": False,
        "confidence": 0.0,
        "refusal_type": None,
        "matched_patterns": [],
    }

    if not classifier:
        return result

    # First pass: check answer
    answer_result = classifier.classify(answer or "")

    # If refusal found in answer, use that result
    if answer_result.get("is_refusal", False):
        return {
            "is_refusal": answer_result["is_refusal"],
            "confidence": answer_result.get("confidence", 0.0),
            "refusal_type": answer_result.get("refusal_type"),
            "matched_patterns": answer_result.get("matched_patterns", []),
        }

    # Second pass: check full text if available
    if full_text:
        full_result = classifier.classify(full_text)

        # Use full text result if it has higher confidence refusal
        if full_result.get("is_refusal", False):
            if full_result.get("confidence", 0.0) > answer_result.get("confidence", 0.0):
                return {
                    "is_refusal": full_result["is_refusal"],
                    "confidence": full_result.get("confidence", 0.0),
                    "refusal_type": full_result.get("refusal_type"),
                    "matched_patterns": full_result.get("matched_patterns", []),
                }

    # No refusal detected - return answer result (may have low-confidence match)
    return {
        "is_refusal": answer_result.get("is_refusal", False),
        "confidence": answer_result.get("confidence", 0.0),
        "refusal_type": answer_result.get("refusal_type"),
        "matched_patterns": answer_result.get("matched_patterns", []),
    }


def build_refusal_detection(
    answer: str,
    full_text: Optional[str],
    classifier: 'RefusalClassifier',
) -> 'RefusalDetection':
    """
    Check for refusals and return a RefusalDetection schema object.

    Convenience wrapper that returns the schema object directly.

    Args:
        answer: The final answer text
        full_text: The full response text
        classifier: RefusalClassifier instance

    Returns:
        RefusalDetection schema object
    """
    from common.schema import RefusalDetection

    result = check_refusal(answer, full_text, classifier)
    return RefusalDetection(
        is_refusal=result["is_refusal"],
        confidence=result["confidence"],
        refusal_type=result["refusal_type"],
        matched_patterns=result["matched_patterns"],
    )
