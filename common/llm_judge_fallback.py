#!/usr/bin/env python3
"""
LLM Judge Fallback Logic.

Provides a unified interface for using LLM-as-judge verification
when primary verification has low confidence.
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Confidence thresholds
MIN_PRIMARY_CONFIDENCE = 0.5  # Use LLM judge if primary confidence below this
MIN_LLM_JUDGE_CONFIDENCE = 0.7  # Accept LLM judge result if confidence above this


async def maybe_use_llm_judge(
    enabled: bool,
    question: str,
    answer: str,
    ground_truth: Optional[str],
    split: str,
    is_verified: bool,
    confidence: float,
    is_empty: bool,
    provider: str,
    api_key: str,
) -> Tuple[bool, float, str, str, bool, bool]:
    """
    Optionally use LLM judge fallback for verification.

    Args:
        enabled: Whether LLM judge fallback is enabled
        question: The original question
        answer: The candidate answer to verify
        ground_truth: The expected correct answer (if available)
        split: The dataset split (math, code, tool_calling)
        is_verified: Current verification status from primary verifier
        confidence: Current confidence from primary verifier
        is_empty: Whether the response is empty
        provider: LLM provider ("openai" or "claude")
        api_key: API key for the provider

    Returns:
        Tuple of (is_verified, confidence, explanation, verifier_name, llm_judge_used, llm_judge_failed)

    Example:
        (is_verified, confidence, explanation, verifier_name,
         llm_judge_used, llm_judge_failed) = await maybe_use_llm_judge(
            enabled=args.llm_judge_fallback,
            question=question,
            answer=final_answer,
            ground_truth=ground_truth,
            split=split,
            is_verified=is_verified,
            confidence=confidence,
            is_empty=is_empty,
            provider="openai",
            api_key=args.api_key,
        )
    """
    # Import here to avoid circular dependency
    from common.llm_judge import get_llm_judge

    llm_judge_used = False
    llm_judge_failed = False
    explanation = ""
    verifier_name = ""

    # Early exit if not enabled or no ground truth
    if not enabled or not ground_truth:
        return is_verified, confidence, explanation, verifier_name, llm_judge_used, llm_judge_failed

    # Skip for empty responses - no point judging an empty answer
    if is_empty:
        return is_verified, confidence, explanation, verifier_name, llm_judge_used, llm_judge_failed

    # Determine if we should use LLM judge:
    # 1. Primary verification failed with low confidence, OR
    # 2. Split is code/tool_calling (need semantic verification)
    should_use_llm = (
        (not is_verified and confidence < MIN_PRIMARY_CONFIDENCE) or
        (split in ['code', 'tool_calling'])
    )

    if not should_use_llm:
        return is_verified, confidence, explanation, verifier_name, llm_judge_used, llm_judge_failed

    logger.debug(f"Using LLM judge for {split} (primary confidence={confidence})...")
    llm_judge_used = True

    try:
        llm_judge = get_llm_judge(provider=provider, api_key=api_key)
        llm_result = await llm_judge.verify(
            question=question,
            candidate_answer=answer,
            ground_truth=ground_truth,
            split=split,
        )

        # Use LLM judge result if high confidence
        if llm_result["confidence"] > MIN_LLM_JUDGE_CONFIDENCE:
            is_verified = llm_result["is_correct"]
            confidence = llm_result["confidence"]
            explanation = llm_result["explanation"]
            verifier_name = llm_result["verifier_name"]
            logger.debug(f"LLM judge result: {is_verified} (confidence={confidence})")
        else:
            logger.debug(f"LLM judge confidence too low: {llm_result['confidence']}")

    except Exception as e:
        llm_judge_failed = True
        logger.warning(f"LLM judge fallback failed: {e}")

    return is_verified, confidence, explanation, verifier_name, llm_judge_used, llm_judge_failed


def should_use_llm_judge(
    is_verified: bool,
    confidence: float,
    split: str,
    has_ground_truth: bool,
    is_empty: bool,
) -> bool:
    """
    Determine whether LLM judge should be used.

    Args:
        is_verified: Current verification status
        confidence: Current verification confidence
        split: The dataset split
        has_ground_truth: Whether ground truth is available
        is_empty: Whether response is empty

    Returns:
        True if LLM judge should be used
    """
    if not has_ground_truth or is_empty:
        return False

    return (
        (not is_verified and confidence < MIN_PRIMARY_CONFIDENCE) or
        (split in ['code', 'tool_calling'])
    )
