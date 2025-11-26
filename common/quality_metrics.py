#!/usr/bin/env python3
"""
Quality Metrics Utilities.

Provides helper functions to build QualityMetrics for responses.
Used by both OpenAI and Claude generators.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from common.schema import QualityMetrics

logger = logging.getLogger(__name__)


def compute_quality_metrics(
    answer: str,
    reasoning: Optional[str] = None,
    plan: Optional[str] = None,
    steps: Optional[List] = None,
    is_empty: bool = False,
    completeness_score: Optional[float] = None,
) -> 'QualityMetrics':
    """
    Compute quality metrics for a model response.

    Args:
        answer: The final answer text
        reasoning: The reasoning/analysis text
        plan: Optional plan text (OpenAI structured output)
        steps: Optional list of reasoning steps
        is_empty: Whether the response is empty (from is_empty_response check)
        completeness_score: Optional pre-computed completeness score.
            If not provided, computed as:
            - 1.0 if steps and answer present
            - 0.5 otherwise

    Returns:
        QualityMetrics schema object

    Example:
        quality = compute_quality_metrics(
            answer=final_answer,
            reasoning=analysis,
            steps=output.steps,
            is_empty=is_empty_response(answer, analysis),
        )
    """
    from common.schema import QualityMetrics

    answer_length = len(answer.strip()) if answer else 0
    reasoning_length = len(reasoning.strip()) if reasoning else 0
    plan_length = len(plan.strip()) if plan else 0
    total_length = answer_length + reasoning_length + plan_length

    # Default completeness score if not provided
    if completeness_score is None:
        has_steps = bool(steps) if steps is not None else reasoning_length > 0
        completeness_score = 1.0 if (has_steps and answer_length > 0) else 0.5

    return QualityMetrics(
        answer_length=answer_length,
        reasoning_length=reasoning_length,
        plan_length=plan_length,
        total_response_length=total_length,
        has_reasoning=reasoning_length > 0,
        has_plan=plan_length > 0,
        is_short_answer=answer_length < 50,
        is_substantive=reasoning_length > 100 or answer_length > 50,
        is_empty=is_empty,
        completeness_score=completeness_score,
    )


def compute_completeness_from_fields(
    normalized_query: Optional[str] = None,
    plan: Optional[str] = None,
    reasoning: Optional[str] = None,
    answer: Optional[str] = None,
    evaluation: Optional[str] = None,
) -> float:
    """
    Compute completeness score from field presence (OpenAI non-structured).

    Args:
        normalized_query: Extracted normalized query
        plan: Extracted plan text
        reasoning: Extracted reasoning text
        answer: Final answer text
        evaluation: Self-evaluation text

    Returns:
        Completeness score between 0.0 and 1.0
    """
    fields_present = sum([
        bool(normalized_query),
        bool(plan),
        bool(reasoning),
        bool(answer),
        bool(evaluation),
    ])
    return fields_present / 5.0


def compute_completeness_from_steps(
    steps: Optional[List] = None,
    final_answer: Optional[str] = None,
) -> float:
    """
    Compute completeness score from structured output (OpenAI structured).

    Args:
        steps: List of reasoning steps
        final_answer: Final answer text

    Returns:
        Completeness score between 0.0 and 1.0
    """
    if not steps:
        return 0.5 if final_answer else 0.0

    # Steps quality + answer presence
    step_score = min(1.0, len(steps) / max(1, len(steps)))  # Always 1.0 if steps exist
    answer_score = 1.0 if final_answer else 0.0
    return (step_score + answer_score) / 2.0
