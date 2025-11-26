"""
Common utilities shared between OpenAI and Claude generation pipelines.

This module contains API-agnostic code:
- Pydantic schemas for data validation
- Dataset loading utilities (Nemotron)
- AST syntax checking for code verification
- Shared generation utilities (question extraction, etc.)
"""

# Pydantic schemas
from common.schema import (
    BestOfNRecord,
    HarmonyMessage,
    ModelOutput,
    ReasoningStep,
    QualityMetrics,
    VerificationResults,
    RefusalDetection,
    PersonaEvaluation,
    DatasetMetadata,
)

# Dataset utilities
from common.nemotron_utils import (
    load_nemotron_split,
    extract_first_user_message,
    get_dataset_version,
)

# AST syntax checking
from common.ast_syntax_checker import (
    check_code_syntax,
    extract_code_from_text,
    check_python_syntax,
    check_javascript_syntax,
)

# Generation utilities
from common.generation_utils import (
    extract_question_from_row,
    extract_boxed_content,
    extract_ground_truth_from_message,
    extract_xml,
    parse_score_tag,
    extract_plan_steps,
    get_verifier,
    init_debug_logging,
    log_request_response,
    format_question,
    PROMPT_TEMPLATE,
)

# LLM Judge (uses Claude Sonnet 4.5)
from common.llm_judge import (
    LLMJudgeVerifier,
    get_llm_judge,
)

# API retry logic
from common.api_retry import (
    call_with_retry,
    is_retryable_error,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)

# Response validation
from common.response_validation import (
    truncate_if_needed,
    truncate_response_fields,
    is_empty_response,
    was_truncated,
    MAX_ANSWER_LENGTH,
    MAX_ANALYSIS_LENGTH,
    MAX_RAW_TEXT_LENGTH,
)

# Refusal checking
from common.refusal_check import (
    check_refusal,
    build_refusal_detection,
)

# Quality metrics
from common.quality_metrics import (
    compute_quality_metrics,
    compute_completeness_from_fields,
    compute_completeness_from_steps,
)

# LLM judge fallback
from common.llm_judge_fallback import (
    maybe_use_llm_judge,
    should_use_llm_judge,
    MIN_PRIMARY_CONFIDENCE,
    MIN_LLM_JUDGE_CONFIDENCE,
)

# Regen pipeline utilities
from common.regen_pipeline import (
    load_parquet,
    filter_rows,
    extract_unique_queries,
    merge_results,
    save_checkpoint,
    clean_nan_values,
    extract_question_from_input_messages,
)

__all__ = [
    # Schema
    "BestOfNRecord",
    "HarmonyMessage",
    "ModelOutput",
    "ReasoningStep",
    "QualityMetrics",
    "VerificationResults",
    "RefusalDetection",
    "PersonaEvaluation",
    "DatasetMetadata",
    # Dataset
    "load_nemotron_split",
    "extract_first_user_message",
    "get_dataset_version",
    # AST
    "check_code_syntax",
    "extract_code_from_text",
    "check_python_syntax",
    "check_javascript_syntax",
    # Generation utilities
    "extract_question_from_row",
    "extract_boxed_content",
    "extract_ground_truth_from_message",
    "extract_xml",
    "parse_score_tag",
    "extract_plan_steps",
    "get_verifier",
    "init_debug_logging",
    "log_request_response",
    "format_question",
    "PROMPT_TEMPLATE",
    # LLM Judge
    "LLMJudgeVerifier",
    "get_llm_judge",
    # API retry
    "call_with_retry",
    "is_retryable_error",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    # Response validation
    "truncate_if_needed",
    "truncate_response_fields",
    "is_empty_response",
    "was_truncated",
    "MAX_ANSWER_LENGTH",
    "MAX_ANALYSIS_LENGTH",
    "MAX_RAW_TEXT_LENGTH",
    # Refusal checking
    "check_refusal",
    "build_refusal_detection",
    # Quality metrics
    "compute_quality_metrics",
    "compute_completeness_from_fields",
    "compute_completeness_from_steps",
    # LLM judge fallback
    "maybe_use_llm_judge",
    "should_use_llm_judge",
    "MIN_PRIMARY_CONFIDENCE",
    "MIN_LLM_JUDGE_CONFIDENCE",
    # Regen pipeline
    "load_parquet",
    "filter_rows",
    "extract_unique_queries",
    "merge_results",
    "save_checkpoint",
    "clean_nan_values",
    "extract_question_from_input_messages",
]
