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
]
