"""
OpenAI-specific generation and verification modules.

This package contains:
- generate.py: Best-of-N generation using OpenAI Responses API
- regen.py: Regeneration of specific splits/failed rows
- tool_executor.py: Tool calling execution loop
"""

# Main generation functions
from openai_gen.generate import (
    generate_candidates,
    process_item,
    main,
)

# Regeneration
from openai_gen.regen import (
    load_parquet,
    filter_rows,
    extract_unique_queries,
)

# Tool execution
from openai_gen.tool_executor import (
    ToolExecutor,
    generate_candidates_with_tool_calling,
)

# Re-export LLM judge from common (for backwards compatibility)
from common.llm_judge import (
    LLMJudgeVerifier,
    get_llm_judge,
)

__all__ = [
    # Generate
    "generate_candidates",
    "process_item",
    "main",
    # Regen
    "load_parquet",
    "filter_rows",
    "extract_unique_queries",
    # Tool executor
    "ToolExecutor",
    "generate_candidates_with_tool_calling",
    # LLM judge (from common)
    "LLMJudgeVerifier",
    "get_llm_judge",
]
