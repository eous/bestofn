"""
Claude-specific generation and verification modules.

This package contains:
- generate.py: Best-of-N generation using Claude's extended thinking mode
- regen.py: Regeneration of specific splits/failed rows
- tool_executor.py: Tool calling execution loop (Claude format)
"""

# Main generation functions
from claude_gen.generate import (
    generate_candidates_claude,
    process_item,
    main,
)

# Regeneration
from claude_gen.regen import (
    load_parquet,
    filter_rows,
    extract_unique_queries,
)

# Tool execution
from claude_gen.tool_executor import (
    ClaudeToolExecutor,
    generate_candidates_with_tool_calling,
)

__all__ = [
    # Generate
    "generate_candidates_claude",
    "process_item",
    "main",
    # Regen
    "load_parquet",
    "filter_rows",
    "extract_unique_queries",
    # Tool executor
    "ClaudeToolExecutor",
    "generate_candidates_with_tool_calling",
]
