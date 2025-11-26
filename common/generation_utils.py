"""
Shared generation utilities for OpenAI and Claude pipelines.

Contains API-agnostic functions:
- Question extraction from dataset rows
- Ground truth extraction
- Boxed content parsing (LaTeX)
- XML tag extraction
- Verifier selection
- Debug logging helpers
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Question Extraction
# =============================================================================

def extract_question_from_row(row: Dict[str, Any], min_len: int = 5) -> Optional[str]:
    """
    Extract the first non-empty user message from row["messages"].

    Works with both Nemotron v1 (object format) and v2 (dict format).

    Args:
        row: Dataset row with 'messages' field
        min_len: Minimum message length to accept (default: 5)

    Returns:
        First user message content, or None if invalid/not found
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return None

    for msg in msgs:
        # Handle both dict (v2) and object (v1) formats
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        elif hasattr(msg, 'role') and hasattr(msg, 'content'):
            # v1 format: Message objects with attributes
            role = msg.role
            content = msg.content
        else:
            continue

        if isinstance(role, str) and role.lower() == "user" and isinstance(content, str):
            text = content.strip()
            if text and text != "-" and len(text) >= min_len:
                return text
    return None


# =============================================================================
# Ground Truth Extraction
# =============================================================================

def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from \\boxed{...} with proper handling of nested braces.

    Handles arbitrarily nested braces like:
    - \\boxed{5}
    - \\boxed{\\frac{1}{2}}
    - \\boxed{\\frac{\\sqrt{a^{2}+b^{2}}}{c}}

    Args:
        text: Text containing \\boxed{...}

    Returns:
        Content inside \\boxed{} or None if not found
    """
    # Find start of \boxed{
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None

    start = match.end()  # Position right after \boxed{
    depth = 1
    pos = start

    while pos < len(text) and depth > 0:
        char = text[pos]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        # Successfully found matching brace
        return text[start:pos-1]  # Exclude the final }

    return None  # Unmatched braces


def extract_ground_truth_from_message(msg: Any) -> Optional[str]:
    """
    Extract ground truth answer from an assistant message.

    Handles both dict-style messages and object-style messages.
    Extracts \\boxed{} content if present, otherwise uses full content.

    Args:
        msg: Message object (dict or object with role/content attributes)

    Returns:
        Extracted ground truth string or None
    """
    # Get role and content based on message type
    if isinstance(msg, dict):
        role = msg.get("role", "")
        content = msg.get("content", "")
    elif hasattr(msg, 'role') and hasattr(msg, 'content'):
        role = msg.role
        content = msg.content if msg.content else ""
    else:
        return None

    # Only process assistant messages
    if role != "assistant" or not content:
        return None

    # Try to extract \boxed{} answer (handles nested braces)
    extracted = extract_boxed_content(content)
    if extracted:
        logger.debug(f"Extracted \\boxed{{}} from ground truth: {extracted[:100]}...")
        return extracted

    # Fallback: use full content
    logger.debug(f"No \\boxed{{}} found, using full ground truth: {content[:100]}...")
    return content


# =============================================================================
# XML Tag Extraction
# =============================================================================

def extract_xml(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags, with fallback for unclosed tags.

    Handles:
    - Normal: <tag>content</tag>
    - Unclosed at end: <tag>content (end of string)
    - Case insensitive matching

    Args:
        text: Text containing XML tags
        tag: Tag name to extract

    Returns:
        Extracted content or empty string
    """
    # First try standard closed tag pattern
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: look for unclosed tag at end of text
    # This captures content from opening tag to end of string
    match = re.search(rf"<{tag}>(.*)$", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Only return if there's actual content and no other opening tag follows
        if content and not re.match(r"^\s*<\w+>", content):
            return content

    return ""


def parse_score_tag(text: str) -> Optional[float]:
    """Parse <score> tag content into float in [0, 5] if possible."""
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if not m:
        return None
    try:
        val = float(m.group(0))
        return max(0.0, min(5.0, val))
    except Exception:
        return None


def extract_plan_steps(plan_text: str) -> List[str]:
    """Split plan into steps, using <step> tags if present, else lines."""
    if not plan_text:
        return []
    step_matches = re.findall(r"<step>(.*?)</step>", plan_text, re.DOTALL | re.IGNORECASE)
    if step_matches:
        return [s.strip() for s in step_matches if s.strip()]
    # Fallback: each non-empty line is a step
    lines = [ln.strip(" -\t") for ln in plan_text.splitlines()]
    return [ln for ln in lines if ln]


# =============================================================================
# Verifier Selection
# =============================================================================

def get_verifier(split: str, row: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Factory to select the correct verifier based on data split + row.

    Uses enhanced verifier system from verifiers module.

    Args:
        split: Dataset split (math, code, tool_calling)
        row: Dataset row

    Returns:
        Tuple of (verifier_instance, spec_dict)
    """
    from verifiers import get_verifier_for_split, load_config as load_verifier_config

    spec: Dict[str, Any] = {}

    # Try to find ground truth in common dataset columns
    for col in ["solution", "ground_truth", "answer", "output"]:
        if row.get(col) is not None:
            spec["ground_truth"] = row[col]
            break

    # For Nemotron v2: Extract ground truth from assistant message
    if "ground_truth" not in spec and "messages" in row:
        msgs = row.get("messages", [])
        for msg in msgs:
            ground_truth = extract_ground_truth_from_message(msg)
            if ground_truth:
                spec["ground_truth"] = ground_truth
                break

    # Load verifier config
    try:
        verifier_config = load_verifier_config()
        config_dict = verifier_config.get_verifier_config(split) if verifier_config else None
    except Exception as e:
        logger.warning(f"Could not load verifier config: {e}, using defaults")
        config_dict = None

    # Get appropriate verifier for this split
    verifier = get_verifier_for_split(split, config=config_dict)
    return verifier, spec


# =============================================================================
# Debug Logging
# =============================================================================

# Global debug log state (set by each vendor's generate.py)
DEBUG_LOG_FILE: Optional[str] = None
DEBUG_LOG_LOCK: Optional[asyncio.Lock] = None


def init_debug_logging(file_path: str, metadata: Dict[str, Any]) -> None:
    """
    Initialize debug logging to file.

    Args:
        file_path: Path to debug log file
        metadata: Experiment metadata to write as header
    """
    global DEBUG_LOG_FILE, DEBUG_LOG_LOCK
    DEBUG_LOG_FILE = file_path
    DEBUG_LOG_LOCK = asyncio.Lock()

    with open(DEBUG_LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"DEBUG LOG - Generated at {datetime.now().isoformat()}\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 80 + "\n\n")

    logger.info(f"Debug logging enabled: {DEBUG_LOG_FILE}")


async def log_request_response(
    query_id: str,
    question: str,
    request_body: Dict[str, Any],
    responses: List[str],
    error: Optional[str] = None,
) -> None:
    """
    Log raw request and response to debug file.

    Thread-safe via asyncio lock.

    Args:
        query_id: Unique query identifier
        question: The original question text
        request_body: Dict with model, messages, n, temperature, max_tokens
        responses: List of response texts (one per candidate)
        error: Optional error message if generation failed
    """
    if not DEBUG_LOG_FILE:
        return

    global DEBUG_LOG_LOCK
    # Lazy initialization of async lock - must be done in async context
    if DEBUG_LOG_LOCK is None:
        DEBUG_LOG_LOCK = asyncio.Lock()

    async with DEBUG_LOG_LOCK:
        try:
            with open(DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"QUERY ID: {query_id}\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

                f.write("QUESTION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{question}\n\n")

                f.write("REQUEST BODY:\n")
                f.write("-" * 80 + "\n")
                f.write(json.dumps(request_body, indent=2, ensure_ascii=False))
                f.write("\n\n")

                if error:
                    f.write("ERROR:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{error}\n\n")
                else:
                    f.write(f"RESPONSES ({len(responses)} candidates):\n")
                    f.write("-" * 80 + "\n")
                    for i, response in enumerate(responses):
                        f.write(f"\n--- Candidate {i} ---\n")
                        f.write(response)
                        f.write("\n")
                    f.write("\n")

                f.write("\n\n")
        except Exception as e:
            logger.warning(f"Failed to write debug log: {e}")


# =============================================================================
# Prompt Template
# =============================================================================

PROMPT_TEMPLATE = """{question}

Show your step-by-step reasoning. Maintain your character/persona.

CRITICAL FORMATTING: Your final_answer field must contain ONLY the boxed result for math (e.g., \\boxed{{5}}, \\boxed{{\\frac{{1}}{{2}}}}, \\boxed{{Monday}}). You may add persona commentary before or after the box, but the \\boxed{{}} MUST be present.
"""


def format_question(question: str) -> str:
    """Format question with standard prompt template."""
    return PROMPT_TEMPLATE.format(question=question)
