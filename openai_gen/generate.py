#!/usr/bin/env python3
"""
Best-of-N Generator with Verification Scaffolding (Patched + Planner-Friendly)
------------------------------------------------------------------------------

Architectural Pattern: Compute-Over-Data / Iterative Rejection Sampling

Goal:
    Generate high-quality training data by producing N solutions per query and
    filtering them via deterministic verification (Math / Code / ToolCall),
    while also capturing rich planning fields for training a local planner.

Key Features:
1. Generates N candidates per query (using 'n' param when supported, or
   sequential fallback for models that don't support n>1, e.g., some
   reasoning models like o1/o3).
2. Verifies candidates using:
   - MathVerifier: numeric checks against ground truth fields.
   - CodeVerifier: executes Python code (unsafe demo sandbox).
   - ToolVerifier: basic JSON parsing for tool-calling answers.
3. Extracts richer planner/solver tags:
   - <normalized_query>...</normalized_query>
   - <plan>...</plan>
   - <reasoning>...</reasoning>
   - <answer>...</answer>
   - <evaluation>...</evaluation>
   - <score>...</score>
4. Saves all (query, candidate) pairs to a Parquet file for high-performance
   training data ingestion.

Usage Example:
    python generate_best_of_n.py \
        --dataset nvidia/Nemotron-Post-Training-Dataset-v1 \
        --splits math,code,tool_calling \
        --max-queries 50 \
        --num-candidates 4 \
        --model gpt-5.1 \
        --output candidates.parquet \
        --streaming \
        --concurrency 8

Notes:
- This script is intended as a scaffold. The verifiers are minimal examples;
  you should replace them with production-safe implementations for your tasks.
- For very large runs (e.g., 1M+ queries), you will likely want to add
  chunked writing to Parquet rather than keeping all_results in memory.
"""

import os
import re
import sys
import json
import asyncio
import argparse
import logging
import traceback
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, BadRequestError
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BestOfN")

# -----------------------------------------------------------------------------
# Verification Logic (Enhanced - Using Secure Verifier Module)
# -----------------------------------------------------------------------------

# Add parent directory to path for common and verifiers modules
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import shared utilities from common module
from common.schema import (
    BestOfNRecord,
    HarmonyMessage as SchemaHarmonyMessage,
    ModelOutput,
    QualityMetrics,
    VerificationResults,
    RefusalDetection,
    PersonaEvaluation,
)
from common.nemotron_utils import load_nemotron_split
from common.ast_syntax_checker import check_code_syntax, extract_code_from_text
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
from common.api_retry import call_with_retry
from common.response_validation import (
    MAX_ANSWER_LENGTH,
    MAX_ANALYSIS_LENGTH,
    truncate_if_needed,
    is_empty_response,
)
from common.refusal_check import build_refusal_detection

# Import verifiers
from verifiers import (
    get_verifier_for_split,
    load_config as load_verifier_config,
    VerificationResult as EnhancedVerificationResult, RefusalClassifier
)
refusal_classifier = RefusalClassifier()
ENHANCED_VERIFIERS_AVAILABLE = True
logger.info("Enhanced verification system loaded (secure Docker-based execution)")

# Import shared LLM judge (uses Claude Sonnet 4.5)
from common.llm_judge import get_llm_judge
from openai_gen.tool_executor import generate_candidates_with_tool_calling, ToolExecutor

# Import OpenAI Harmony for multi-channel responses (optional)
try:
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
        Role,
        SystemContent,
        ReasoningEffort,
        HarmonyEncodingName,
        RenderConversationConfig,
        load_harmony_encoding,
    )
    HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    HARMONY_AVAILABLE = True
    logger.info("OpenAI Harmony available - will use GPT-OSS multi-channel format")
except ImportError as e:
    HARMONY_AVAILABLE = False
    HARMONY_ENCODING = None
    logger.warning(f"OpenAI Harmony not available: {e}. Multi-channel parsing disabled.")


# -----------------------------------------------------------------------------
# Helper Functions (Note: Most moved to common/generation_utils.py)
# -----------------------------------------------------------------------------

# Note: API retry logic moved to common/api_retry.py (call_with_retry)
# Note: Response validation moved to common/response_validation.py


# Note: get_verifier and PROMPT_TEMPLATE are imported from common.generation_utils

# -----------------------------------------------------------------------------
# Generation Logic (Planner + Reasoning + Answer + Evaluation + Score)
# -----------------------------------------------------------------------------

HARMONY_INPUT_MESSAGES_CACHE = None  # Schema messages for input_messages field

def build_input_messages_cache(persona: Optional[str] = None) -> List[SchemaHarmonyMessage]:
    """
    Build cached input messages (system + developer) for training data.

    Args:
        persona: Optional persona/instructions for developer message

    Returns:
        List of SchemaHarmonyMessage for system and developer messages
    """
    messages = []

    # System message (minimal)
    messages.append(SchemaHarmonyMessage(
        role="system",
        content="You are a helpful assistant.",
        channel=None,
    ))

    # Developer message with persona (if provided)
    if persona:
        messages.append(SchemaHarmonyMessage(
            role="developer",
            content=persona,
            channel=None,
        ))

    return messages

def build_harmony_prompt_prefix(persona: Optional[str] = None, experiment_date: Optional[str] = None) -> Conversation:
    """
    Build Harmony system+developer message prefix.

    Args:
        persona: Optional persona/instructions for developer message
        experiment_date: Fixed date for entire experiment run

    Returns:
        Tuple of (formatted_text, schema_messages) for sending and storage
    """
    from datetime import datetime
    if experiment_date is None:
        experiment_date = datetime.now().strftime("%Y-%m-%d")

    # Build system message with metadata
    system_content = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date(experiment_date)
    )

    developer_content = DeveloperContent.new().with_instructions(persona)


    # Render to text (we'll send as "user" message to simple server)
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.DEVELOPER, developer_content)
        ]
    )
    return convo


def parse_harmony_response(text: str) -> Dict[str, List[str]]:
    """
    Parse Harmony multi-channel response from model output.

    Args:
        text: Raw model output with Harmony special tokens

    Returns:
        Dict with keys 'analysis', 'commentary', 'final' containing channel content lists
    """
    # Early return if Harmony not available
    if not HARMONY_AVAILABLE or HARMONY_ENCODING is None:
        logger.debug("Harmony not available, skipping channel parsing")
        return {'analysis': [], 'commentary': [], 'final': []}

    logger.info("Parsing Harmony response channels")
    logger.debug("Raw text:\n%s", text[:500] if len(text) > 500 else text)
    try:
        # Tokenize the response
        token_ids = HARMONY_ENCODING.encode(text)

        # Parse messages from tokens
        messages = HARMONY_ENCODING.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)

        # Group by channel
        channels = {'analysis': [], 'commentary': [], 'final': []}
        for msg in messages:
            if msg.role == Role.ASSISTANT:
                channel = msg.channel or 'final'
                if channel in channels:
                    # Handle both string and list content from Harmony library
                    content = msg.content
                    if isinstance(content, list):
                        content = ''.join(content)
                    elif not isinstance(content, str):
                        content = str(content)
                    channels[channel].append(content)

        return channels

    except Exception as e:
        logger.debug(f"Harmony parsing failed: {e}")
        return {'analysis': [], 'commentary': [], 'final': []}


# Note: extract_xml, parse_score_tag, extract_plan_steps imported from common.generation_utils
# Note: log_request_response imported from common.generation_utils (uses init_debug_logging)


async def generate_candidates(
    client: AsyncOpenAI,
    model: str,
    question: str,
    n: int,
    sem: asyncio.Semaphore,
    temperature: float = 1.0,
    max_tokens: int = 70000,
    persona: Optional[str] = "",
    query_id: Optional[str] = None,
    use_structured_output: bool = True,
    split: Optional[str] = None,
    row_metadata: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """
    Generates N candidates for a question using OpenAI responses API.

    Args:
        persona: Optional persona/instructions for developer message
        query_id: Optional query ID for debug logging
        use_structured_output: Whether to use structured outputs with ModelOutput schema
        split: Dataset split (e.g., 'tool_calling') for special handling
        row_metadata: Row metadata for tool extraction (tool_calling split)

    Returns:
        List of dicts with keys 'analysis' and 'final_answer', or 'raw_text' if not structured
    """
    # Build input messages cache (for training data) on first call
    global HARMONY_INPUT_MESSAGES_CACHE
    if HARMONY_INPUT_MESSAGES_CACHE is None and persona:
        HARMONY_INPUT_MESSAGES_CACHE = build_input_messages_cache(persona)
        logger.info(f"Built input messages cache with {len(HARMONY_INPUT_MESSAGES_CACHE)} messages")

    # Special handling for tool_calling split
    if split == 'tool_calling' and row_metadata:
        logger.info(f"Tool-calling split detected for query {query_id}")
        # Extract tools from metadata
        tools = ToolExecutor.extract_tools_from_metadata(row_metadata)

        if tools:
            logger.info(f"Found {len(tools)} tool definitions - using tool executor")
            try:
                # Use tool executor for tool-calling queries
                # Pass sem for rate limiting (consistent with claude_gen pattern)
                results = await generate_candidates_with_tool_calling(
                    client=client,
                    model=model,
                    question=question,
                    metadata=row_metadata,
                    n=n,
                    persona=persona,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_iterations=3,
                    sem=sem,
                )

                # Log for debugging (log_request_response handles debug_log check internally)
                if query_id:
                    raw_responses = [r.get('raw_text', r.get('final_answer', '')) for r in results]
                    request_body = {
                        "model": model,
                        "instructions": persona,
                        "input": question,
                        "tools": [t.get('function', {}).get('name') for t in tools],
                        "reasoning": {"effort": "medium"},
                        "max_output_tokens": max_tokens,
                    }
                    await log_request_response(query_id, question, request_body, raw_responses)

                return results

            except Exception as e:
                logger.error(f"Tool executor failed: {e}, falling back to standard generation")
                # Fall through to standard generation
        else:
            logger.warning(f"No tools found in metadata for tool_calling split - using standard generation")

    # Format question
    user_content = PROMPT_TEMPLATE.format(question=question)

    # Build request body for debug logging
    request_body = {
        "model": model,
        "instructions": persona,
        "input": user_content,
        "reasoning": {"effort": "medium"},
        "max_output_tokens": max_tokens,
        "text_format": "ModelOutput" if use_structured_output else None,
    }

    async with sem:
        try:
            # responses API doesn't support n>1, so we loop for multiple candidates
            results: List[Dict[str, str]] = []
            for i in range(n):
                logger.info(f"Generating candidate {i+1}/{n} for query {query_id}")

                if use_structured_output:
                    # Use .parse() method with text_format for structured outputs
                    # Wrapped in retry helper for resilience
                    response = await call_with_retry(
                        lambda: client.responses.parse(
                            model=model,
                            instructions=persona if persona else None,
                            input=user_content,
                            reasoning={"effort": "medium"},
                            max_output_tokens=max_tokens,
                            text_format=ModelOutput,
                        )
                    )

                    # Extract structured output
                    output = response.output_parsed
                    if output:
                        # Structured output successfully parsed
                        # Convert steps to dict format for downstream processing
                        steps_dicts = [
                            {'explanation': step.explanation, 'output': step.output}
                            for step in output.steps
                        ]
                        results.append({
                            'steps': steps_dicts,
                            'final_answer': output.final_answer,
                            'raw_text': response.output_text,  # Keep raw text for debugging
                        })
                    else:
                        # Fallback if parsing failed
                        logger.warning(f"Structured output parsing failed for candidate {i+1}, using raw text")
                        results.append({
                            'analysis': '',
                            'final_answer': response.output_text,
                            'raw_text': response.output_text,
                        })
                else:
                    # Fallback to raw text output
                    # Wrapped in retry helper for resilience
                    response = await call_with_retry(
                        lambda: client.responses.create(
                            model=model,
                            instructions=persona if persona else None,
                            input=user_content,
                            reasoning={"effort": "medium"},
                            max_output_tokens=max_tokens,
                        )
                    )

                    results.append({
                        'analysis': '',
                        'final_answer': response.output_text,
                        'raw_text': response.output_text,
                    })

            # Log request/response (log_request_response handles debug_log check internally)
            if query_id:
                # Extract raw_text for logging
                raw_responses = [r.get('raw_text', r.get('final_answer', '')) for r in results]
                await log_request_response(query_id, question, request_body, raw_responses)

            return results

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # Log error (log_request_response handles debug_log check internally)
            if query_id:
                await log_request_response(query_id, question, request_body, [], error=error_msg)

            return []


# -----------------------------------------------------------------------------
# Main Per-Item Processing
# -----------------------------------------------------------------------------

def extract_harmony_channels(text: str) -> Dict[str, str]:
    """
    Extract Harmony channel content from response.

    Parses Harmony special tokens to extract analysis, commentary, and final channels.
    Returns content joined as strings (generator joins multiple messages per channel).

    Args:
        text: Raw model output with Harmony special tokens

    Returns:
        Dict with keys 'analysis', 'commentary', 'final' (empty string if not found)
    """
    # Use Harmony library to parse channels
    channels_raw = parse_harmony_response(text)

    # Join list content into strings for easier handling
    return {
        'analysis': '\n'.join(channels_raw['analysis']),
        'commentary': '\n'.join(channels_raw['commentary']),
        'final': '\n'.join(channels_raw['final']),
    }


# Note: extract_question_from_row imported from common.generation_utils

async def process_item(
    row: Dict[str, Any],
    split: str,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """
    For a single dataset row:
      - Extract question
      - Generate N candidates
      - Parse planner/solving tags
      - Run verifier on each candidate
      - Return a list of candidate records
    """
    # 1. Extract Question and Query ID
    question = extract_question_from_row(row, min_len=args.min_query_chars)
    if not question:
        return []

    query_id = row.get("uuid") or row.get("id") or row.get("uid")
    query_id = str(query_id) if query_id is not None else None

    # 2. Generate N Candidates
    raw_outputs = await generate_candidates(
        client=client,
        model=args.model,
        question=question,
        n=args.num_candidates,
        sem=sem,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        persona=getattr(args, 'persona', None),
        query_id=query_id,
        use_structured_output=getattr(args, 'structured_output', True),
        split=split,
        row_metadata=row.get('metadata'),
    )
    if not raw_outputs:
        return []

    # 2.a Print raw outputs for debugging
    for i, out in enumerate(raw_outputs):
        raw_text = out.get('raw_text', out.get('final_answer', ''))
        logger.debug(f"Raw output candidate {i}:\n{raw_text}\n{'-'*40}")

    # 3. Verifier for this row
    verifier, spec = get_verifier(split, row)

    # 4. Build input_messages (system + developer + user)
    # Cache is system + developer (built once per experiment)
    # Add user message specific to this question
    if HARMONY_INPUT_MESSAGES_CACHE:
        input_messages = HARMONY_INPUT_MESSAGES_CACHE.copy()
        input_messages.append(SchemaHarmonyMessage(
            role="user",
            content=PROMPT_TEMPLATE.format(question=question),
            channel=None,
        ))
    else:
        # Fallback for non-Harmony models
        input_messages = [
            SchemaHarmonyMessage(role="user", content=question, channel=None)
        ]

    # 5. Build candidate records (as dicts after Pydantic validation)
    results: List[Dict[str, Any]] = []

    for i, output_dict in enumerate(raw_outputs):
        # Extract structured or raw output
        # Check for tool-calling conversation history first
        if 'conversation_history' in output_dict and output_dict['conversation_history']:
            # Tool-calling output with full multi-turn conversation
            conversation_history = output_dict['conversation_history']
            final_answer = output_dict['final_answer']
            raw_text = output_dict.get('raw_text', final_answer)
            has_structured = True  # Mark as structured since we have structured conversation
            has_tool_calling = True

            # Build analysis from tool call metadata
            tool_metadata = output_dict.get('tool_metadata', {})
            analysis = f"Tool-calling conversation with {tool_metadata.get('total_tool_calls', 0)} tool call(s) over {tool_metadata.get('iterations', 1)} iteration(s)"
            if tool_metadata.get('tool_calls_by_name'):
                analysis += f"\nTools used: {tool_metadata['tool_calls_by_name']}"

            # Build output_messages from conversation_history
            # This preserves the full multi-turn exchange in OpenAI format
            output_messages = []
            for msg in conversation_history:
                role = msg.get('role', 'assistant')
                content = msg.get('content', '')

                # Determine channel based on message type
                if role == 'user':
                    # User message (skip - already in input_messages)
                    continue
                elif role == 'tool':
                    # Tool result message
                    tool_call_id = msg.get('tool_call_id', '')
                    tool_name = msg.get('name', 'unknown')
                    output_messages.append(SchemaHarmonyMessage(
                        role="tool",
                        channel="tool_result",
                        content=json.dumps({
                            'tool_call_id': tool_call_id,
                            'name': tool_name,
                            'result': content,
                        }),
                    ))
                elif role == 'assistant':
                    # Assistant message - check for tool_calls
                    tool_calls = msg.get('tool_calls', [])
                    if tool_calls:
                        # Assistant message with tool calls
                        output_messages.append(SchemaHarmonyMessage(
                            role="assistant",
                            channel="tool_call",
                            content=json.dumps({
                                'thinking': content,
                                'tool_calls': tool_calls,
                            }),
                        ))
                    else:
                        # Regular assistant message (final answer)
                        output_messages.append(SchemaHarmonyMessage(
                            role="assistant",
                            channel="final",
                            content=content,
                        ))

            # Also store tools_used in a special channel for reproducibility
            tools_used = output_dict.get('tools_used', [])
            if tools_used:
                output_messages.insert(0, SchemaHarmonyMessage(
                    role="system",
                    channel="tools",
                    content=json.dumps(tools_used),
                ))

        elif 'steps' in output_dict and 'final_answer' in output_dict:
            # New structured output with steps
            steps = output_dict['steps']
            final_answer = output_dict['final_answer']
            raw_text = output_dict.get('raw_text', '')
            has_structured = True
            has_tool_calling = False

            # Store steps as JSON for consistent parsing
            # Each step has 'explanation' and 'output' fields
            analysis = json.dumps({
                'type': 'reasoning_steps',
                'steps': steps,
            })

            # Build output_messages with each step as a separate message
            output_messages = []
            for idx, step in enumerate(steps):
                output_messages.append(SchemaHarmonyMessage(
                    role="assistant",
                    channel="reasoning",
                    content=json.dumps({
                        'step': idx + 1,
                        'explanation': step.get('explanation', ''),
                        'output': step.get('output', ''),
                    }),
                ))
            # Add final answer
            output_messages.append(SchemaHarmonyMessage(
                role="assistant",
                channel="final",
                content=final_answer,
            ))

        elif 'analysis' in output_dict and 'final_answer' in output_dict:
            # Old structured output (flat analysis)
            analysis = output_dict['analysis']
            final_answer = output_dict['final_answer']
            raw_text = output_dict.get('raw_text', '')
            has_structured = True
            has_tool_calling = False

            # Build output_messages with analysis as JSON
            output_messages = []
            output_messages.append(SchemaHarmonyMessage(
                role="assistant",
                channel="reasoning",
                content=json.dumps({
                    'type': 'analysis',
                    'content': analysis,
                }),
            ))
            output_messages.append(SchemaHarmonyMessage(
                role="assistant",
                channel="final",
                content=final_answer,
            ))

        else:
            # Fallback to raw text
            raw_text = output_dict.get('raw_text', output_dict.get('final_answer', ''))
            analysis = ''
            final_answer = raw_text
            has_structured = False
            has_tool_calling = False

            # Build output_messages - try parsing Harmony channels from raw text first
            output_messages = []
            if raw_text:
                channels = extract_harmony_channels(raw_text)
                if any(channels.values()):
                    # Found Harmony channels in raw text - convert to JSON format
                    if channels['analysis']:
                        output_messages.append(SchemaHarmonyMessage(
                            role="assistant",
                            channel="reasoning",
                            content=json.dumps({
                                'type': 'analysis',
                                'content': channels['analysis'],
                            }),
                        ))
                    if channels['commentary']:
                        output_messages.append(SchemaHarmonyMessage(
                            role="assistant",
                            channel="commentary",
                            content=json.dumps({
                                'type': 'commentary',
                                'content': channels['commentary'],
                            }),
                        ))
                    if channels['final']:
                        output_messages.append(SchemaHarmonyMessage(
                            role="assistant",
                            channel="final",
                            content=channels['final'],
                        ))
                    analysis = channels['analysis']
                    final_answer = channels['final']
                else:
                    # No Harmony channels - store raw text as final
                    output_messages.append(SchemaHarmonyMessage(
                        role="assistant",
                        channel="final",
                        content=raw_text,
                    ))

        # Truncate overly long responses to prevent memory/storage issues
        final_answer = truncate_if_needed(final_answer, MAX_ANSWER_LENGTH, "final_answer")
        analysis = truncate_if_needed(analysis, MAX_ANALYSIS_LENGTH, "analysis")
        raw_text = truncate_if_needed(raw_text, MAX_ANALYSIS_LENGTH, "raw_text")

        # Extract fields for quality metrics
        if has_structured:
            # Structured output: analysis field is the full reasoning, no XML tags
            reasoning = analysis  # Full analysis is the reasoning
            plan = ""  # No separate plan field in structured output
            evaluation = ""  # No separate evaluation field
            answer = final_answer  # Use final_answer directly
        else:
            # Fallback: Try XML extraction for non-structured outputs
            reasoning = extract_xml(analysis, "reasoning") if analysis else ""
            plan = extract_xml(analysis, "plan") if analysis else ""
            evaluation = extract_xml(analysis, "evaluation") if analysis else ""

            # For answer, use final_answer or try to extract from it
            answer = extract_xml(final_answer, "answer") if final_answer else ""
            if not answer:
                answer = final_answer  # Use full final_answer if no <answer> tag

        # Calculate quality metrics
        answer_length = len(answer.strip()) if answer else 0
        reasoning_length = len(reasoning.strip()) if reasoning else 0
        plan_length = len(plan.strip()) if plan else 0
        total_length = sum(len(m.content) for m in output_messages)

        # For structured outputs, completeness is simpler (just check both fields exist)
        if has_structured:
            fields_present = sum([bool(analysis), bool(final_answer)])
            completeness_score = fields_present / 2.0
        else:
            # For non-structured, count XML tags
            fields_present = sum([
                bool(extract_xml(analysis, "normalized_query")),
                bool(plan),
                bool(reasoning),
                bool(answer),
                bool(evaluation),
            ])
            completeness_score = fields_present / 5.0

        # Detect empty/truncated responses (truly empty, not just short)
        # Short answers like "4" or "Yes" are valid - only flag if answer is completely empty
        is_empty = is_empty_response(answer, reasoning)

        quality_metrics = QualityMetrics(
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

        # 7. Run verification
        candidate_for_verify = {"text": answer}
        v_result_raw = verifier.verify(question, candidate_for_verify, spec)

        # Convert to enhanced format
        if ENHANCED_VERIFIERS_AVAILABLE and hasattr(v_result_raw, 'is_correct'):
            is_verified = v_result_raw.is_correct
            confidence = v_result_raw.confidence
            explanation = v_result_raw.explanation
            verifier_name = v_result_raw.verifier_name
        else:
            is_verified = v_result_raw.is_verified
            confidence = v_result_raw.score
            explanation = v_result_raw.info
            verifier_name = verifier.name

        # Override for empty responses - always mark as failed
        if is_empty:
            is_verified = False
            confidence = 0.0
            explanation = f"Empty/truncated response (answer_length={answer_length})"
            verifier_name = "empty_check"

        # LLM-as-judge fallback for low-confidence verifications
        use_llm_judge = getattr(args, 'llm_judge_fallback', False)
        ground_truth = spec.get("ground_truth")

        # AST syntax check for code (fast, free first pass)
        if split == 'code' and not is_verified:
            try:
                # Extract code from answer
                code = extract_code_from_text(answer, language='python')
                if code:
                    syntax_result = check_code_syntax(code, 'python')
                    if syntax_result.get("is_valid") is False:
                        # Syntax error - mark as failed
                        is_verified = False
                        confidence = syntax_result["confidence"]
                        explanation = f"AST: {syntax_result['explanation']}"
                        verifier_name = "ast_syntax"
                        logger.debug(f"AST syntax check failed: {explanation}")
                    elif syntax_result.get("is_valid") is True:
                        # Syntax valid - upgrade confidence slightly
                        confidence = max(confidence, syntax_result["confidence"])
                        logger.debug("AST syntax check passed")
            except Exception as e:
                logger.debug(f"AST check failed: {e}")

        # Track LLM judge usage
        llm_judge_used = False
        llm_judge_failed = False

        # LLM-as-judge fallback for failed verifications OR code/tool splits
        # Skip for empty responses - no point judging an empty answer
        if use_llm_judge and ground_truth and not is_empty:
            # Use LLM judge if:
            # 1. Primary verification failed (low confidence), OR
            # 2. Split is code/tool (need semantic checking)
            should_use_llm = (
                (not is_verified and confidence < 0.5) or  # Failed verification
                (split in ['code', 'tool_calling'])  # Code/tool needs semantic verification
            )

            if should_use_llm:
                logger.debug(f"Using LLM judge for {split} (primary confidence={confidence})...")
                llm_judge_used = True
                try:
                    llm_judge = get_llm_judge(provider="openai", api_key=args.api_key)
                    llm_result = await llm_judge.verify(
                        question=question,
                        candidate_answer=answer,
                        ground_truth=ground_truth,
                        split=split,
                    )

                    # Use LLM judge result if high confidence
                    if llm_result["confidence"] > 0.7:
                        is_verified = llm_result["is_correct"]
                        confidence = llm_result["confidence"]
                        explanation = llm_result["explanation"]
                        verifier_name = llm_result["verifier_name"]
                        logger.debug(f"LLM judge result: {is_verified} (confidence={confidence})")

                except Exception as e:
                    llm_judge_failed = True
                    logger.warning(f"LLM judge fallback failed: {e}")

        verification_results = VerificationResults(
            is_verified=is_verified,
            score=confidence,
            info=explanation,
            verifier_name=verifier_name,
            llm_judge_used=llm_judge_used,
            llm_judge_failed=llm_judge_failed,
        )

        # 8. Classify refusal (uses common/refusal_check.py)
        full_text = raw_text if raw_text else (analysis + "\n" + final_answer if analysis else final_answer)
        refusal_classification = build_refusal_detection(answer, full_text, refusal_classifier)

        # 9. Build Pydantic record (validates schema)
        try:
            # Parse metadata if it's a JSON string
            import json as json_lib
            metadata_raw = row.get("metadata")
            if isinstance(metadata_raw, str):
                try:
                    metadata_dict = json_lib.loads(metadata_raw)
                except (json_lib.JSONDecodeError, ValueError):
                    metadata_dict = None
            elif isinstance(metadata_raw, dict):
                metadata_dict = metadata_raw
            else:
                metadata_dict = None

            record = BestOfNRecord(
                query_id=query_id or f"unknown_{i}",
                candidate_idx=i,
                split=split,
                category=row.get("category"),
                source_dataset=args.dataset,
                reasoning_mode=row.get("reasoning"),
                source_metadata=metadata_dict,
                ground_truth_answer=ground_truth,  # Store for re-verification later
                input_messages=input_messages,
                output_messages=output_messages,
                quality=quality_metrics,
                verification=verification_results,
                refusal=refusal_classification,
                persona=None,  # TODO: Add PersonaEvaluation when enabled
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timestamp=datetime.now(),  # Use datetime.now() instead of deprecated utcnow()
                harmony_channels_detected=has_structured or len(output_messages) > 1,
            )

            # Convert to dict for parquet (flattened)
            results.append(record.to_dict())

        except Exception as e:
            logger.error(f"Failed to create BestOfNRecord: {e}")
            logger.debug(f"Data: query_id={query_id}, split={split}")
            # Skip this candidate on validation error
            continue

    return results


# -----------------------------------------------------------------------------
# Async Main
# -----------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY not set. Use --api-key or export OPENAI_API_KEY.")

    # Set up debug logging if requested (uses common.generation_utils)
    if args.debug_log:
        init_debug_logging(args.debug_log, {
            "Model": args.model,
            "Temperature": args.temperature,
            "Max tokens": args.max_tokens,
            "Num candidates": args.num_candidates,
        })

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    sem = asyncio.Semaphore(args.concurrency)

    try:
        # Main processing wrapped in try/finally for client cleanup
        await _async_main_inner(client, sem, args)
    finally:
        # Ensure client is properly closed to release connections
        await client.close()
        logger.debug("AsyncOpenAI client closed")


async def _async_main_inner(client: AsyncOpenAI, sem: asyncio.Semaphore, args: argparse.Namespace) -> None:
    """Inner async main logic, separated for proper client cleanup."""
    # Load checkpoint if resuming
    completed_query_ids: set = set()
    existing_results: List[Dict[str, Any]] = []
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            try:
                import pyarrow.parquet as pq
                checkpoint_table = pq.read_table(checkpoint_path)
                checkpoint_df = checkpoint_table.to_pandas()

                # Extract completed query_ids (unique)
                if 'query_id' in checkpoint_df.columns:
                    completed_query_ids = set(checkpoint_df['query_id'].unique())
                    logger.info(f"Found {len(completed_query_ids)} completed query_ids to skip")

                # Load existing results to merge with new ones
                existing_results = checkpoint_df.to_dict('records')
                logger.info(f"Loaded {len(existing_results)} existing records from checkpoint")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                completed_query_ids = set()
                existing_results = []
        else:
            logger.warning(f"Checkpoint file not found: {checkpoint_path}. Starting fresh.")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits provided; use --splits math,code,tool_calling for example.")

    logger.info("Dataset: %s", args.dataset)
    logger.info("Splits: %s", splits)
    logger.info("Model: %s", args.model)
    logger.info("Max queries: %d", args.max_queries)
    logger.info("Num candidates per query: %d", args.num_candidates)
    logger.info("Streaming: %s", args.streaming)
    logger.info("Concurrency: %d", args.concurrency)

    # Start with existing results from checkpoint (if any)
    all_results: List[Dict[str, Any]] = existing_results.copy()
    total_input_tokens = 0
    total_output_tokens = 0

    for split in splits:
        logger.info("Processing split: %s", split)

        # Load dataset with correct version (v2 for code/math, v1 for tool_calling)
        if 'nemotron' in args.dataset.lower():
            logger.info("  Using Nemotron-aware loading (v2 for code/math, v1 for tool_calling)")
            ds = load_nemotron_split(split, streaming=args.streaming)
        else:
            ds = load_dataset(args.dataset, split=split, streaming=args.streaming)

        # Apply max_queries limit per split
        if args.max_queries:
            if args.streaming:
                ds_iter = ds.take(args.max_queries)
            else:
                # Non-streaming: ds is a Dataset; select first max_queries rows
                n = min(args.max_queries, len(ds))
                ds_iter = ds.select(range(n))
        else:
            ds_iter = ds

        tasks: List[asyncio.Task] = []
        skipped_count = 0
        for row in ds_iter:
            # Generate query_id to check if already completed
            query_id = row.get("id") or row.get("query_id")
            if query_id is None:
                # Generate from content hash if no id
                question = extract_question_from_row(row)
                query_id = f"{split}_{hash(question) % 10**8}"

            # Skip if already completed in checkpoint
            if query_id in completed_query_ids:
                skipped_count += 1
                continue

            tasks.append(asyncio.create_task(process_item(row, split, args, client, sem)))

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already-completed queries in split '{split}'")

        if not tasks:
            logger.warning("No valid rows found in split '%s'.", split)
            continue

        logger.info("Scheduled %d query tasks for split '%s'.", len(tasks), split)

        # Use asyncio.as_completed wrapped with tqdm for progress
        # Add checkpointing every N queries to prevent data loss
        checkpoint_interval = args.checkpoint_every
        completed = 0
        failed_count = 0

        try:
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Split {split}"):
                try:
                    res = await fut
                    if res:
                        all_results.extend(res)
                        completed += 1

                        # Checkpoint periodically (if enabled)
                        if checkpoint_interval > 0 and completed % checkpoint_interval == 0 and all_results:
                            checkpoint_file = args.output.replace('.parquet', f'_checkpoint_{completed}.parquet')
                            logger.info(f"Saving checkpoint at {completed} queries to {checkpoint_file}")
                            try:
                                ds_checkpoint = Dataset.from_list(all_results)
                                ds_checkpoint.to_parquet(checkpoint_file)
                                logger.info(f"Checkpoint saved: {len(all_results)} records")
                            except Exception as e:
                                logger.warning(f"Checkpoint save failed: {e}")
                except Exception as e:
                    failed_count += 1
                    logger.warning("Task in split '%s' failed: %s", split, e)
                    logger.debug(traceback.format_exc())

                    # If >20% of tasks fail, abort early to prevent resource waste
                    if failed_count > len(tasks) * 0.2:
                        logger.error(f"Too many failures ({failed_count}/{len(tasks)}), aborting split '{split}'")
                        break
        finally:
            # Cancel any remaining tasks to prevent orphaned coroutines
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait briefly for cancellations to complete
            if any(not t.done() for t in tasks):
                await asyncio.gather(*[t for t in tasks if not t.done()], return_exceptions=True)

    if not all_results:
        logger.warning("No candidates generated; no Parquet file will be written.")
        return

    logger.info("Total candidate records: %d", len(all_results))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logger.info("Writing candidates to Parquet: %s", args.output)
    ds_out = Dataset.from_list(all_results)

    # Add experiment metadata to parquet file
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build metadata dict with experiment details
    experiment_metadata = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'num_candidates': str(args.num_candidates),
        'temperature': str(args.temperature),
        'splits': args.splits,
        'total_records': str(len(all_results)),
    }

    # Add config info if available
    if args._config:
        experiment_metadata['config_file'] = args.config or 'inline'
        if args._config_notes:
            experiment_metadata['notes'] = args._config_notes

    # Convert dataset to PyArrow table
    table = pa.Table.from_pandas(ds_out.to_pandas())

    # Add custom metadata (must be bytes)
    existing_meta = table.schema.metadata or {}
    combined_meta = {**existing_meta, **{k.encode(): v.encode() for k, v in experiment_metadata.items()}}
    table = table.cast(table.schema.with_metadata(combined_meta))

    # Write table with metadata
    pq.write_table(table, args.output)

    logger.info("Done. Experiment metadata saved in parquet.")
    if args._config_notes:
        logger.info(f"Experiment notes:\n{args._config_notes}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-of-N candidate generation with verifiers + planner-friendly tags."
    )

    # Experiment configuration
    parser.add_argument(
        "--config",
        help="Path to experiment config YAML file. CLI args override config values.",
    )

    # Data
    parser.add_argument(
        "--dataset",
        default="nvidia/Nemotron-Post-Training-Dataset-v1",
        help="Hugging Face dataset name (default: nvidia/Nemotron-Post-Training-Dataset-v1)",
    )
    parser.add_argument(
        "--splits",
        default="math,code,tool_calling",
        help="Comma-separated list of splits (default: math,code,tool_calling)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Maximum number of queries to process per split (default: 50).",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="Number of candidates per query (Best-of-N, default: 4).",
    )
    parser.add_argument(
        "--min-query-chars",
        type=int,
        default=5,
        help="Minimum question length after stripping to keep (default: 5).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode for dataset loading (default: True for memory efficiency).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming mode (loads entire dataset into memory).",
    )

    # Model / API
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model name (e.g., gpt-4o, gpt-5.1). Required if not in config.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (default: from OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL (default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=131072,
        help="Max tokens to generate per candidate (default: 131072 - model max).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls (default: 10).",
    )
    parser.add_argument(
        "--persona",
        help="Optional system prompt to inject personality/style into responses. "
             "Can be a string or path to a text file containing the persona.",
    )

    # Output
    parser.add_argument(
        "--output",
        default="candidates.parquet",
        help="Output Parquet file path (default: candidates.parquet).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Path to checkpoint parquet file to resume from. Skips already-processed query_ids.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save checkpoint every N queries (default: 25). Set to 0 to disable checkpointing.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO).",
    )
    parser.add_argument(
        "--debug-log",
        help="Path to debug log file for raw requests/responses (optional).",
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        default=True,
        help="Use structured outputs with ModelOutput schema (default: True).",
    )
    parser.add_argument(
        "--no-structured-output",
        dest="structured_output",
        action="store_false",
        help="Disable structured outputs, use raw text parsing instead.",
    )
    parser.add_argument(
        "--llm-judge-fallback",
        action="store_true",
        default=True,
        help="Use LLM-as-judge (gpt-4o-mini) as fallback when verification fails (default: True).",
    )
    parser.add_argument(
        "--no-llm-judge-fallback",
        dest="llm_judge_fallback",
        action="store_false",
        help="Disable LLM-as-judge fallback.",
    )

    # Parse CLI args first
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        logger.info(f"Loading experiment config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Merge config into args (CLI takes precedence)
        # Only set from config if NOT explicitly provided on CLI
        cli_provided = set()
        for action in parser._actions:
            if action.dest != 'help' and action.dest != 'config':
                # Check if this arg was explicitly provided on CLI
                if hasattr(args, action.dest):
                    default_value = action.default
                    actual_value = getattr(args, action.dest)
                    # If value differs from default, it was provided on CLI
                    if actual_value != default_value:
                        cli_provided.add(action.dest)

        # Apply config values for args not provided on CLI
        for key, value in config.items():
            # Convert key from YAML (e.g., 'max-queries' or 'max_queries') to argparse format
            arg_key = key.replace('-', '_')
            if arg_key not in cli_provided and hasattr(args, arg_key):
                setattr(args, arg_key, value)
                logger.debug(f"  Setting {arg_key} = {value} from config")

        # Store the loaded config and any notes in args for later use
        args._config = config
        args._config_notes = config.get('notes', '')
    else:
        args._config = {}
        args._config_notes = ''

    # Validate required args after config loading
    if not args.model:
        parser.error("--model is required (either via CLI or config file)")

    # Process persona: load from file if it's a path
    if hasattr(args, 'persona') and args.persona:
        from pathlib import Path
        persona_path = Path(args.persona)
        if persona_path.is_file():
            logger.info(f"Loading persona from file: {args.persona}")
            with open(persona_path, 'r', encoding='utf-8') as f:
                args.persona = f.read().strip()
        else:
            # Use as-is (inline persona string)
            logger.info("Using inline persona string")

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    return args


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")


if __name__ == "__main__":
    main()