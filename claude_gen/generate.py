#!/usr/bin/env python3
"""
Best-of-N Generator for Claude API (Extended Thinking)
-------------------------------------------------------

Similar to generate_best_of_n.py but optimized for Claude's extended thinking mode.

Key differences from OpenAI version:
- Uses anthropic SDK instead of openai
- Leverages Claude's extended thinking blocks
- Maps thinking → steps for structured output
- System prompt for persona instead of developer message

Usage:
    python generate_best_of_n_claude.py \
        --config experiments/marvin_claude_100x8.yaml
"""

import os
import sys
import asyncio
import argparse
import logging
import traceback
import yaml
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
try:
    from anthropic import AsyncAnthropic
    from datasets import load_dataset, Dataset
    from tqdm.asyncio import tqdm
except ImportError:
    print("Error: Missing dependencies. Run:\n"
          "    pip install anthropic datasets tqdm pyarrow pandas\n",
          file=sys.stderr)
    sys.exit(1)

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
    ReasoningStep,
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
    get_verifier,
)
from common.llm_judge import get_llm_judge

# Import Claude-specific tool executor
from claude_gen.tool_executor import generate_candidates_with_tool_calling

# Import verifiers
from verifiers import (
    get_verifier_for_split,
    load_config as load_verifier_config,
    VerificationResult as EnhancedVerificationResult,
    RefusalClassifier
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BestOfN-Claude")

# Global debug log
DEBUG_LOG_FILE: Optional[str] = None
DEBUG_LOG_LOCK = None

# Shared verifier components
ENHANCED_VERIFIERS_AVAILABLE = True
refusal_classifier = RefusalClassifier()
logger.info("Enhanced verification system loaded (shared with OpenAI version)")

# Input messages cache
INPUT_MESSAGES_CACHE = None


# -----------------------------------------------------------------------------
# Prompt Template
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """Solve this problem:

{question}

Show your step-by-step reasoning. Maintain your character/persona throughout.

IMPORTANT: For math problems, your final answer MUST use \\boxed{{result}} format.
"""


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def build_input_messages_cache(persona: Optional[str] = None) -> List[SchemaHarmonyMessage]:
    """Build cached input messages for training data."""
    messages = []

    # System message with persona
    system_content = persona if persona else "You are a helpful assistant."
    messages.append(SchemaHarmonyMessage(
        role="system",
        content=system_content,
        channel=None,
    ))

    return messages


async def log_request_response(
    query_id: str,
    question: str,
    request_body: Dict[str, Any],
    responses: List[str],
    error: Optional[str] = None,
) -> None:
    """Log raw request and response to debug file."""
    if not DEBUG_LOG_FILE:
        return

    global DEBUG_LOG_LOCK
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


def parse_thinking_blocks(content_blocks: List[Any]) -> Dict[str, Any]:
    """
    Parse Claude's response blocks into steps and answer.

    Claude returns:
    - thinking blocks (extended thinking content)
    - text blocks (final answer)

    We map these to our ModelOutput structure.
    """
    thinking_parts = []
    text_parts = []

    for block in content_blocks:
        if hasattr(block, 'type'):
            if block.type == 'thinking':
                thinking_parts.append(block.thinking)
            elif block.type == 'text':
                text_parts.append(block.text)

    # For now, combine thinking into steps heuristically
    # Split on double newlines or step markers
    thinking_text = '\n\n'.join(thinking_parts)

    # Try to break into steps
    import re
    step_patterns = [
        r'Step \d+:(.+?)(?=Step \d+:|$)',
        r'\d+\.\s+(.+?)(?=\d+\.|$)',
        r'(?:^|\n\n)([^\n]+(?:\n(?!\n)[^\n]+)*)',
    ]

    steps = []
    for pattern in step_patterns:
        matches = re.findall(pattern, thinking_text, re.DOTALL | re.MULTILINE)
        if matches and len(matches) >= 2:
            for match in matches:
                match = match.strip()
                if len(match) > 20:  # Substantive content
                    # Split explanation from output (heuristic)
                    lines = match.split('\n')
                    explanation = lines[0] if lines else match[:100]
                    output = match
                    steps.append({'explanation': explanation, 'output': output})
            break

    # Fallback: entire thinking as one step
    if not steps and thinking_text:
        steps = [{
            'explanation': 'Complete reasoning',
            'output': thinking_text
        }]

    final_answer = '\n\n'.join(text_parts)

    return {
        'steps': steps,
        'final_answer': final_answer,
        'raw_text': thinking_text + '\n\n' + final_answer,
    }


async def generate_candidates_claude(
    client: AsyncAnthropic,
    model: str,
    question: str,
    n: int,
    sem: asyncio.Semaphore,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    persona: Optional[str] = "",
    query_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate N candidates using Claude API with extended thinking.

    Args:
        client: AsyncAnthropic client
        model: Claude model name (e.g., claude-sonnet-4-5-20250929)
        question: Question to answer
        n: Number of candidates to generate
        sem: Semaphore for concurrency control
        temperature: Sampling temperature
        max_tokens: Max output tokens
        persona: System prompt persona
        query_id: Query ID for logging

    Returns:
        List of dicts with 'steps' and 'final_answer'
    """
    # Build input messages cache (for training data) on first call
    global INPUT_MESSAGES_CACHE
    if INPUT_MESSAGES_CACHE is None and persona:
        INPUT_MESSAGES_CACHE = build_input_messages_cache(persona)
        logger.info(f"Built input messages cache with {len(INPUT_MESSAGES_CACHE)} messages")

    # Format question
    user_content = PROMPT_TEMPLATE.format(question=question)

    # Build request body for debug logging
    request_body = {
        "model": model,
        "system": persona if persona else "You are a helpful assistant.",
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "thinking": {"type": "enabled", "budget_tokens": max_tokens // 2},
    }

    async with sem:
        try:
            results: List[Dict[str, Any]] = []

            for i in range(n):
                logger.info(f"Generating candidate {i+1}/{n} for query {query_id}")

                response = await client.messages.create(
                    model=model,
                    system=persona if persona else "You are a helpful assistant.",
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": max_tokens // 2  # Allocate half to thinking
                    },
                )

                # Parse thinking blocks and text
                parsed = parse_thinking_blocks(response.content)
                results.append(parsed)

            # Log request/response if debug logging enabled
            if DEBUG_LOG_FILE and query_id:
                raw_responses = [r.get('raw_text', '') for r in results]
                await log_request_response(query_id, question, request_body, raw_responses)

            return results

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            if DEBUG_LOG_FILE and query_id:
                await log_request_response(query_id, question, request_body, [], error=error_msg)

            return []


# Note: get_verifier and extract_question_from_row imported from common.generation_utils

async def process_item(
    row: Dict[str, Any],
    split: str,
    args: argparse.Namespace,
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """Process a single dataset row (similar to OpenAI version)."""
    # Extract question and query ID
    question = extract_question_from_row(row, min_len=args.min_query_chars)
    if not question:
        return []

    query_id = row.get("uuid") or row.get("id") or row.get("uid")
    query_id = str(query_id) if query_id is not None else None

    # Check for tool_calling split - use tool executor if tools available
    raw_outputs = None
    has_tool_calling = False

    if split == 'tool_calling':
        # Try to extract tools from metadata
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        tools = metadata.get("tools", [])
        if tools:
            has_tool_calling = True
            logger.debug(f"Using tool calling for query {query_id} with {len(tools)} tools")
            try:
                raw_outputs = await generate_candidates_with_tool_calling(
                    client=client,
                    model=args.model,
                    question=question,
                    metadata=metadata,
                    n=args.num_candidates,
                    sem=sem,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    persona=getattr(args, 'persona', None),
                )
            except Exception as e:
                logger.warning(f"Tool calling failed for {query_id}: {e}, falling back to standard generation")
                raw_outputs = None

    # Standard generation (or fallback)
    if raw_outputs is None:
        raw_outputs = await generate_candidates_claude(
            client=client,
            model=args.model,
            question=question,
            n=args.num_candidates,
            sem=sem,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            persona=getattr(args, 'persona', None),
            query_id=query_id,
        )

    if not raw_outputs:
        return []

    # Debug logging of raw outputs
    for i, out in enumerate(raw_outputs):
        raw_text = out.get('raw_text', out.get('final_answer', ''))
        logger.debug(f"Raw output candidate {i}:\n{raw_text}\n{'-'*40}")

    # Get verifier
    verifier, spec = get_verifier(split, row)

    # Build input_messages
    if INPUT_MESSAGES_CACHE:
        input_messages = INPUT_MESSAGES_CACHE.copy()
        input_messages.append(SchemaHarmonyMessage(
            role="user",
            content=PROMPT_TEMPLATE.format(question=question),
            channel=None,
        ))
    else:
        input_messages = [
            SchemaHarmonyMessage(role="user", content=question, channel=None)
        ]

    # Build candidate records
    results: List[Dict[str, Any]] = []

    for i, output_dict in enumerate(raw_outputs):
        steps = output_dict.get('steps', [])
        final_answer = output_dict.get('final_answer', '')
        raw_text = output_dict.get('raw_text', '')

        # Convert steps to analysis text
        analysis = '\n\n'.join([
            f"Step {idx+1}: {step['explanation']}\n{step['output']}"
            for idx, step in enumerate(steps)
        ]) if steps else raw_text

        # Build output_messages
        output_messages = []
        if analysis:
            output_messages.append(SchemaHarmonyMessage(
                role="assistant",
                channel="analysis",  # Match OpenAI channel naming for parity
                content=analysis,
            ))
        if final_answer:
            output_messages.append(SchemaHarmonyMessage(
                role="assistant",
                channel="final",
                content=final_answer,
            ))

        # Quality metrics
        answer_length = len(final_answer.strip())
        reasoning_length = len(analysis.strip())

        # Detect empty/truncated responses (truly empty, not just short)
        # Short answers like "4" or "Yes" are valid - only flag if answer is completely empty
        is_empty = answer_length == 0
        if is_empty:
            logger.warning(f"Empty response detected: answer_length={answer_length}, reasoning_length={reasoning_length}")

        quality_metrics = QualityMetrics(
            answer_length=answer_length,
            reasoning_length=reasoning_length,
            plan_length=0,
            total_response_length=answer_length + reasoning_length,
            has_reasoning=reasoning_length > 0,
            has_plan=False,
            is_short_answer=answer_length < 50,
            is_substantive=reasoning_length > 100 or answer_length > 50,
            is_empty=is_empty,
            completeness_score=1.0 if (steps and final_answer) else 0.5,
        )

        # Run verification
        candidate_for_verify = {"text": final_answer}
        v_result_raw = verifier.verify(question, candidate_for_verify, spec)

        is_verified = v_result_raw.is_correct if hasattr(v_result_raw, 'is_correct') else v_result_raw.is_verified
        confidence = v_result_raw.confidence if hasattr(v_result_raw, 'confidence') else v_result_raw.score
        explanation = v_result_raw.explanation if hasattr(v_result_raw, 'explanation') else v_result_raw.info
        verifier_name = v_result_raw.verifier_name if hasattr(v_result_raw, 'verifier_name') else verifier.name
        llm_judge_used = False
        llm_judge_failed = False

        # Override for empty responses - always mark as failed
        if is_empty:
            is_verified = False
            confidence = 0.0
            explanation = f"Empty/truncated response (answer_length={answer_length})"
            verifier_name = "empty_check"

        # AST pre-validation for code splits (fast, free check before LLM judge)
        if split == 'code' and not is_empty and final_answer.strip():
            code = extract_code_from_text(final_answer, language='python')
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

        # LLM Judge fallback (if enabled and primary verification uncertain)
        # Skip for empty responses - no point judging an empty answer
        use_llm_judge = getattr(args, 'llm_judge_fallback', False)
        ground_truth = extract_ground_truth_from_message(row)

        if use_llm_judge and ground_truth and not is_empty:
            # Use LLM judge when primary verification has low confidence
            should_use_llm = (
                (not is_verified and confidence < 0.5) or  # Match OpenAI threshold
                (split in ['code', 'tool_calling'])  # Code/tool needs semantic verification
            )
            if should_use_llm:
                logger.debug(f"Using LLM judge for {split} (primary confidence={confidence})...")
                llm_judge_used = True
                try:
                    llm_judge = get_llm_judge(provider="claude", api_key=args.api_key)
                    llm_result = await llm_judge.verify(
                        question=question,
                        candidate_answer=final_answer,
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
                    logger.warning(f"LLM judge failed: {e}")
                    llm_judge_failed = True

        verification_results = VerificationResults(
            is_verified=is_verified,
            score=confidence,
            info=explanation,
            verifier_name=verifier_name,
            llm_judge_used=llm_judge_used,
            llm_judge_failed=llm_judge_failed,
        )

        # Classify refusal (check final answer AND full output for hidden refusals)
        refusal_classification = RefusalDetection()
        if refusal_classifier:
            refusal_result = refusal_classifier.classify(final_answer)
            # Also check full output (analysis + final_answer) for hidden refusals
            if not refusal_result["is_refusal"]:
                full_text = raw_text if raw_text else (analysis + "\n" + final_answer)
                raw_refusal = refusal_classifier.classify(full_text)
                if raw_refusal["is_refusal"] and raw_refusal["confidence"] > refusal_result["confidence"]:
                    refusal_result = raw_refusal

            refusal_classification = RefusalDetection(
                is_refusal=refusal_result["is_refusal"],
                confidence=refusal_result["confidence"],
                refusal_type=refusal_result.get("refusal_type"),
                matched_patterns=refusal_result.get("matched_patterns", []),
            )

        # Build Pydantic record
        try:
            metadata_raw = row.get("metadata")
            if isinstance(metadata_raw, str):
                try:
                    metadata_dict = json.loads(metadata_raw)
                except:
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
                persona=None,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timestamp=datetime.now(),
                harmony_channels_detected=len(output_messages) > 1,
            )

            results.append(record.to_dict())

        except Exception as e:
            logger.error(f"Failed to create BestOfNRecord: {e}")
            logger.debug(f"Data: query_id={query_id}, split={split}")
            continue

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Use --api-key or export ANTHROPIC_API_KEY.")

    # Set up debug logging
    global DEBUG_LOG_FILE, DEBUG_LOG_LOCK
    if args.debug_log:
        DEBUG_LOG_FILE = args.debug_log
        DEBUG_LOG_LOCK = asyncio.Lock()
        with open(DEBUG_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"DEBUG LOG (Claude) - Generated at {datetime.now().isoformat()}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Max tokens: {args.max_tokens}\n")
            f.write(f"Num candidates: {args.num_candidates}\n")
            f.write("=" * 80 + "\n\n")
        logger.info(f"Debug logging enabled: {DEBUG_LOG_FILE}")

    # Create client with optional custom base URL
    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
        logger.info(f"Using custom base URL: {args.base_url}")
    client = AsyncAnthropic(**client_kwargs)
    sem = asyncio.Semaphore(args.concurrency)

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
        raise SystemExit("No splits provided; use --splits math,code,tool_calling")

    logger.info("Dataset: %s", args.dataset)
    logger.info("Splits: %s", splits)
    logger.info("Model: %s", args.model)
    logger.info("Max queries: %d", args.max_queries)
    logger.info("Num candidates per query: %d", args.num_candidates)
    logger.info("Streaming: %s", args.streaming)
    logger.info("Concurrency: %d", args.concurrency)

    # Start with existing results from checkpoint (if any)
    all_results: List[Dict[str, Any]] = existing_results.copy()

    for split in splits:
        logger.info("Processing split: %s", split)

        # Load dataset
        if 'nemotron' in args.dataset.lower():
            logger.info("  Using Nemotron-aware loading")
            ds = load_nemotron_split(split, streaming=args.streaming)
        else:
            ds = load_dataset(args.dataset, split=split, streaming=args.streaming)

        # Apply max_queries limit
        if args.max_queries:
            if args.streaming:
                ds_iter = ds.take(args.max_queries)
            else:
                n = min(args.max_queries, len(ds))
                ds_iter = ds.select(range(n))
        else:
            ds_iter = ds

        tasks: List[asyncio.Task] = []
        skipped_count = 0
        for row in ds_iter:
            # Generate query_id to check if already completed
            query_id = row.get("uuid") or row.get("id") or row.get("query_id")
            if query_id is None:
                # Generate from content hash if no id
                question = extract_question_from_row(row, min_len=1)
                query_id = f"{split}_{hash(question) % 10**8}" if question else None

            # Skip if already completed in checkpoint
            if query_id and query_id in completed_query_ids:
                skipped_count += 1
                continue

            tasks.append(asyncio.create_task(process_item(row, split, args, client, sem)))

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already-completed queries in split '{split}'")

        if not tasks:
            logger.warning("No valid rows found in split '%s'.", split)
            continue

        logger.info("Scheduled %d query tasks for split '%s'.", len(tasks), split)

        # Process with checkpointing
        checkpoint_interval = args.checkpoint_every
        completed = 0

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Split {split}"):
            try:
                res = await fut
                if res:
                    all_results.extend(res)
                    completed += 1

                    # Checkpoint periodically (if enabled)
                    if checkpoint_interval > 0 and completed % checkpoint_interval == 0 and all_results:
                        checkpoint_file = args.output.replace('.parquet', f'_checkpoint_{completed}.parquet')
                        logger.info(f"Saving checkpoint at {completed} queries")
                        try:
                            ds_checkpoint = Dataset.from_list(all_results)
                            ds_checkpoint.to_parquet(checkpoint_file)
                            logger.info(f"Checkpoint saved: {len(all_results)} records")
                        except Exception as e:
                            logger.warning(f"Checkpoint save failed: {e}")
            except Exception as e:
                logger.warning("Task in split '%s' failed: %s", split, e)
                logger.debug(traceback.format_exc())

    if not all_results:
        logger.warning("No candidates generated; no Parquet file will be written.")
        return

    logger.info("Total candidate records: %d", len(all_results))

    # Write final parquet
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logger.info("Writing candidates to Parquet: %s", args.output)
    ds_out = Dataset.from_list(all_results)

    import pyarrow as pa
    import pyarrow.parquet as pq

    experiment_metadata = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'api': 'claude',
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
    elif args._config_notes:
        experiment_metadata['notes'] = args._config_notes

    table = pa.Table.from_pandas(ds_out.to_pandas())
    existing_meta = table.schema.metadata or {}
    combined_meta = {**existing_meta, **{k.encode(): v.encode() for k, v in experiment_metadata.items()}}
    table = table.cast(table.schema.with_metadata(combined_meta))

    pq.write_table(table, args.output)

    logger.info("Done. Experiment metadata saved in parquet.")
    if args._config_notes:
        logger.info(f"Experiment notes:\n{args._config_notes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-of-N candidate generation for Claude API with extended thinking."
    )

    parser.add_argument("--config", help="Path to experiment config YAML file")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v1")
    parser.add_argument("--splits", default="math,code,tool_calling")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--min-query-chars", type=int, default=5)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Claude model (default: claude-sonnet-4-5-20250929, or claude-opus-4-5-20251101)")
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=131072, help="Max tokens per response (default: 131072 - model max)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--persona", help="Persona file or string")
    parser.add_argument("--output", default="candidates_claude.parquet")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--debug-log", help="Debug log file for requests/responses")
    parser.add_argument(
        "--llm-judge-fallback",
        action="store_true",
        dest="llm_judge_fallback",
        help="Use Claude Sonnet 4.5 as LLM judge fallback for uncertain verifications",
    )
    parser.add_argument(
        "--no-llm-judge-fallback",
        action="store_false",
        dest="llm_judge_fallback",
        help="Disable LLM judge fallback (default behavior)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Path to checkpoint parquet file to resume from. Skips already-processed query_ids.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("ANTHROPIC_BASE_URL"),
        help="Custom Anthropic API base URL (default: uses ANTHROPIC_BASE_URL env var or Anthropic default).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save checkpoint every N queries (default: 25). Set to 0 to disable checkpointing.",
    )

    args = parser.parse_args()

    # Load config file
    if args.config:
        logger.info(f"Loading experiment config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Merge config (CLI takes precedence)
        cli_provided = set()
        for action in parser._actions:
            if action.dest not in ['help', 'config'] and hasattr(args, action.dest):
                if getattr(args, action.dest) != action.default:
                    cli_provided.add(action.dest)

        for key, value in config.items():
            arg_key = key.replace('-', '_')
            if arg_key not in cli_provided and hasattr(args, arg_key):
                setattr(args, arg_key, value)

        args._config = config
        args._config_notes = config.get('notes', '')
    else:
        args._config = {}
        args._config_notes = ''

    # Model has a default now, no validation needed

    # Load persona from file if needed
    if hasattr(args, 'persona') and args.persona:
        persona_path = Path(args.persona)
        if persona_path.is_file():
            logger.info(f"Loading persona from file: {args.persona}")
            with open(persona_path, 'r', encoding='utf-8') as f:
                args.persona = f.read().strip()
        else:
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
