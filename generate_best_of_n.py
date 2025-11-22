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
import asyncio
import argparse
import logging
import traceback
import yaml
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
try:
    from openai import AsyncOpenAI, BadRequestError
    from datasets import load_dataset, Dataset
    from tqdm.asyncio import tqdm
except ImportError:
    print("Error: Missing dependencies. Run:\n"
          "    pip install openai datasets tqdm pyarrow pandas\n",
          file=sys.stderr)
    sys.exit(1)

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

# Import enhanced verifier system
try:
    # Add parent directory to path to import from verifiers module
    import sys
    from pathlib import Path
    verifiers_path = Path(__file__).parent / "verifiers"
    if str(verifiers_path) not in sys.path:
        sys.path.insert(0, str(verifiers_path.parent))

    from verifiers import (
        get_verifier_for_split,
        load_config as load_verifier_config,
        VerificationResult as EnhancedVerificationResult,
    )
    ENHANCED_VERIFIERS_AVAILABLE = True
    logger.info("Enhanced verification system loaded (secure Docker-based execution)")
except ImportError as e:
    logger.warning(f"Enhanced verifiers not available: {e}")
    logger.warning("Falling back to basic verification (NOT SECURE FOR PRODUCTION)")
    ENHANCED_VERIFIERS_AVAILABLE = False

# Import Nemotron dataset utilities
from nemotron_utils import load_nemotron_split


# Legacy VerificationResult for backward compatibility
@dataclass
class VerificationResult:
    """Legacy verification result format (for backward compatibility)."""
    is_verified: bool
    score: float = 0.0  # 0.0 to 1.0
    info: str = ""

    @classmethod
    def from_enhanced(cls, result: 'EnhancedVerificationResult') -> 'VerificationResult':
        """Convert from enhanced VerificationResult to legacy format."""
        return cls(
            is_verified=result.is_correct,
            score=result.confidence,
            info=result.explanation
        )


def get_verifier(split: str, row: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Factory to select the correct verifier based on data split + row.

    Uses enhanced verifier system if available, otherwise falls back to
    basic verification (not recommended for production).
    """
    spec: Dict[str, Any] = {}

    # Try to find ground truth in common dataset columns
    for col in ["solution", "ground_truth", "answer", "output"]:
        if row.get(col) is not None:
            spec["ground_truth"] = row[col]
            break

    if ENHANCED_VERIFIERS_AVAILABLE:
        # Use enhanced verification system with Docker isolation
        # Load configuration (checks environment variables and config files)
        try:
            config = load_verifier_config()
        except Exception as e:
            logger.warning(f"Could not load verifier config: {e}, using defaults")
            config = None

        # Get appropriate verifier for this split
        verifier = get_verifier_for_split(split, config=config)
        return verifier, spec

    else:
        # Fallback to basic verifiers (NOT SECURE - only for compatibility)
        logger.error("=" * 80)
        logger.error("WARNING: Using legacy insecure verifiers!")
        logger.error("Enhanced verifiers not available. To enable secure verification:")
        logger.error("  1. Install dependencies: pip install sympy pint jsonschema docker pyyaml")
        logger.error("  2. Build Docker image: cd scripts/bestofn/verifiers && ./build_docker.sh")
        logger.error("  3. See: scripts/bestofn/verifiers/README.md")
        logger.error("=" * 80)
        raise RuntimeError(
            "Enhanced verifiers required but not available. "
            "See scripts/bestofn/verifiers/README.md for installation instructions."
        )


# -----------------------------------------------------------------------------
# Generation Logic (Planner + Reasoning + Answer + Evaluation + Score)
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a world-class reasoning engine.

Given the user's query, your job is to:

1. Restate the user's goal clearly (normalized).
2. Produce a high-level plan of how to solve or answer the query.
   - The plan should be abstract (no detailed calculations), just steps.
3. Apply that plan step by step, using explicit reasoning and checks.
4. Produce a final answer (concise and user-facing).
5. Briefly evaluate the quality of your own planning+reasoning and give a score 0–5.

USER QUERY:
{question}

You MUST format your entire response using the following XML-like tags exactly once each:

<normalized_query>...</normalized_query>
<plan>...</plan>
<reasoning>...</reasoning>
<answer>...</answer>
<evaluation>...</evaluation>
<score>...</score>
"""

def extract_xml(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags, with fallback for unclosed tags.

    Handles:
    - Normal: <tag>content</tag>
    - Unclosed at end: <tag>content (end of string)
    - Case insensitive matching
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


async def generate_candidates(
    client: AsyncOpenAI,
    model: str,
    question: str,
    n: int,
    sem: asyncio.Semaphore,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    persona: Optional[str] = None,
) -> List[str]:
    """
    Generates N candidates for a question.

    Tries efficient batch generation with n>1 first; if the model does not
    support it (BadRequestError), falls back to sequential calls.

    Args:
        persona: Optional system prompt to inject personality/style
    """
    prompt = PROMPT_TEMPLATE.format(question=question)

    # Build messages list with optional system prompt for persona
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": prompt})

    async with sem:
        try:
            # Try efficient batch generation first
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return [c.message.content for c in resp.choices]
        except BadRequestError as e:
            # Fallback for models that don't support n>1 (e.g., some reasoning models)
            if "n" in str(e):
                logger.warning(f"Model '{model}' does not support n={n}. Falling back to sequential sampling.")
                results: List[str] = []
                for _ in range(n):
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    results.append(resp.choices[0].message.content)
                return results
            raise e
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []


# -----------------------------------------------------------------------------
# Main Per-Item Processing
# -----------------------------------------------------------------------------

def extract_question_from_row(row: Dict[str, Any], min_len: int = 5) -> Optional[str]:
    """
    Extract the first non-empty user message from row["messages"].

    Returns None if no suitable message is found.
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return None

    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(role, str) and role.lower() == "user" and isinstance(content, str):
            text = content.strip()
            if text and text != "-" and len(text) >= min_len:
                return text
    return None


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
    # 1. Extract Question
    question = extract_question_from_row(row, min_len=args.min_query_chars)
    if not question:
        return []

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
    )
    if not raw_outputs:
        return []

    # 3. Verifier for this row
    verifier, spec = get_verifier(split, row)

    # 4. Build candidate records
    results: List[Dict[str, Any]] = []
    query_id = row.get("uuid") or row.get("id") or row.get("uid")
    query_id = str(query_id) if query_id is not None else None

    for i, raw_out in enumerate(raw_outputs):
        normalized_query = extract_xml(raw_out, "normalized_query")
        plan = extract_xml(raw_out, "plan")
        reasoning = extract_xml(raw_out, "reasoning")
        answer = extract_xml(raw_out, "answer")
        evaluation = extract_xml(raw_out, "evaluation")
        score_text = extract_xml(raw_out, "score")
        teacher_self_score = parse_score_tag(score_text)
        plan_steps = extract_plan_steps(plan)

        # Fallback: if no <answer> tag found, use entire raw_out as answer
        if not answer:
            answer = raw_out

        candidate = {
            "normalized_query": normalized_query,
            "plan": plan,
            "reasoning": reasoning,
            "answer": answer,
            "evaluation": evaluation,
            "teacher_self_score": teacher_self_score,
        }

        # 5. Run verification
        # Update candidate format for enhanced verifiers (they expect 'text' field)
        candidate_for_verify = {
            "text": answer,  # Enhanced verifiers look for 'text' field
            **candidate,     # Include all other fields
        }
        v_result_raw = verifier.verify(question, candidate_for_verify, spec)

        # Convert to legacy format if using enhanced verifiers
        if ENHANCED_VERIFIERS_AVAILABLE and hasattr(v_result_raw, 'is_correct'):
            v_result = VerificationResult.from_enhanced(v_result_raw)
        else:
            v_result = v_result_raw

        results.append(
            {
                "query_id": query_id,
                "split": split,
                "category": row.get("category"),
                "reasoning_mode": row.get("reasoning"),
                "source_metadata": row.get("metadata"),
                "question": question,
                "candidate_idx": i,
                "normalized_query": normalized_query,
                "plan": plan,
                "plan_steps": plan_steps,
                "reasoning": reasoning,
                "answer": answer,
                "evaluation": evaluation,
                "teacher_self_score": teacher_self_score,
                "raw_model_output": raw_out,
                "is_verified": v_result.is_verified,
                "verification_score": v_result.score,
                "verification_info": v_result.info,
                "verifier_name": verifier.name,
                "model": args.model,
                "temperature": args.temperature,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    return results


# -----------------------------------------------------------------------------
# Async Main
# -----------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY not set. Use --api-key or export OPENAI_API_KEY.")

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    sem = asyncio.Semaphore(args.concurrency)

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

    all_results: List[Dict[str, Any]] = []

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
        for row in ds_iter:
            tasks.append(asyncio.create_task(process_item(row, split, args, client, sem)))

        if not tasks:
            logger.warning("No valid rows found in split '%s'.", split)
            continue

        logger.info("Scheduled %d query tasks for split '%s'.", len(tasks), split)

        # Use asyncio.as_completed wrapped with tqdm for progress
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Split {split}"):
            try:
                res = await fut
                if res:
                    all_results.extend(res)
            except Exception as e:
                logger.warning("Task in split '%s' failed: %s", split, e)
                logger.debug(traceback.format_exc())

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
        'generated_at': datetime.utcnow().isoformat(),
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
    table = table.replace_schema(table.schema.with_metadata(combined_meta))

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
        help="Use streaming mode for datasets (recommended for very large splits).",
    )

    # Model / API
    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model name (e.g., gpt-4o, gpt-5.1).",
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
        default=2048,
        help="Max tokens to generate per candidate (default: 2048).",
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

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO).",
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