#!/usr/bin/env python3
"""
Regenerate specific splits or failed rows from an existing parquet file (Claude version).

Usage:
    # Regenerate all tool_calling rows
    python -m claude.regen experiments/results/marvin_claude.parquet --split tool_calling

    # Regenerate only failed verifications
    python -m claude.regen experiments/results/marvin_claude.parquet --split tool_calling --failed-only

    # Regenerate to a new file (don't modify original)
    python -m claude.regen experiments/results/marvin_claude.parquet --split tool_calling -o results/fixed.parquet

    # Dry run - show what would be regenerated
    python -m claude.regen experiments/results/marvin_claude.parquet --split tool_calling --dry-run
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import json
import pandas as pd

# Add parent directory to path for common and verifiers modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from Claude generate module
from claude_gen.generate import generate_candidates_claude

# Import shared regen utilities from common module
from common.regen_pipeline import (
    load_parquet,
    filter_rows,
    extract_unique_queries,
    merge_results,
    save_checkpoint,
    clean_nan_values,
)

logger = logging.getLogger(__name__)


async def regenerate_query(
    query: Dict[str, Any],
    model: str,
    persona: Optional[str],
    num_candidates: int,
    temperature: float,
    max_tokens: int,
    client,
    sem: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """Regenerate candidates for a single query."""
    query_id = query.get('query_id', 'unknown')
    question = query['question']
    split = query['split']
    metadata = query.get('metadata', {})

    logger.info(f"Regenerating query {query_id}: {question[:50]}... (split: {split})")

    try:
        candidates = await generate_candidates_claude(
            client=client,
            model=model,
            question=question,
            n=num_candidates,
            sem=sem,
            persona=persona,
            temperature=temperature,
            max_tokens=max_tokens,
            split=split,
            row_metadata=metadata,
        )
        return candidates
    except Exception as e:
        logger.error(f"Failed to regenerate: {e}")
        return []


async def regenerate_all(
    queries: List[Dict[str, Any]],
    model: str,
    persona: Optional[str],
    num_candidates: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    base_url: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 10,
) -> List[Dict[str, Any]]:
    """Regenerate all queries with concurrency control and checkpointing."""
    from anthropic import AsyncAnthropic

    # Create client with optional custom base URL
    client_kwargs = {"api_key": os.getenv("ANTHROPIC_API_KEY")}
    if base_url:
        client_kwargs["base_url"] = base_url
        logger.info(f"Using custom base URL: {base_url}")
    client = AsyncAnthropic(**client_kwargs)

    try:
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        completed = 0  # Use external counter for accurate progress

        async def process_with_semaphore(query):
            # Note: Don't acquire semaphore here - generate_candidates() handles it internally
            # Acquiring here would cause deadlock since asyncio.Semaphore is not reentrant
            candidates = await regenerate_query(
                query=query,
                model=model,
                persona=persona,
                num_candidates=num_candidates,
                temperature=temperature,
                max_tokens=max_tokens,
                client=client,
                sem=semaphore,
            )
            return {
                'query': query,
                'candidates': candidates,
            }

        tasks = [process_with_semaphore(q) for q in queries]

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1
                logger.info(f"Progress: {completed}/{len(queries)}")

                # Checkpoint periodically for crash recovery
                if checkpoint_path and completed % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_path, completed, len(queries), results)

            except Exception as e:
                completed += 1
                logger.warning(f"Task {completed}/{len(queries)} failed: {e}")
                # Continue processing other tasks

        return results

    finally:
        # Ensure client is properly closed
        await client.close()
        logger.debug("AsyncAnthropic client closed")


# Note: _clean_nan_values and merge_results imported from common.regen_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate specific splits or failed rows from parquet file (Claude)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input parquet file to regenerate from",
    )

    parser.add_argument(
        "--split", "-s",
        type=str,
        help="Only regenerate rows from this split (e.g., tool_calling)",
    )

    parser.add_argument(
        "--failed-only", "-f",
        action="store_true",
        help="Only regenerate rows that failed verification",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (default: overwrite input with .bak backup)",
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be regenerated without doing it",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model to use for regeneration (default: claude-sonnet-4-5-20250929, or claude-opus-4-5-20251101)",
    )

    parser.add_argument(
        "--persona", "-p",
        type=Path,
        help="Persona file to use (if not specified, extracts from parquet metadata)",
    )

    parser.add_argument(
        "--num-candidates", "-N",
        type=int,
        default=8,
        help="Number of candidates per query (default: 8)",
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Temperature for generation (default: 1.0)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=131072,
        help="Max tokens per response (default: 131072 - model max)",
    )

    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Number of concurrent requests (default: 3)",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("ANTHROPIC_BASE_URL"),
        help="Custom Anthropic API base URL (default: uses ANTHROPIC_BASE_URL env var or Anthropic default).",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


async def async_main(args):
    """Main async entry point."""
    # Load parquet
    df = load_parquet(args.input)

    # Filter rows
    filtered_df = filter_rows(
        df,
        split=args.split,
        failed_only=args.failed_only,
    )

    if len(filtered_df) == 0:
        logger.info("No rows to regenerate")
        return

    # Extract unique queries
    queries = extract_unique_queries(filtered_df)

    # Dry run - just show stats
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN - Would regenerate (Claude):")
        print(f"  Input file: {args.input}")
        print(f"  Split filter: {args.split or 'all'}")
        print(f"  Failed only: {args.failed_only}")
        print(f"  Rows to regenerate: {len(filtered_df)}")
        print(f"  Unique queries: {len(queries)}")
        print(f"  Model: {args.model}")
        print(f"  Candidates per query: {args.num_candidates}")
        print(f"{'='*60}")

        print("\nSample queries to regenerate:")
        for q in queries[:5]:
            print(f"  - [{q['split']}] {q['question'][:60]}...")

        if len(queries) > 5:
            print(f"  ... and {len(queries) - 5} more")

        return

    # Load persona
    persona = None
    if args.persona:
        persona = args.persona.read_text()
        logger.info(f"Loaded persona from {args.persona}")
    else:
        # Try to extract from parquet metadata
        if 'persona' in df.columns:
            persona = df['persona'].iloc[0]
            logger.info("Extracted persona from parquet")

    # Regenerate with checkpointing
    output_path = args.output or args.input
    logger.info(f"Starting regeneration of {len(queries)} queries with Claude...")
    results = await regenerate_all(
        queries=queries,
        model=args.model,
        persona=persona,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        base_url=args.base_url,
        checkpoint_path=output_path,  # Enable checkpointing
        checkpoint_interval=10,  # Checkpoint every 10 queries
    )

    # Merge results
    merged_df = merge_results(df, results, args.split)

    # Save (output_path already set above for checkpointing)
    if output_path == args.input:
        # Create backup
        backup_path = args.input.with_suffix('.parquet.bak')
        logger.info(f"Creating backup: {backup_path}")
        df.to_parquet(backup_path)

    logger.info(f"Saving to {output_path}")
    merged_df.to_parquet(output_path)

    print(f"\n{'='*60}")
    print(f"Regeneration complete! (Claude)")
    print(f"  Queries regenerated: {len(queries)}")
    print(f"  Output: {output_path}")
    if output_path == args.input:
        print(f"  Backup: {args.input.with_suffix('.parquet.bak')}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
