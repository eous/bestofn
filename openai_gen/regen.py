#!/usr/bin/env python3
"""
Regenerate specific splits or failed rows from an existing parquet file.

Usage:
    # Regenerate all tool_calling rows
    python regen.py experiments/results/j5_100x8.parquet --split tool_calling

    # Regenerate only failed verifications
    python regen.py experiments/results/j5_100x8.parquet --split tool_calling --failed-only

    # Regenerate to a new file (don't modify original)
    python regen.py experiments/results/j5_100x8.parquet --split tool_calling -o results/j5_tool_calling_fixed.parquet

    # Dry run - show what would be regenerated
    python regen.py experiments/results/j5_100x8.parquet --split tool_calling --dry-run
"""

import argparse
import asyncio
import logging
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

from openai_gen.generate import generate_candidates

logger = logging.getLogger(__name__)


def extract_question_from_input_messages(input_messages) -> str:
    """Extract question from input_messages JSON field."""
    if not input_messages:
        return ""

    # Parse JSON if string
    if isinstance(input_messages, str):
        try:
            input_messages = json.loads(input_messages)
        except (json.JSONDecodeError, ValueError):
            return ""

    # Find user message
    if isinstance(input_messages, list):
        for msg in input_messages:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')

    return ""


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet file."""
    logger.info(f"Loading {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def filter_rows(
    df: pd.DataFrame,
    split: Optional[str] = None,
    failed_only: bool = False,
) -> pd.DataFrame:
    """Filter rows to regenerate."""
    original_count = len(df)

    # Filter by split
    if split:
        if 'split' not in df.columns:
            logger.warning("No 'split' column found - cannot filter by split")
        else:
            df = df[df['split'] == split]
            logger.info(f"Filtered to split '{split}': {len(df)} rows")

    # Filter to failed only
    if failed_only:
        if 'verification_is_verified' in df.columns:
            # New schema: flattened verification fields
            df = df[df['verification_is_verified'] == False]
            logger.info(f"Filtered to failed only: {len(df)} rows")
        elif 'is_correct' in df.columns:
            df = df[df['is_correct'] == False]
            logger.info(f"Filtered to failed only: {len(df)} rows")
        elif 'verification_result' in df.columns:
            df = df[df['verification_result'].apply(
                lambda x: not x.get('is_correct', True) if isinstance(x, dict) else True
            )]
            logger.info(f"Filtered to failed only: {len(df)} rows")
        else:
            logger.warning("No verification column found - cannot filter by failed")

    logger.info(f"Will regenerate {len(df)}/{original_count} rows")
    return df


def extract_unique_queries(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract unique queries from dataframe."""
    # Group by query_id to avoid regenerating same query multiple times
    queries = []
    seen = set()

    for _, row in df.iterrows():
        # Get query_id (unique identifier)
        query_id = row.get('query_id')

        # Extract question from input_messages (new schema) or fallback to old columns
        if 'input_messages' in row.index:
            question = extract_question_from_input_messages(row.get('input_messages'))
        else:
            question = row.get('question', row.get('input', ''))

        # Use query_id as key if available, else use question+split
        if query_id:
            query_key = query_id
        else:
            query_key = (question, row.get('split', 'unknown'))

        if query_key in seen:
            continue
        seen.add(query_key)

        # Extract metadata - handle JSON string
        metadata = row.get('source_metadata', row.get('metadata', {}))
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                metadata = {}

        queries.append({
            'query_id': query_id,
            'question': question,
            'ground_truth': row.get('ground_truth_answer', row.get('ground_truth', row.get('expected_answer', ''))),
            'split': row.get('split', 'unknown'),
            'metadata': metadata,
            'original_index': row.name,
        })

    logger.info(f"Extracted {len(queries)} unique queries")
    return queries


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
        candidates = await generate_candidates(
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
) -> List[Dict[str, Any]]:
    """Regenerate all queries with concurrency control."""
    from openai import AsyncOpenAI
    import os

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def process_with_semaphore(query):
        async with semaphore:
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

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        logger.info(f"Progress: {i+1}/{len(queries)}")

    return results


def merge_results(
    original_df: pd.DataFrame,
    regenerated: List[Dict[str, Any]],
    split: Optional[str],
) -> pd.DataFrame:
    """Merge regenerated results back into dataframe."""
    # Create mapping from question to new candidates
    regen_map = {}
    for item in regenerated:
        question = item['query']['question']
        regen_map[question] = item['candidates']

    # Update rows
    new_rows = []
    for _, row in original_df.iterrows():
        # Check if this row should be updated
        if split and row.get('split') != split:
            new_rows.append(row.to_dict())
            continue

        question = row.get('question', row.get('input', ''))
        if question in regen_map and regen_map[question]:
            # Get candidate index for this row
            candidate_idx = row.get('candidate_index', 0)
            candidates = regen_map[question]

            if candidate_idx < len(candidates):
                new_candidate = candidates[candidate_idx]
                # Update row with new candidate data
                row_dict = row.to_dict()
                row_dict['raw_response'] = new_candidate.get('raw_text', '')
                row_dict['analysis'] = new_candidate.get('analysis', '')
                row_dict['final_answer'] = new_candidate.get('final_answer', '')
                row_dict['regenerated_at'] = datetime.now().isoformat()
                new_rows.append(row_dict)
            else:
                new_rows.append(row.to_dict())
        else:
            new_rows.append(row.to_dict())

    return pd.DataFrame(new_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate specific splits or failed rows from parquet file",
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
        default="gpt-5.1",
        help="Model to use for regeneration (default: gpt-5.1)",
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
        help="Max tokens per response (default: 131072)",
    )

    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)",
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
        print(f"DRY RUN - Would regenerate:")
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

    # Regenerate
    logger.info(f"Starting regeneration of {len(queries)} queries...")
    results = await regenerate_all(
        queries=queries,
        model=args.model,
        persona=persona,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )

    # Merge results
    merged_df = merge_results(df, results, args.split)

    # Save
    output_path = args.output or args.input
    if output_path == args.input:
        # Create backup
        backup_path = args.input.with_suffix('.parquet.bak')
        logger.info(f"Creating backup: {backup_path}")
        df.to_parquet(backup_path)

    logger.info(f"Saving to {output_path}")
    merged_df.to_parquet(output_path)

    print(f"\n{'='*60}")
    print(f"Regeneration complete!")
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
