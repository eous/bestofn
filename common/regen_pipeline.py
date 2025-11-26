#!/usr/bin/env python3
"""
Regeneration Pipeline Utilities.

Shared utilities for regenerating candidates from existing parquet files.
Used by both OpenAI and Claude regen.py scripts.

Key functions:
- load_parquet: Load parquet file into DataFrame
- filter_rows: Filter rows by split and/or failed status
- extract_unique_queries: Extract unique queries from DataFrame
- merge_results: Merge regenerated candidates back into DataFrame
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_question_from_input_messages(input_messages) -> str:
    """
    Extract question from input_messages JSON field.

    Args:
        input_messages: Input messages (JSON string or list of dicts)

    Returns:
        Question text from user message, or empty string if not found

    Example:
        question = extract_question_from_input_messages(row['input_messages'])
    """
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
    """
    Load parquet file into DataFrame.

    Args:
        path: Path to parquet file

    Returns:
        Loaded DataFrame

    Example:
        df = load_parquet(Path("results/experiment.parquet"))
    """
    logger.info(f"Loading {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def filter_rows(
    df: pd.DataFrame,
    split: Optional[str] = None,
    failed_only: bool = False,
) -> pd.DataFrame:
    """
    Filter DataFrame rows for regeneration.

    Args:
        df: DataFrame to filter
        split: Optional split to filter by (e.g., 'math', 'code', 'tool_calling')
        failed_only: If True, only include rows where verification failed

    Returns:
        Filtered DataFrame

    Example:
        filtered_df = filter_rows(df, split='tool_calling', failed_only=True)
    """
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
    """
    Extract unique queries from DataFrame.

    Groups by query_id to avoid regenerating the same query multiple times.

    Args:
        df: DataFrame containing queries

    Returns:
        List of unique query dicts with keys:
        - query_id: Unique identifier
        - question: Question text
        - ground_truth: Expected answer
        - split: Dataset split
        - metadata: Source metadata
        - original_index: Original row index

    Example:
        queries = extract_unique_queries(filtered_df)
        for query in queries:
            results = await regenerate_query(query, ...)
    """
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


def clean_nan_values(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace NaN values with appropriate defaults.

    Prevents data corruption when writing to parquet.

    Args:
        row_dict: Row data as dictionary

    Returns:
        Cleaned dictionary with NaN values replaced

    Example:
        clean_row = clean_nan_values(row.to_dict())
    """
    for key, value in row_dict.items():
        if pd.isna(value):
            # Replace NaN with empty string for text fields, None for others
            if key in ('raw_response', 'analysis', 'final_answer', 'question', 'input'):
                row_dict[key] = ''
            else:
                row_dict[key] = None
    return row_dict


def merge_results(
    original_df: pd.DataFrame,
    regenerated: List[Dict[str, Any]],
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge regenerated results back into DataFrame.

    Uses query_id as primary key, falling back to question text.

    Args:
        original_df: Original DataFrame
        regenerated: List of regeneration results with 'query' and 'candidates' keys
        split: Optional split to filter updates to

    Returns:
        Updated DataFrame with regenerated candidates merged in

    Example:
        updated_df = merge_results(original_df, regenerated_results, split='tool_calling')
        updated_df.to_parquet("output.parquet")
    """
    # Create mapping using query_id as primary key (falls back to question)
    # This prevents overwrites when multiple rows have the same question
    regen_map = {}
    for item in regenerated:
        query_id = item['query'].get('query_id')
        question = item['query']['question']
        # Use query_id if available, otherwise fall back to question
        key = query_id if query_id else question
        regen_map[key] = item['candidates']

    # Update rows
    new_rows = []
    for _, row in original_df.iterrows():
        # Check if this row should be updated
        if split and row.get('split') != split:
            new_rows.append(clean_nan_values(row.to_dict()))
            continue

        # Try to match by query_id first, then by question
        query_id = row.get('query_id')
        question = row.get('question', row.get('input', ''))
        key = query_id if query_id and query_id in regen_map else question

        if key in regen_map and regen_map[key]:
            # Get candidate index for this row
            candidate_idx = row.get('candidate_index', 0)
            # Handle NaN candidate_index
            if pd.isna(candidate_idx):
                candidate_idx = 0
            else:
                candidate_idx = int(candidate_idx)
            candidates = regen_map[key]

            if candidate_idx < len(candidates):
                new_candidate = candidates[candidate_idx]
                # Update row with new candidate data
                row_dict = clean_nan_values(row.to_dict())
                row_dict['raw_response'] = new_candidate.get('raw_text', '')
                row_dict['analysis'] = new_candidate.get('analysis', '')
                row_dict['final_answer'] = new_candidate.get('final_answer', '')
                row_dict['regenerated_at'] = datetime.now().isoformat()
                new_rows.append(row_dict)
            else:
                new_rows.append(clean_nan_values(row.to_dict()))
        else:
            new_rows.append(clean_nan_values(row.to_dict()))

    return pd.DataFrame(new_rows)


def save_checkpoint(
    checkpoint_path: Path,
    completed: int,
    total: int,
    results: List[Dict[str, Any]],
) -> bool:
    """
    Save checkpoint for crash recovery.

    Args:
        checkpoint_path: Base path for checkpoint file
        completed: Number of completed queries
        total: Total number of queries
        results: Current results list

    Returns:
        True if checkpoint saved successfully, False otherwise

    Example:
        if completed % 10 == 0:
            save_checkpoint(output_path, completed, total, results)
    """
    try:
        checkpoint_file = checkpoint_path.with_suffix('.checkpoint.json')
        checkpoint_data = {
            'completed': completed,
            'total': total,
            'results': results,
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        logger.debug(f"Checkpoint saved: {completed}/{total} queries")
        return True
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")
        return False
