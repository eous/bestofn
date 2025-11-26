#!/usr/bin/env python3
"""
Minimal Nemotron dataset utilities for Best-of-N generation.

Provides:
- Extract first user message from Nemotron conversation format
- Skip invalid samples (empty or "-" messages)
- Auto-select correct dataset version (v2 for code/math, v1 for tool_calling)

Adapted from scripts/gpt_oss/dataset.py validation logic.
"""

from typing import Optional
from datasets import load_dataset


def extract_first_user_message(sample: dict) -> Optional[str]:
    """
    Extract first user message from Nemotron sample.

    Args:
        sample: Nemotron dataset sample with 'messages' field

    Returns:
        First user message content, or None if invalid

    Examples:
        >>> sample = {'messages': [
        ...     {'role': 'user', 'content': 'Hello'},
        ...     {'role': 'assistant', 'content': 'Hi there'}
        ... ]}
        >>> extract_first_user_message(sample)
        'Hello'

        >>> sample = {'messages': [{'role': 'user', 'content': ''}]}
        >>> extract_first_user_message(sample)
        None  # Empty message - skip

        >>> sample = {'messages': [{'role': 'user', 'content': '-'}]}
        >>> extract_first_user_message(sample)
        None  # Placeholder - skip
    """
    # Check messages array exists
    if 'messages' not in sample or not sample['messages']:
        return None

    # Find first user message
    for msg in sample['messages']:
        # Handle both dict (v2) and object (v1) formats
        if isinstance(msg, dict):
            role = msg.get('role')
            content = msg.get('content', '')
        elif hasattr(msg, 'role') and hasattr(msg, 'content'):
            # v1 format: Message objects with attributes
            role = msg.role
            content = msg.content
        else:
            continue

        if role == 'user':
            content = content.strip() if isinstance(content, str) else str(content).strip()

            # Skip empty or placeholder messages
            if content and content != '-':
                return content
            else:
                # Invalid user message - skip this sample
                return None

    # No user message found
    return None


def get_dataset_version(split: str) -> str:
    """
    Return correct Nemotron dataset version for split.

    Args:
        split: Dataset split name

    Returns:
        'v2' for code/math (newer data), 'v1' for tool_calling

    Examples:
        >>> get_dataset_version('code')
        'v2'
        >>> get_dataset_version('math')
        'v2'
        >>> get_dataset_version('tool_calling')
        'v1'
    """
    if split in ['code', 'math']:
        return 'v2'  # Use v2 for code/math (newer data)
    else:
        return 'v1'  # Use v1 for tool_calling (not available in v2)


def load_nemotron_split(split: str, streaming: bool = True):
    """
    Load Nemotron dataset with correct version.

    Auto-selects v2 for code/math, v1 for tool_calling.

    Args:
        split: Dataset split to load ('code', 'math', 'tool_calling')
        streaming: Whether to use streaming mode (default: True)

    Returns:
        HuggingFace dataset (streaming or loaded)

    Examples:
        >>> ds = load_nemotron_split('code', streaming=True)
        # Loads nvidia/Nemotron-Post-Training-Dataset-v2, split='code'

        >>> ds = load_nemotron_split('tool_calling', streaming=True)
        # Loads nvidia/Nemotron-Post-Training-Dataset-v1, split='tool_calling'
    """
    version = get_dataset_version(split)
    dataset_name = f"nvidia/Nemotron-Post-Training-Dataset-{version}"

    return load_dataset(dataset_name, split=split, streaming=streaming)
