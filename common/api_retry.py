#!/usr/bin/env python3
"""
API Retry Logic with Exponential Backoff.

Provides a generic async retry wrapper for API calls that handles:
- Timeout errors
- Rate limits (429)
- Server errors (500, 502, 503)
- Connection errors

Uses exponential backoff with jitter to avoid thundering herd effects.
"""

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 300.0  # 5 minutes for extended thinking/reasoning
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0  # Base delay for exponential backoff

# Retryable error patterns (case-insensitive)
RETRYABLE_PATTERNS = [
    'rate_limit', 'rate limit', '429',
    'overloaded', '503', '502', '500',
    'connection', 'timeout', 'server_error'
]


async def call_with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    operation_name: str = "API call",
) -> Any:
    """
    Call an async function with timeout and exponential backoff retry.

    Args:
        coro_factory: Function that returns a coroutine (e.g., lambda: client.chat(...))
        timeout: Timeout in seconds per attempt
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (multiplied by 2^attempt)
        operation_name: Human-readable name for logging

    Returns:
        The result of the coroutine

    Raises:
        asyncio.TimeoutError: If all attempts time out
        Exception: The last exception if all retries are exhausted

    Example:
        response = await call_with_retry(
            lambda: client.messages.create(model="claude-3", messages=[...]),
            timeout=120.0,
            max_retries=3,
        )
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            # Wrap API call in timeout
            response = await asyncio.wait_for(coro_factory(), timeout=timeout)
            return response

        except asyncio.TimeoutError:
            last_exception = asyncio.TimeoutError(
                f"{operation_name} timed out after {timeout}s"
            )
            logger.warning(
                f"{operation_name} timeout on attempt {attempt + 1}/{max_retries}"
            )

        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if this is a retryable error
            is_retryable = any(pattern in error_str for pattern in RETRYABLE_PATTERNS)

            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                logger.info(f"Retrying {operation_name} in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                # Non-retryable error or last attempt
                raise

    # All retries exhausted
    raise last_exception or RuntimeError(f"{operation_name} failed after all retries")


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable.

    Args:
        exception: The exception to check

    Returns:
        True if the error is retryable (rate limit, server error, connection issue)
    """
    error_str = str(exception).lower()
    return any(pattern in error_str for pattern in RETRYABLE_PATTERNS)
