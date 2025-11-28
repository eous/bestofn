#!/usr/bin/env python3
"""
Tool calling execution loop for OpenAI Responses API.

Handles the complete tool-calling lifecycle:
1. Extract tools from Nemotron metadata
2. Pass tools to API
3. Execute tool calls via Docker sandbox
4. Feed results back to model
5. Loop until final answer or max iterations

Uses deterministic mock implementations for reproducibility.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI

# Add parent directory to path for verifiers module
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import tool sandbox for execution
from verifiers.tool_sandbox import ToolSandbox


logger = logging.getLogger(__name__)


# =============================================================================
# Singleton ToolSandbox for efficient resource management
# =============================================================================

import threading

_shared_tool_sandbox: Optional[ToolSandbox] = None
_tool_sandbox_lock = threading.Lock()


def get_shared_tool_sandbox(timeout: float = 40.0) -> ToolSandbox:
    """
    Get or create shared ToolSandbox singleton (thread-safe).

    Using a singleton prevents creating hundreds of Docker clients and containers
    during large batch runs. The sandbox's internal container pool is reused
    across all queries, dramatically reducing file descriptor usage.

    Args:
        timeout: Tool execution timeout in seconds

    Returns:
        Shared ToolSandbox instance with container pooling enabled

    Raises:
        RuntimeError: If lock acquisition times out (Docker may be stuck)
    """
    global _shared_tool_sandbox
    # Double-checked locking pattern for thread safety
    if _shared_tool_sandbox is None:
        acquired = _tool_sandbox_lock.acquire(timeout=60)
        if not acquired:
            raise RuntimeError("Timeout acquiring ToolSandbox lock - Docker may be stuck")
        try:
            if _shared_tool_sandbox is None:
                logger.info("Creating shared ToolSandbox singleton with container pooling")
                _shared_tool_sandbox = ToolSandbox(config={
                    "timeout": timeout,
                    "container_pool_size": 5,  # Enable pooling to reuse containers
                })
        finally:
            _tool_sandbox_lock.release()
    return _shared_tool_sandbox


def cleanup_shared_tool_sandbox():
    """
    Clean up the shared ToolSandbox singleton (thread-safe).

    Call this at the end of a batch run to release Docker resources.
    """
    global _shared_tool_sandbox
    with _tool_sandbox_lock:
        if _shared_tool_sandbox is not None:
            try:
                _shared_tool_sandbox.sandbox.cleanup()
                logger.info("Cleaned up shared ToolSandbox")
            except Exception as e:
                logger.warning(f"Error cleaning up shared ToolSandbox: {e}")
            _shared_tool_sandbox = None


@dataclass
class ToolCallResult:
    """Result of tool call execution."""
    tool_call_id: str
    tool_name: str
    result: Any
    error: Optional[str] = None


class ToolExecutor:
    """
    Executes tool-calling loop with OpenAI Responses API.

    Features:
    - Extracts tools from Nemotron metadata
    - Manages conversation state across tool call rounds
    - Executes tools in Docker sandbox
    - Handles errors gracefully
    - Supports configurable max iterations
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        max_iterations: int = 100,
        timeout: float = 40.0,
    ):
        """
        Initialize tool executor.

        Args:
            client: AsyncOpenAI client instance
            max_iterations: Maximum number of tool-call rounds (default: 100)
            timeout: Timeout for tool execution in seconds (default: 40.0)
        """
        self.client = client
        self.max_iterations = max_iterations
        # Use shared singleton to prevent file descriptor exhaustion
        self.sandbox = get_shared_tool_sandbox(timeout=timeout)

    @staticmethod
    def convert_to_responses_api_format(tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a tool from Chat Completions format to Responses API format.

        Chat Completions format (Nemotron):
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Responses API format:
            {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        # If already in Responses API format (has top-level 'name'), return as-is
        if 'name' in tool:
            return tool

        # Convert from Chat Completions format
        if tool.get('type') == 'function' and 'function' in tool:
            func = tool['function']
            return {
                'type': 'function',
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'parameters': func.get('parameters', {}),
            }

        # Unknown format - return as-is and let API error
        return tool

    @staticmethod
    def extract_tools_from_metadata(metadata: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tools from Nemotron metadata field.

        Args:
            metadata: Metadata field from dataset row (can be dict or JSON string)

        Returns:
            List of tool definitions in OpenAI Responses API format, or None if no tools

        Example metadata format (Nemotron/Chat Completions):
            {"tools": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get weather for location",
                    "parameters": {...}
                }}
            ]}

        Output format (Responses API):
            [{"type": "function", "name": "get_weather", "description": "...", "parameters": {...}}]
        """
        # Parse if JSON string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                logger.debug("Failed to parse metadata as JSON")
                return None

        # Extract tools array
        if not isinstance(metadata, dict):
            return None

        tools = metadata.get('tools')
        if not tools or not isinstance(tools, list):
            return None

        # Convert tools to Responses API format
        converted_tools = [
            ToolExecutor.convert_to_responses_api_format(t)
            for t in tools
        ]

        logger.debug(f"Extracted and converted {len(converted_tools)} tools from metadata")
        return converted_tools

    async def execute_with_tools(
        self,
        model: str,
        question: str,
        tools: List[Dict[str, Any]],
        persona: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 70000,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute query with tool-calling loop.

        Args:
            model: Model name
            question: User query
            tools: List of tool definitions in OpenAI format
            persona: Optional system/developer instructions
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Tuple of (final_answer, conversation_history, metadata)

        The conversation_history contains all messages in OpenAI format:
        - {"role": "user", "content": "..."}
        - {"role": "assistant", "content": "...", "tool_calls": [...]}
        - {"role": "tool", "tool_call_id": "...", "content": "..."}
        - {"role": "assistant", "content": "final answer"}

        The metadata contains execution statistics.
        """
        # Track the full conversation history for training data
        conversation_history: List[Dict[str, Any]] = []

        # Add initial user message
        conversation_history.append({
            'role': 'user',
            'content': question,
        })

        # Track metadata
        metadata = {
            'iterations': 0,
            'total_tool_calls': 0,
            'tool_calls_by_name': {},
            'errors': [],
            'tools_available': [t.get('function', {}).get('name', 'unknown') for t in tools],
        }

        # For responses API, we build up input differently than chat completions
        # We'll track what to send as input
        current_input = question

        # Execute tool-calling loop
        for iteration in range(self.max_iterations):
            metadata['iterations'] = iteration + 1
            logger.info(f"Tool-calling iteration {iteration + 1}/{self.max_iterations} (model={model})")

            try:
                # Call API with tools
                # Note: temperature is not passed because reasoning mode doesn't support it
                response = await self.client.responses.create(
                    model=model,
                    instructions=persona if persona else None,
                    input=current_input,
                    tools=tools,
                    reasoning={"effort": "medium"},
                    max_output_tokens=max_tokens,
                )

                # Check if response contains tool calls
                has_tool_calls = False
                tool_calls = []
                assistant_content = ""

                # Parse response output for tool calls and text
                # The responses API returns output as list of ResponseOutputItem
                if hasattr(response, 'output') and response.output:
                    for item in response.output:
                        # Check for function_call type (OpenAI Responses API format)
                        if hasattr(item, 'type'):
                            if item.type == 'function_call':
                                has_tool_calls = True
                                call_id = getattr(item, 'call_id', f"call_{iteration}_{len(tool_calls)}")
                                tool_calls.append({
                                    'id': call_id,
                                    'type': 'function',
                                    'function': {
                                        'name': item.name,
                                        'arguments': item.arguments,
                                    }
                                })
                            elif item.type == 'message' and hasattr(item, 'content'):
                                # Capture any text content
                                for content_part in item.content:
                                    if hasattr(content_part, 'text'):
                                        assistant_content += content_part.text

                # Also check output_text for text content
                if hasattr(response, 'output_text') and response.output_text:
                    if not assistant_content:
                        assistant_content = response.output_text

                # If no tool calls, we have the final answer
                if not has_tool_calls:
                    final_answer = assistant_content or str(response.output)
                    logger.info("No tool calls detected - returning final answer")

                    # Add final assistant message to conversation
                    conversation_history.append({
                        'role': 'assistant',
                        'content': final_answer,
                    })

                    return final_answer, conversation_history, metadata

                # We have tool calls - add assistant message with tool_calls to history
                assistant_msg = {
                    'role': 'assistant',
                    'content': assistant_content,  # May be empty, that's OK
                    'tool_calls': tool_calls,
                }
                conversation_history.append(assistant_msg)

                # Execute tool calls
                logger.info(f"Executing {len(tool_calls)} tool call(s)")
                metadata['total_tool_calls'] += len(tool_calls)

                tool_results = []
                for tool_call in tool_calls:
                    func_name = tool_call['function']['name']
                    func_args_str = tool_call['function']['arguments']

                    # Track tool usage
                    metadata['tool_calls_by_name'][func_name] = metadata['tool_calls_by_name'].get(func_name, 0) + 1

                    # Parse arguments
                    try:
                        if isinstance(func_args_str, str):
                            func_args = json.loads(func_args_str)
                        else:
                            func_args = func_args_str
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON arguments: {func_args_str[:100]}"
                        logger.warning(f"Tool {func_name}: {error_msg}")
                        metadata['errors'].append(error_msg)
                        result_content = json.dumps({'error': error_msg})

                        tool_result_msg = {
                            'role': 'tool',
                            'tool_call_id': tool_call['id'],
                            'name': func_name,
                            'content': result_content,
                        }
                        tool_results.append(tool_result_msg)
                        conversation_history.append(tool_result_msg)
                        continue

                    # Execute tool in sandbox (wrapped in executor to avoid blocking event loop
                    # when sandbox makes sync LLM calls for unknown tools)
                    logger.debug(f"Executing {func_name}({json.dumps(func_args)[:100]}...)")
                    try:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None,  # Use default thread pool
                            self.sandbox.execute_tool_call,
                            func_name,
                            func_args
                        )
                    except Exception as executor_error:
                        # Handle thread pool or sandbox execution errors
                        error_msg = f"Executor error for {func_name}: {str(executor_error)[:200]}"
                        logger.error(error_msg)
                        metadata['errors'].append(error_msg)
                        tool_result_msg = {
                            'role': 'tool',
                            'tool_call_id': tool_call['id'],
                            'name': func_name,
                            'content': json.dumps({'error': error_msg}),
                        }
                        tool_results.append(tool_result_msg)
                        conversation_history.append(tool_result_msg)
                        continue

                    if result.get('success'):
                        result_content = json.dumps(result['result'])
                        logger.debug(f"Tool {func_name} succeeded: {result_content[:100]}...")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        logger.warning(f"Tool {func_name} failed: {error_msg}")
                        metadata['errors'].append(f"{func_name}: {error_msg}")
                        result_content = json.dumps({'error': error_msg})

                    tool_result_msg = {
                        'role': 'tool',
                        'tool_call_id': tool_call['id'],
                        'name': func_name,
                        'content': result_content,
                    }
                    tool_results.append(tool_result_msg)
                    conversation_history.append(tool_result_msg)

                # Build new input with tool results for next iteration
                # For responses API, format as continuation with tool results
                tool_results_text = "\n\n".join([
                    f"Tool '{r['name']}' (call_id: {r['tool_call_id']}) returned:\n{r['content']}"
                    for r in tool_results
                ])

                current_input = (
                    f"Original question: {question}\n\n"
                    f"Tool execution results:\n{tool_results_text}\n\n"
                    f"Based on these tool results, provide your final answer to the original question."
                )

            except Exception as e:
                error_msg = f"Error in tool-calling loop iteration {iteration + 1}: {e}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)

                # Add error to conversation history
                conversation_history.append({
                    'role': 'assistant',
                    'content': f"Error during tool execution: {e}",
                    'error': True,
                })

                return f"Error during tool execution: {e}", conversation_history, metadata

        # Max iterations reached without final answer
        logger.warning(f"Max iterations ({self.max_iterations}) reached without final answer")
        metadata['errors'].append(f"Max iterations ({self.max_iterations}) reached")

        # Add final message indicating incomplete
        conversation_history.append({
            'role': 'assistant',
            'content': "Unable to complete task within iteration limit.",
            'incomplete': True,
        })

        return "Unable to complete task within iteration limit.", conversation_history, metadata

    async def generate_candidates_with_tools(
        self,
        model: str,
        question: str,
        tools: List[Dict[str, Any]],
        n: int,
        persona: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 70000,
    ) -> List[Dict[str, Any]]:
        """
        Generate N candidates with tool calling.

        Args:
            model: Model name
            question: User query
            tools: List of tool definitions
            n: Number of candidates to generate
            persona: Optional system/developer instructions
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            List of candidate dicts with keys:
            - 'final_answer': Final answer text
            - 'raw_text': Full response text
            - 'conversation_history': Full multi-turn exchange as list of messages
            - 'tool_metadata': Execution metadata (iterations, tool counts, errors)
            - 'tools_used': List of tool definitions that were available
        """
        results = []

        for i in range(n):
            logger.info(f"Generating tool-calling candidate {i+1}/{n}")

            try:
                final_answer, conversation, metadata = await self.execute_with_tools(
                    model=model,
                    question=question,
                    tools=tools,
                    persona=persona,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                results.append({
                    'final_answer': final_answer,
                    'raw_text': final_answer,  # For compatibility with existing pipeline
                    'analysis': '',  # No separate analysis field for tool calls
                    'conversation_history': conversation,  # Full multi-turn exchange
                    'tool_metadata': metadata,
                    'tools_used': tools,  # Include tool definitions for reproducibility
                })

            except Exception as e:
                logger.error(f"Failed to generate candidate {i+1}: {e}")
                # Return error result with empty conversation
                results.append({
                    'final_answer': f"Error: {e}",
                    'raw_text': f"Error: {e}",
                    'analysis': '',
                    'conversation_history': [
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': f"Error: {e}", 'error': True},
                    ],
                    'tool_metadata': {'errors': [str(e)], 'iterations': 0, 'total_tool_calls': 0},
                    'tools_used': tools,
                })

        return results


# Convenience function for integration
async def generate_candidates_with_tool_calling(
    client: AsyncOpenAI,
    model: str,
    question: str,
    metadata: Any,
    n: int,
    persona: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 70000,
    max_iterations: int = 100,
    sem: Optional[asyncio.Semaphore] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for tool-calling generation.

    Extracts tools from metadata and generates N candidates with tool execution.
    Falls back to standard generation if no tools found.

    Args:
        client: AsyncOpenAI client
        model: Model name
        question: User query
        metadata: Row metadata (may contain tools)
        n: Number of candidates
        persona: Optional persona/instructions
        temperature: Sampling temperature
        max_tokens: Max output tokens
        max_iterations: Max tool-calling iterations
        sem: Optional semaphore for rate limiting (for API consistency with claude_gen)

    Returns:
        List of candidate dicts
    """
    # Extract tools
    tools = ToolExecutor.extract_tools_from_metadata(metadata)

    if not tools:
        logger.warning("No tools found in metadata - cannot execute tool calling")
        # Return empty results - caller should handle fallback
        return []

    # Create executor and generate candidates
    executor = ToolExecutor(
        client=client,
        max_iterations=max_iterations,
    )

    # Use semaphore for rate limiting if provided (consistent with claude_gen pattern)
    if sem:
        async with sem:
            return await executor.generate_candidates_with_tools(
                model=model,
                question=question,
                tools=tools,
                n=n,
                persona=persona,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    else:
        return await executor.generate_candidates_with_tools(
            model=model,
            question=question,
            tools=tools,
            n=n,
            persona=persona,
            temperature=temperature,
            max_tokens=max_tokens,
        )
