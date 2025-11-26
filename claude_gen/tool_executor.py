#!/usr/bin/env python3
"""
Tool calling execution loop for Claude API.

Handles the complete tool-calling lifecycle for Claude:
1. Extract tools from Nemotron metadata
2. Pass tools to Claude API
3. Execute tool calls via sandbox
4. Feed results back to model as tool_result blocks
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

from anthropic import AsyncAnthropic

# Add parent directory to path for verifiers module
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import tool sandbox for execution
from verifiers.tool_sandbox import ToolSandbox


logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Result of tool call execution."""
    tool_use_id: str
    tool_name: str
    result: Any
    error: Optional[str] = None


class ClaudeToolExecutor:
    """
    Executes tool-calling loop with Claude API.

    Features:
    - Extracts tools from Nemotron metadata
    - Manages conversation state across tool call rounds
    - Executes tools in sandbox
    - Handles errors gracefully
    - Supports configurable max iterations
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        max_iterations: int = 3,
        timeout: float = 10.0,
    ):
        """
        Initialize tool executor.

        Args:
            client: AsyncAnthropic client instance
            max_iterations: Maximum number of tool-call rounds (default: 3)
            timeout: Timeout for tool execution in seconds (default: 10.0)
        """
        self.client = client
        self.max_iterations = max_iterations
        self.sandbox = ToolSandbox(config={"timeout": timeout})

    @staticmethod
    def convert_to_claude_format(tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a tool from OpenAI/Nemotron format to Claude format.

        OpenAI/Nemotron format:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Claude format:
            {"name": ..., "description": ..., "input_schema": ...}
        """
        # If already in Claude format (has input_schema), return as-is
        if 'input_schema' in tool:
            return tool

        # Extract from nested function object
        if tool.get('type') == 'function' and 'function' in tool:
            func = tool['function']
            return {
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {'type': 'object', 'properties': {}}),
            }

        # If has top-level name (Responses API format), convert
        if 'name' in tool:
            return {
                'name': tool.get('name', ''),
                'description': tool.get('description', ''),
                'input_schema': tool.get('parameters', {'type': 'object', 'properties': {}}),
            }

        # Unknown format - return as-is
        return tool

    @staticmethod
    def extract_tools_from_metadata(metadata: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tools from Nemotron metadata field.

        Args:
            metadata: Metadata field from dataset row (can be dict or JSON string)

        Returns:
            List of tool definitions in Claude format, or None if no tools
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

        # Convert tools to Claude format
        converted_tools = [
            ClaudeToolExecutor.convert_to_claude_format(t)
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
            tools: List of tool definitions in Claude format
            persona: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Tuple of (final_answer, conversation_history, metadata)
        """
        # Track conversation history
        messages: List[Dict[str, Any]] = []

        # Add initial user message
        messages.append({
            'role': 'user',
            'content': question,
        })

        # Track metadata
        metadata = {
            'iterations': 0,
            'total_tool_calls': 0,
            'tool_calls_by_name': {},
            'errors': [],
            'tools_available': [t.get('name', 'unknown') for t in tools],
        }

        # Execute tool-calling loop
        for iteration in range(self.max_iterations):
            metadata['iterations'] = iteration + 1
            logger.info(f"Tool-calling iteration {iteration + 1}/{self.max_iterations}")

            try:
                # Call Claude API with tools
                response = await self.client.messages.create(
                    model=model,
                    system=persona if persona else "You are a helpful assistant.",
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Check response for tool_use blocks
                has_tool_use = False
                tool_use_blocks = []
                text_content = []

                for block in response.content:
                    if block.type == 'tool_use':
                        has_tool_use = True
                        tool_use_blocks.append(block)
                    elif block.type == 'text':
                        text_content.append(block.text)

                # If no tool calls, we have the final answer
                if not has_tool_use:
                    final_answer = '\n'.join(text_content) if text_content else str(response.content)
                    logger.info("No tool calls detected - returning final answer")

                    # Add final assistant message to history
                    messages.append({
                        'role': 'assistant',
                        'content': response.content,  # Keep original content blocks
                    })

                    return final_answer, messages, metadata

                # We have tool calls - add assistant response to messages
                messages.append({
                    'role': 'assistant',
                    'content': response.content,  # Contains tool_use blocks
                })

                # Execute tool calls
                logger.info(f"Executing {len(tool_use_blocks)} tool call(s)")
                metadata['total_tool_calls'] += len(tool_use_blocks)

                tool_results = []
                for tool_use in tool_use_blocks:
                    tool_name = tool_use.name
                    tool_id = tool_use.id

                    # Parse tool input (Claude returns dict, but handle string case for robustness)
                    try:
                        tool_input = tool_use.input
                        if isinstance(tool_input, str):
                            tool_input = json.loads(tool_input)
                    except (json.JSONDecodeError, TypeError) as e:
                        error_msg = f"Invalid tool input for {tool_name}: {str(tool_use.input)[:100]}"
                        logger.warning(f"Tool {tool_name}: {error_msg}")
                        metadata['errors'].append(error_msg)
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': json.dumps({'error': error_msg}),
                            'is_error': True,
                        })
                        continue

                    # Track tool usage
                    metadata['tool_calls_by_name'][tool_name] = metadata['tool_calls_by_name'].get(tool_name, 0) + 1

                    # Execute tool in sandbox (wrapped in executor to avoid blocking event loop
                    # when sandbox makes sync LLM calls for unknown tools)
                    logger.debug(f"Executing {tool_name}({json.dumps(tool_input)[:100]}...)")
                    try:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None,  # Use default thread pool
                            self.sandbox.execute_tool_call,
                            tool_name,
                            tool_input
                        )
                    except Exception as executor_error:
                        # Handle thread pool or sandbox execution errors
                        error_msg = f"Executor error for {tool_name}: {str(executor_error)[:200]}"
                        logger.error(error_msg)
                        metadata['errors'].append(error_msg)
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': json.dumps({'error': error_msg}),
                            'is_error': True,
                        })
                        continue

                    if result.get('success'):
                        result_content = json.dumps(result['result'])
                        logger.debug(f"Tool {tool_name} succeeded: {result_content[:100]}...")
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': result_content,
                        })
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        logger.warning(f"Tool {tool_name} failed: {error_msg}")
                        metadata['errors'].append(f"{tool_name}: {error_msg}")
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': tool_id,
                            'content': json.dumps({'error': error_msg}),
                            'is_error': True,
                        })

                # Add tool results as user message
                messages.append({
                    'role': 'user',
                    'content': tool_results,
                })

            except Exception as e:
                error_msg = f"Error in tool-calling loop iteration {iteration + 1}: {e}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)

                return f"Error during tool execution: {e}", messages, metadata

        # Max iterations reached without final answer
        logger.warning(f"Max iterations ({self.max_iterations}) reached without final answer")
        metadata['errors'].append(f"Max iterations ({self.max_iterations}) reached")

        return "Unable to complete task within iteration limit.", messages, metadata

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
            persona: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            List of candidate dicts
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
                    'raw_text': final_answer,
                    'analysis': '',
                    'conversation_history': conversation,
                    'tool_metadata': metadata,
                    'tools_used': tools,
                })

            except Exception as e:
                logger.error(f"Failed to generate candidate {i+1}: {e}")
                results.append({
                    'final_answer': f"Error: {e}",
                    'raw_text': f"Error: {e}",
                    'analysis': '',
                    'conversation_history': [
                        {'role': 'user', 'content': question},
                    ],
                    'tool_metadata': {'errors': [str(e)], 'iterations': 0, 'total_tool_calls': 0},
                    'tools_used': tools,
                })

        return results


# Convenience function for integration
async def generate_candidates_with_tool_calling(
    client: AsyncAnthropic,
    model: str,
    question: str,
    metadata: Any,
    n: int,
    persona: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 70000,
    max_iterations: int = 3,
    sem: Optional[asyncio.Semaphore] = None,  # Accept semaphore for rate limiting compatibility
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for tool-calling generation with Claude.

    Extracts tools from metadata and generates N candidates with tool execution.
    Falls back to empty list if no tools found.

    Args:
        client: AsyncAnthropic client
        model: Model name
        question: User query
        metadata: Row metadata (may contain tools)
        n: Number of candidates
        persona: Optional persona/instructions
        temperature: Sampling temperature
        max_tokens: Max output tokens
        max_iterations: Max tool-calling iterations

    Returns:
        List of candidate dicts
    """
    # Extract tools
    tools = ClaudeToolExecutor.extract_tools_from_metadata(metadata)

    if not tools:
        logger.warning("No tools found in metadata - cannot execute tool calling")
        return []

    # Create executor and generate candidates
    executor = ClaudeToolExecutor(
        client=client,
        max_iterations=max_iterations,
    )

    # Use semaphore for rate limiting if provided (consistent with openai_gen pattern)
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
