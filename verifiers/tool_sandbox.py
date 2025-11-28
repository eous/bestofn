"""
Tool execution sandbox for tool_calling verification.

Uses dynamic mock generation and optional LLM fallback for generating
realistic tool responses without Docker overhead.

Features:
- Dynamic mock generator (category-based, deterministic, ~1ms per call)
- Confidence-based hybrid fallback (dynamic mock → LLM when uncertain)
- LLM mock with schema/context awareness for better responses
- Platform-aware LLM selection (OpenAI gpt-4o-mini or Claude sonnet-4.5)
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, Literal

from .dynamic_mock import (
    generate_dynamic_mock,
    generate_dynamic_mock_with_confidence,
    detect_category,
    detect_category_with_confidence,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Mock Fallback (for low-confidence or generic tools)
# ============================================================================

# Platform type for LLM mock generation
Platform = Literal["openai", "claude"]


def _llm_mock_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]] = None,
    problem_context: Optional[str] = None,
    platform: Platform = "openai",
) -> Dict[str, Any]:
    """
    Use LLM to generate plausible mock response for unknown tool.

    Enhanced version that uses:
    - Tool schema (description, parameter definitions) when available
    - Problem context (the original question being solved) for relevant data
    - Platform-aware model selection (gpt-4o-mini for OpenAI, sonnet-4.5 for Claude)

    Args:
        tool_name: Name of the tool
        arguments: Arguments passed to the tool
        tool_schema: Optional tool schema with description and parameters
        problem_context: Optional context about the problem being solved
        platform: Which platform to use for LLM mock ("openai" or "claude")
    """
    # Build enhanced prompt with all available context
    prompt_parts = [
        "Generate a realistic mock JSON response for this API tool call.",
        "",
        f"Tool name: {tool_name}",
    ]

    # Add tool schema information if available
    if tool_schema:
        if tool_schema.get("description"):
            prompt_parts.append(f"Tool description: {tool_schema['description']}")
        if tool_schema.get("parameters"):
            params_desc = []
            props = tool_schema.get("parameters", {}).get("properties", {})
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                params_desc.append(f"  - {param_name} ({param_type}): {param_desc}")
            if params_desc:
                prompt_parts.append("Parameter definitions:")
                prompt_parts.extend(params_desc)

    prompt_parts.append(f"\nArguments provided: {json.dumps(arguments, indent=2)}")

    # Add problem context if available
    if problem_context:
        ctx = problem_context[:1000] + "..." if len(problem_context) > 1000 else problem_context
        prompt_parts.append(f"\nProblem being solved: {ctx}")

    prompt_parts.extend([
        "",
        "Requirements:",
        "1. Return ONLY valid JSON (no markdown, no explanation)",
        "2. Generate realistic but fake data appropriate for this tool's purpose",
        "3. The response should help solve the problem if context was provided",
        "4. Keep response concise but complete",
        "5. Use appropriate data types (numbers, strings, arrays, etc.)",
        "6. If it's a search/list tool, return 3-5 items",
        "7. If it's a get/fetch tool, return a single detailed object",
        "8. If parameters have descriptions, generate data matching those descriptions",
        "",
        "Think carefully about what this specific tool should return based on its name, "
        "description, and the problem context.",
    ])

    prompt = "\n".join(prompt_parts)

    # Use platform-appropriate LLM
    if platform == "claude":
        return _llm_mock_tool_claude(tool_name, prompt)
    else:
        return _llm_mock_tool_openai(tool_name, prompt)


def _llm_mock_tool_openai(tool_name: str, prompt: str) -> Dict[str, Any]:
    """Generate mock using OpenAI gpt-4o-mini."""
    try:
        from openai import OpenAI
        client = OpenAI()  # Uses OPENAI_API_KEY env var

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)
        logger.info(f"OpenAI LLM generated mock for tool: {tool_name}")
        return result

    except ImportError:
        logger.warning("OpenAI not available for LLM mock fallback")
        return {"error": f"Unknown tool: {tool_name}", "fallback": "openai_unavailable"}
    except Exception as e:
        logger.warning(f"OpenAI LLM mock fallback failed for {tool_name}: {e}")
        return {"error": f"Unknown tool: {tool_name}", "fallback_error": str(e)}


def _llm_mock_tool_claude(tool_name: str, prompt: str) -> Dict[str, Any]:
    """Generate mock using Claude sonnet-4.5."""
    try:
        from anthropic import Anthropic
        client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

        # Claude doesn't have response_format, so we need to be explicit in the prompt
        claude_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no other text."

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{"role": "user", "content": claude_prompt}],
        )

        # Extract text content
        content = response.content[0].text if response.content else "{}"

        # Try to parse JSON from response
        # Handle potential markdown wrapping
        if content.startswith("```"):
            # Extract from code block
            import re
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()

        result = json.loads(content)
        logger.info(f"Claude LLM generated mock for tool: {tool_name}")
        return result

    except ImportError:
        logger.warning("Anthropic not available for LLM mock fallback")
        return {"error": f"Unknown tool: {tool_name}", "fallback": "claude_unavailable"}
    except json.JSONDecodeError as e:
        logger.warning(f"Claude LLM mock returned invalid JSON for {tool_name}: {e}")
        return {"error": f"Unknown tool: {tool_name}", "fallback_error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.warning(f"Claude LLM mock fallback failed for {tool_name}: {e}")
        return {"error": f"Unknown tool: {tool_name}", "fallback_error": str(e)}


# ============================================================================
# ToolSandbox Class
# ============================================================================

class ToolSandbox:
    """
    Mock tool execution sandbox for tool_calling verification.

    Uses dynamic mock generation with optional LLM fallback for
    generating realistic tool responses.

    The hybrid approach:
    1. Use dynamic mock for high-confidence category matches (fast, deterministic)
    2. Fall back to LLM mock for low-confidence/generic tools (slower but contextual)
    3. Always have a fallback - never fail silently
    """

    # Confidence threshold for using LLM fallback instead of dynamic mock
    CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, config: Optional[Dict[str, Any]] = None, platform: Platform = "openai"):
        """
        Initialize the tool sandbox.

        Args:
            config: Optional configuration dict with:
                - confidence_threshold: Override default confidence threshold
            platform: Which LLM platform to use for mock generation ("openai" or "claude")
        """
        self.config = config or {}
        self.platform = platform
        if "confidence_threshold" in self.config:
            self.CONFIDENCE_THRESHOLD = self.config["confidence_threshold"]

    def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_schema: Optional[Dict[str, Any]] = None,
        problem_context: Optional[str] = None,
        force_llm_mock: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a tool call and return mock result.

        Strategy:
        1. If force_llm_mock: skip dynamic mock, go straight to LLM
        2. Generate dynamic mock with confidence score
        3. If confidence >= threshold: use dynamic mock (fast path)
           - Include llm_retry_available flag for potential retry on refusal
        4. If confidence < threshold: use LLM mock only
           - If LLM fails, return error (don't use bad dynamic mock)

        Args:
            tool_name: Name of the tool/function to call
            arguments: Arguments to pass to the tool
            tool_schema: Optional tool schema with description and parameters
            problem_context: Optional context about the problem being solved
            force_llm_mock: If True, bypass dynamic mock and use LLM directly
                           (used when retrying after capability refusal)

        Returns:
            Dict with 'success', 'result', 'source', and optionally 'confidence'
        """
        # Force LLM mock mode - used for retries after capability refusals
        if force_llm_mock:
            logger.info(f"Tool '{tool_name}' - force_llm_mock enabled, using LLM mock (platform={self.platform})")
            try:
                llm_result = _llm_mock_tool(
                    tool_name,
                    arguments,
                    tool_schema=tool_schema,
                    problem_context=problem_context,
                    platform=self.platform,
                )

                if "error" not in llm_result or len(llm_result) > 2:
                    return {
                        "success": True,
                        "result": llm_result,
                        "source": "llm_mock_forced",
                    }

                # LLM failed even with force - return error
                logger.warning(f"Forced LLM mock failed for '{tool_name}'")
                return {
                    "success": False,
                    "error": "Forced LLM mock failed",
                    "source": "error",
                }
            except Exception as e:
                logger.error(f"Forced LLM mock error for '{tool_name}': {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "source": "error",
                }

        # Extract description from schema for category detection
        tool_description = ""
        if tool_schema:
            tool_description = tool_schema.get("description", "")

        try:
            # Generate dynamic mock with confidence score
            dynamic_result, category, confidence = generate_dynamic_mock_with_confidence(
                tool_name, arguments, tool_description
            )

            # If high confidence, use dynamic mock directly (fast path)
            if confidence >= self.CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"Tool '{tool_name}' - using dynamic mock "
                    f"(category: {category}, confidence: {confidence:.2f})"
                )
                return {
                    "success": True,
                    "result": dynamic_result,
                    "source": "dynamic_mock",
                    "confidence": confidence,
                    "category": category,
                    # Flag indicating LLM retry is available if this causes a refusal
                    "llm_retry_available": True,
                }

            # Low confidence - use LLM mock (no fallback to bad dynamic mock)
            logger.info(
                f"Tool '{tool_name}' - low confidence ({confidence:.2f}), "
                f"using LLM mock with schema/context (platform={self.platform})"
            )
            llm_result = _llm_mock_tool(
                tool_name,
                arguments,
                tool_schema=tool_schema,
                problem_context=problem_context,
                platform=self.platform,
            )

            # If LLM succeeded (no error key or has actual data)
            if "error" not in llm_result or len(llm_result) > 2:
                return {
                    "success": True,
                    "result": llm_result,
                    "source": "llm_mock",
                    "dynamic_mock_confidence": confidence,
                }

            # LLM failed - return error, don't use low-confidence dynamic mock
            logger.warning(
                f"LLM mock failed for '{tool_name}' and dynamic mock confidence "
                f"too low ({confidence:.2f}) - returning error"
            )
            return {
                "success": False,
                "error": f"LLM mock failed and dynamic mock confidence too low ({confidence:.2f})",
                "source": "error",
                "dynamic_mock_confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Mock generation failed for '{tool_name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "source": "error",
            }

    def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        problem_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls in OpenAI format:
                [{"function": {"name": "...", "arguments": "..."}}]
            tool_schemas: Optional dict mapping tool names to schemas
            problem_context: Optional context about the problem being solved

        Returns:
            List of results
        """
        results = []
        tool_schemas = tool_schemas or {}

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                results.append({
                    "success": False,
                    "error": f"Invalid arguments JSON: {args_str[:100]}",
                })
                continue

            # Get schema for this tool if available
            schema = tool_schemas.get(name)

            result = self.execute_tool_call(
                name, args,
                tool_schema=schema,
                problem_context=problem_context,
            )
            results.append(result)

        return results

    def verify_tool_call(
        self,
        tool_call: Dict[str, Any],
        expected_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify a tool call produces expected result.

        Args:
            tool_call: Tool call to execute
            expected_result: Expected result

        Returns:
            Dict with 'is_correct', 'actual', 'expected', 'explanation'
        """
        func = tool_call.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            return {
                "is_correct": False,
                "explanation": "Invalid arguments JSON",
            }

        actual = self.execute_tool_call(name, args)

        if not actual.get("success"):
            return {
                "is_correct": False,
                "actual": actual,
                "expected": expected_result,
                "explanation": f"Tool execution failed: {actual.get('error')}",
            }

        actual_result = actual.get("result", {})

        # Compare results (with some flexibility)
        is_correct = self._compare_results(actual_result, expected_result)

        return {
            "is_correct": is_correct,
            "actual": actual_result,
            "expected": expected_result,
            "explanation": "Results match" if is_correct else "Results differ",
        }

    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare results with flexibility for mock vs real data."""
        # Exact match
        if actual == expected:
            return True

        # For dicts, check key overlap and type matching
        if isinstance(actual, dict) and isinstance(expected, dict):
            for key in expected:
                if key not in actual:
                    return False
                if isinstance(expected[key], dict):
                    if not self._compare_results(actual[key], expected[key]):
                        return False
            return True

        # For numbers, allow small differences
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) < 0.01 * max(abs(actual), abs(expected), 1)

        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_tool_call_sandboxed(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]] = None,
    problem_context: Optional[str] = None,
    platform: Platform = "openai",
) -> Dict[str, Any]:
    """
    Execute a single tool call with mock sandbox.

    Args:
        tool_name: Name of the tool
        arguments: Arguments for the tool
        tool_schema: Optional tool schema
        problem_context: Optional problem context
        platform: LLM platform for mock generation ("openai" or "claude")

    Example:
        >>> result = execute_tool_call_sandboxed("get_stock_quote", {"symbol": "AAPL"})
        >>> result["success"]
        True
        >>> "price" in result["result"]
        True
    """
    sandbox = ToolSandbox(platform=platform)
    return sandbox.execute_tool_call(
        tool_name, arguments,
        tool_schema=tool_schema,
        problem_context=problem_context,
    )


def verify_tool_calls(
    candidate_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify candidate tool calls match expected calls.

    Checks:
    1. Same number of calls
    2. Same tool names
    3. Compatible arguments
    4. (Optional) Same results when executed

    Returns:
        Dict with 'is_correct', 'score', 'details'
    """
    if len(candidate_calls) != len(expected_calls):
        return {
            "is_correct": False,
            "score": 0.0,
            "details": f"Call count mismatch: {len(candidate_calls)} vs {len(expected_calls)}",
        }

    sandbox = ToolSandbox()
    matches = 0
    details = []

    for i, (cand, exp) in enumerate(zip(candidate_calls, expected_calls)):
        cand_name = cand.get("function", {}).get("name", "")
        exp_name = exp.get("function", {}).get("name", "")

        if cand_name != exp_name:
            details.append(f"Call {i}: tool name mismatch ({cand_name} vs {exp_name})")
            continue

        # Execute both and compare
        cand_result = sandbox.execute_tool_call(
            cand_name,
            json.loads(cand.get("function", {}).get("arguments", "{}"))
        )
        exp_result = sandbox.execute_tool_call(
            exp_name,
            json.loads(exp.get("function", {}).get("arguments", "{}"))
        )

        if cand_result == exp_result:
            matches += 1
            details.append(f"Call {i}: match ({cand_name})")
        else:
            details.append(f"Call {i}: result mismatch")

    score = matches / len(expected_calls) if expected_calls else 1.0

    return {
        "is_correct": score >= 0.8,
        "score": score,
        "details": "; ".join(details),
    }
