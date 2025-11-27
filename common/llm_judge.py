"""
LLM-as-Judge Verifier - Fallback for complex math/code verification.

Supports both OpenAI and Claude backends:
- OpenAI: GPT-4o (used when called from OpenAI generation pipeline)
- Claude: Sonnet 4.5 (used when called from Claude generation pipeline)

Features:
- LRU cache to avoid re-verifying identical answer pairs
- Configurable cache size (default: 1000 entries)
- Provider auto-selection based on available API keys
"""
import hashlib
import json
import logging
import os
import re
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)


def _escape_latex_backslashes(text: str) -> str:
    """
    Escape LaTeX backslashes in JSON strings to make them valid JSON.

    LaTeX commands like ``\\frac``, ``\\sqrt`` contain backslashes that are
    invalid JSON escape sequences (``\\f`` is form feed, ``\\s`` is invalid).
    This function converts them to escaped backslashes.

    Only escapes backslashes that are NOT valid JSON escapes:
    - Valid JSON escapes: ``\\"``, ``\\\\``, ``\\/``, ``\\b``, ``\\f``, ``\\n``, ``\\r``, ``\\t``, ``\\uXXXX``
    - Invalid (LaTeX): ``\\frac``, ``\\sqrt``, ``\\pi``, ``\\cdot``, etc.

    Args:
        text: JSON string with potential LaTeX backslashes

    Returns:
        String with LaTeX backslashes properly escaped
    """
    # Valid JSON escape sequences (the character after backslash)
    valid_escapes = set('"\\bfnrtu/')

    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char in valid_escapes:
                # Valid JSON escape - keep as is
                result.append(text[i])
            else:
                # Invalid escape (likely LaTeX) - double the backslash
                result.append('\\\\')
                i += 1
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)


def _repair_json(text: str) -> str:
    """
    Attempt to repair malformed JSON that uses Python dict syntax or LaTeX.

    GPT-4o sometimes returns Python dict format instead of valid JSON.
    Claude sometimes returns LaTeX math in reasoning that breaks JSON parsing.

    Repairs performed:
    - LaTeX backslashes (frac, sqrt) escaped to valid JSON
    - Single quotes to double quotes (for keys and string values)
    - Python booleans (True/False) to JSON booleans (true/false)
    - Python None to JSON null

    Args:
        text: Potentially malformed JSON string

    Returns:
        Repaired JSON string (or original if it looks valid)

    Example:
        >>> _repair_json("{'is_correct': True, 'value': None}")
        '{"is_correct": true, "value": null}'
    """
    import re

    # First, escape LaTeX backslashes (common in math reasoning)
    text = _escape_latex_backslashes(text)

    # Skip remaining repairs if it looks like valid JSON already
    if text.strip().startswith('{') and '"' in text:
        return text

    # Replace single quotes with double quotes, but be careful about apostrophes
    # Strategy: replace 'key': patterns and ': 'value' patterns

    # Replace keys: 'key': -> "key":
    text = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', text)

    # Replace string values: : 'value' -> : "value" (handles values with spaces)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)

    # Replace true/false/null Python -> JSON
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    return text


# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-5-20250929",
}


def _hash_inputs(question: str, candidate: str, ground_truth: str, split: str) -> str:
    """Create a stable hash key for caching verification results."""
    content = f"{question.strip()}|{candidate.strip()}|{ground_truth.strip()}|{split}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class LLMJudgeVerifier:
    """Uses LLM to judge if candidate answer matches ground truth."""

    def __init__(
        self,
        provider: Literal["openai", "claude"] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_size: int = 1000
    ):
        """
        Initialize LLM judge.

        Args:
            provider: Which API to use ("openai" or "claude")
            api_key: API key (defaults to env var based on provider)
            model: Model to use (defaults based on provider)
            cache_size: Maximum number of verification results to cache
        """
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.name = "llm_judge"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

        # Initialize the appropriate client
        if provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        else:  # claude
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        logger.info(f"LLM judge initialized: provider={provider}, model={self.model}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def _build_prompt(self, question: str, candidate_answer: str, ground_truth: str, split: str) -> str:
        """Build domain-specific verification prompt."""
        if split == "math":
            domain_instructions = """You are a math teacher grading answers.
Two answers are equivalent if they represent the same mathematical value, even if formatted differently.

Examples of equivalent answers:
- 1/2, 0.5, 50%
- (2024!)^2 / 4048!  and  1/binom(4048, 2024)
- sqrt(2)  and  1.414...
- Monday  and  monday (case insensitive for text answers)"""

        elif split == "code":
            domain_instructions = """Does this code produce CORRECT OUTPUT? (Efficiency doesn't matter)
Answer: equivalent=true if correct, equivalent=false only if you found a BUG."""

        else:  # tool_calling
            domain_instructions = """Did this response accomplish the SAME TASK as the ground truth? (Format doesn't matter)
Answer: equivalent=true if same result achieved, equivalent=false only if WRONG result or task NOT completed."""

        return f'''{domain_instructions}

QUESTION:
{question}

CANDIDATE ANSWER:
{candidate_answer}

GROUND TRUTH:
{ground_truth}

Respond with JSON only:
{{
  "equivalent": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "ONE SENTENCE explaining your decision"
}}'''

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        return response.choices[0].message.content

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API with JSON enforcement via prefill technique."""
        response = await self.client.messages.create(
            model=self.model,
            system="Output valid JSON only. No other text, no markdown, no explanation outside JSON.",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{"}  # Prefill forces JSON start
            ],
            max_tokens=1000,  # Increased for reasoning-first approach
            temperature=0.0,
        )
        # Prepend the '{' we used for prefill
        return "{" + response.content[0].text

    async def verify(
        self,
        question: str,
        candidate_answer: str,
        ground_truth: str,
        split: str = "math",
    ) -> Dict[str, Any]:
        """
        Verify if candidate answer is equivalent to ground truth.

        Args:
            question: Original question
            candidate_answer: Model's answer
            ground_truth: Correct answer
            split: Domain (math, code, tool_calling)

        Returns:
            Dict with is_correct, confidence, explanation
        """
        # Check cache first
        cache_key = _hash_inputs(question, candidate_answer, ground_truth, split)
        if cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key].copy()
            cached["cached"] = True
            logger.debug(f"LLM judge cache hit (key={cache_key[:8]}...)")
            return cached

        self._cache_misses += 1

        prompt = self._build_prompt(question, candidate_answer, ground_truth, split)

        try:
            # Call appropriate API
            if self.provider == "openai":
                response_text = await self._call_openai(prompt)
            else:
                response_text = await self._call_claude(prompt)

            response_text = response_text.strip()

            # Strategy: Try parsing in order of likelihood
            # 1. Full response (LLM often returns clean JSON)
            # 2. Strip markdown code blocks if present
            # 3. Fallback regex (last resort, can fail with nested braces)

            json_text = response_text
            result = None
            first_error = None

            # Try 1: Parse full response directly
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                first_error = e

                # Try 2: Strip markdown code blocks (```json ... ```)
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if code_block_match:
                    json_text = code_block_match.group(1)
                    try:
                        result = json.loads(json_text)
                    except json.JSONDecodeError:
                        pass

                # Try 3: Find JSON that starts at beginning (handles trailing text)
                if result is None and response_text.startswith('{'):
                    # Find matching closing brace by counting
                    depth = 0
                    end_idx = 0
                    for i, char in enumerate(response_text):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end_idx = i + 1
                                break
                    if end_idx > 0:
                        json_text = response_text[:end_idx]
                        try:
                            result = json.loads(json_text)
                        except json.JSONDecodeError:
                            pass

            # If still no result, try repair on best candidate
            if result is None:
                # Attempt to repair malformed JSON (single quotes, Python bools, etc.)
                repaired_text = _repair_json(json_text)
                try:
                    result = json.loads(repaired_text)
                except json.JSONDecodeError as repair_error:
                    # Log detailed diagnostics before re-raising
                    logger.error(
                        f"LLM judge JSON parse failed after repair attempt:\n"
                        f"  Original error: {first_error}\n"
                        f"  Repair error: {repair_error}\n"
                        f"  Raw LLM response ({len(response_text)} chars): {response_text[:500]}{'...' if len(response_text) > 500 else ''}\n"
                        f"  Extracted JSON text: {json_text[:300]}{'...' if len(json_text) > 300 else ''}\n"
                        f"  After repair: {repaired_text[:300]}{'...' if len(repaired_text) > 300 else ''}"
                    )
                    raise

            verification_result = {
                "is_correct": result.get("equivalent", False),
                "confidence": result.get("confidence", 0.0),
                "explanation": f"LLM Judge ({self.model}): {result.get('reasoning', 'No reasoning provided')}",
                "verifier_name": "llm_judge",
            }

            # Cache the result (LRU eviction if full)
            if len(self._cache) >= self._cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = verification_result

            return verification_result

        except Exception as e:
            # Include truncated context for debugging
            question_preview = question[:100] + "..." if len(question) > 100 else question
            candidate_preview = candidate_answer[:100] + "..." if len(candidate_answer) > 100 else candidate_answer
            logger.error(
                f"LLM judge verification failed: {type(e).__name__}: {e}\n"
                f"  Split: {split}\n"
                f"  Question: {question_preview}\n"
                f"  Candidate answer: {candidate_preview}"
            )
            return {
                "is_correct": False,
                "confidence": 0.0,
                "explanation": f"LLM judge error: {type(e).__name__}: {e}",
                "verifier_name": "llm_judge",
            }


# Global singletons (one per provider)
_llm_judge_instances: Dict[str, LLMJudgeVerifier] = {}


def get_llm_judge(
    provider: Literal["openai", "claude"] = "openai",
    api_key: Optional[str] = None
) -> LLMJudgeVerifier:
    """
    Get or create LLM judge singleton for the specified provider.

    Args:
        provider: "openai" (uses GPT-4o) or "claude" (uses Sonnet 4.5)
        api_key: Optional API key override

    Returns:
        LLMJudgeVerifier instance
    """
    global _llm_judge_instances
    if provider not in _llm_judge_instances:
        _llm_judge_instances[provider] = LLMJudgeVerifier(provider=provider, api_key=api_key)
    return _llm_judge_instances[provider]
