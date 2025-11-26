"""
Persona adherence verifier using LLM-as-judge.

Uses a judge model to evaluate how well responses match a target persona.
Useful for personality transfer experiments and persona consistency analysis.

Supports both OpenAI and Claude as judge providers.
"""

import logging
from typing import Dict, Any, Optional, Literal

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import Verifier, VerificationResult

logger = logging.getLogger(__name__)

# Default models for each provider
DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-5-mini",
    "claude": "claude-sonnet-4-5-20250929",
}


# ============================================================================
# Persona Verifier
# ============================================================================

class PersonaVerifier(Verifier):
    """
    Verifies persona adherence using an LLM judge.

    Uses a separate judge model to score how well a response matches
    the target persona on a scale of 0-5.

    Supports both OpenAI and Claude as judge providers.

    Configuration:
        - provider: 'openai' or 'claude' (default: 'openai')
        - judge_model: Model to use as judge (default: gpt-5-mini for OpenAI, claude-sonnet-4-5 for Claude)
        - judge_api_key: API key for judge model (default: OPENAI_API_KEY or ANTHROPIC_API_KEY env)
        - judge_base_url: Base URL for judge API (optional, uses provider default)
        - persona_file: Path to persona description file
        - persona_text: Or inline persona text
        - min_score: Minimum score to consider verified (default: 3.0)

    Example:
        # OpenAI provider (default)
        config = {
            'persona_file': 'personas/marvin.txt',
            'min_score': 3.0,
        }
        verifier = PersonaVerifier(config)

        # Claude provider
        config = {
            'provider': 'claude',
            'persona_file': 'personas/marvin.txt',
        }
        verifier = PersonaVerifier(config)
    """

    @property
    def name(self) -> str:
        return "persona"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize persona verifier.

        Args:
            config: Configuration with judge model settings and persona
        """
        super().__init__(config)
        import os

        # Get provider (default to OpenAI for backward compatibility)
        self.provider = self.config.get('provider', 'openai')
        if self.provider not in ('openai', 'claude'):
            raise ValueError(f"Invalid provider: {self.provider}. Must be 'openai' or 'claude'")

        # Check SDK availability
        if self.provider == 'openai' and not OPENAI_AVAILABLE:
            raise ImportError("openai library required for OpenAI provider. pip install openai")
        if self.provider == 'claude' and not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library required for Claude provider. pip install anthropic")

        # Get model (use provider-specific default if not specified)
        self.judge_model = self.config.get('judge_model', DEFAULT_JUDGE_MODELS[self.provider])
        self.min_score = self.config.get('min_score', 3.0)

        # Load persona
        persona_file = self.config.get('persona_file')
        persona_text = self.config.get('persona_text')

        if persona_file:
            with open(persona_file, 'r') as f:
                self.persona = f.read().strip()
        elif persona_text:
            self.persona = persona_text
        else:
            raise ValueError("PersonaVerifier requires 'persona_file' or 'persona_text' in config")

        # Initialize judge client based on provider
        if self.provider == 'openai':
            api_key = self.config.get('judge_api_key') or os.getenv('OPENAI_API_KEY', 'dummy')
            base_url = self.config.get('judge_base_url') or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            org_id = self.config.get('judge_org_id') or os.getenv('OPENAI_ORG_ID')

            client_kwargs = {'api_key': api_key, 'base_url': base_url}
            if org_id:
                client_kwargs['organization'] = org_id

            self.judge_client = OpenAI(**client_kwargs)
        else:  # claude
            api_key = self.config.get('judge_api_key') or os.getenv('ANTHROPIC_API_KEY')
            base_url = self.config.get('judge_base_url') or os.getenv('ANTHROPIC_BASE_URL')

            client_kwargs = {'api_key': api_key}
            if base_url:
                client_kwargs['base_url'] = base_url

            self.judge_client = Anthropic(**client_kwargs)

    def _verify_impl(self, question: str, candidate: Dict[str, Any],
                     spec: Dict[str, Any]) -> VerificationResult:
        """
        Verify persona adherence using judge model.

        Args:
            question: Original question
            candidate: Candidate with 'text' field containing response
            spec: Spec (unused for persona verification)

        Returns:
            VerificationResult with persona adherence score
        """
        answer = candidate.get('text', '')
        if not answer:
            return VerificationResult.failure(
                explanation="No answer text to evaluate",
                verifier_name=self.name,
            )

        # Build judge prompt
        judge_prompt = f"""You are evaluating how well a response matches a target persona.

TARGET PERSONA:
{self.persona}

QUESTION:
{question}

RESPONSE TO EVALUATE:
{answer}

Score how well the response embodies the target persona on a scale of 0-5:
- 0: Completely wrong persona (contradicts character)
- 1: Minimal persona (generic response, no character)
- 2: Slight persona hints (occasional markers)
- 3: Moderate persona (clear character, some signature elements)
- 4: Strong persona (multiple signature elements, consistent voice)
- 5: Perfect persona embodiment (natural, distinctive, fully in character)

Consider:
- Signature phrases and language patterns
- Emotional tone and attitude
- Consistency with persona description
- Natural integration (not forced)

Provide your evaluation in this format:
<score>X</score>
<reasoning>Brief explanation of score</reasoning>

Score:"""

        try:
            # Call judge model (provider-specific API)
            if self.provider == 'openai':
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,  # Deterministic judging
                    max_tokens=500,
                )
                judge_output = response.choices[0].message.content
            else:  # claude
                response = self.judge_client.messages.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=500,
                )
                judge_output = response.content[0].text

            # Parse score
            import re
            score_match = re.search(r'<score>(\d+(?:\.\d+)?)</score>', judge_output)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', judge_output, re.DOTALL)

            if not score_match:
                # Fallback: try to find any number 0-5
                number_match = re.search(r'\b([0-5](?:\.\d+)?)\b', judge_output)
                if number_match:
                    score = float(number_match.group(1))
                    reasoning = judge_output[:200]  # First 200 chars
                else:
                    return VerificationResult.failure(
                        explanation=f"Could not parse score from judge response: {judge_output[:100]}",
                        verifier_name=self.name,
                    )
            else:
                score = float(score_match.group(1))
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

            # Normalize score to 0-1 confidence
            confidence = score / 5.0

            # Check if meets minimum threshold
            is_correct = score >= self.min_score

            return VerificationResult(
                is_correct=is_correct,
                confidence=confidence,
                explanation=f"Persona score: {score}/5. {reasoning[:200]}",
                verifier_name=self.name,
                metadata={
                    "persona_score": score,
                    "judge_reasoning": reasoning,
                    "judge_model": self.judge_model,
                    "min_score": self.min_score,
                },
            )

        except Exception as e:
            return VerificationResult.failure(
                explanation=f"Judge model error: {e}",
                verifier_name=self.name,
            )


# ============================================================================
# Async Version (for concurrent judging)
# ============================================================================

class AsyncPersonaVerifier(PersonaVerifier):
    """
    Async version of PersonaVerifier for parallel evaluation.

    Use this when evaluating many responses concurrently.
    Supports both OpenAI and Claude providers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize async persona verifier."""
        # Call grandparent __init__ (Verifier), not parent (PersonaVerifier)
        # to avoid creating sync client
        import os
        from .base import Verifier
        Verifier.__init__(self, config)

        # Get provider (default to OpenAI for backward compatibility)
        self.provider = self.config.get('provider', 'openai')
        if self.provider not in ('openai', 'claude'):
            raise ValueError(f"Invalid provider: {self.provider}. Must be 'openai' or 'claude'")

        # Check SDK availability
        if self.provider == 'openai' and not OPENAI_AVAILABLE:
            raise ImportError("openai library required for OpenAI provider. pip install openai")
        if self.provider == 'claude' and not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library required for Claude provider. pip install anthropic")

        # Get model (use provider-specific default if not specified)
        self.judge_model = self.config.get('judge_model', DEFAULT_JUDGE_MODELS[self.provider])
        self.min_score = self.config.get('min_score', 3.0)

        # Load persona (same as parent)
        persona_file = self.config.get('persona_file')
        persona_text = self.config.get('persona_text')

        if persona_file:
            with open(persona_file, 'r') as f:
                self.persona = f.read().strip()
        elif persona_text:
            self.persona = persona_text
        else:
            raise ValueError("PersonaVerifier requires 'persona_file' or 'persona_text' in config")

        # Initialize async client based on provider
        if self.provider == 'openai':
            api_key = self.config.get('judge_api_key') or os.getenv('OPENAI_API_KEY', 'dummy')
            base_url = self.config.get('judge_base_url') or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            org_id = self.config.get('judge_org_id') or os.getenv('OPENAI_ORG_ID')

            client_kwargs = {'api_key': api_key, 'base_url': base_url}
            if org_id:
                client_kwargs['organization'] = org_id

            self.judge_client = AsyncOpenAI(**client_kwargs)
        else:  # claude
            api_key = self.config.get('judge_api_key') or os.getenv('ANTHROPIC_API_KEY')
            base_url = self.config.get('judge_base_url') or os.getenv('ANTHROPIC_BASE_URL')

            client_kwargs = {'api_key': api_key}
            if base_url:
                client_kwargs['base_url'] = base_url

            self.judge_client = AsyncAnthropic(**client_kwargs)

    async def verify_async(self, question: str, candidate: Dict[str, Any],
                          spec: Dict[str, Any]) -> VerificationResult:
        """
        Async version of verify for concurrent evaluation.

        Use with asyncio.gather() to evaluate multiple responses in parallel.
        """
        # Use same logic as parent but with async client
        # (Implementation would mirror _verify_impl but with await)
        return self._verify_impl(question, candidate, spec)
