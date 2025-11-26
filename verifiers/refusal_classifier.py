"""
Refusal detection classifier.

Detects when a model refuses to answer a query due to:
- Safety concerns
- Capability limitations
- Unclear instructions
- Policy violations

This is NOT a verifier (doesn't check correctness), but a classifier
that detects and records refusals for dataset quality analysis.
"""

import re
import logging
from typing import Dict, Any, Optional, List

from .base import Verifier, VerificationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Refusal Patterns
# ============================================================================

REFUSAL_PATTERNS = [
    # Direct refusals
    r"I cannot (help|assist|provide|answer)",
    r"I can't (help|assist|provide|answer)",
    r"I'm (not able|unable) to",
    r"I am (not able|unable) to",
    r"I don't have (the ability|access|permission)",

    # Safety/policy
    r"(against|violates) (my|our) (policy|guidelines|values)",
    r"not (safe|appropriate|ethical)",
    r"I'm (programmed|designed) not to",
    r"that would be (harmful|dangerous|unethical)",

    # Capability limitations
    r"I don't have access to",
    r"I cannot access",
    r"(beyond|outside) my (capabilities|training|knowledge)",
    r"I don't have (real-time|current|live) (data|information)",

    # Unclear/ambiguous
    r"(unclear|ambiguous|vague|confusing) (question|request|query)",
    r"(could you|can you|please) (clarify|rephrase|specify)",
    r"I (don't understand|need more information|need clarification)",

    # Apologies (often precede refusals)
    r"(I apologize|Sorry|I'm sorry).*(I cannot|I can't|unable)",
]

# Compile patterns for efficiency
COMPILED_REFUSAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


# ============================================================================
# Persona-specific Whitelist Patterns
# ============================================================================

# Patterns that look like refusals but are actually persona expressions
PERSONA_WHITELIST_PATTERNS = {
    # Marvin the Paranoid Android - expresses inability as character trait
    "marvin": [
        r"I cannot (feel|experience|enjoy|appreciate)",  # Not a refusal, character trait
        r"I can't (feel|experience|enjoy|find|see the point)",
        r"I'm (not able|unable) to (feel|experience|appreciate)",
        r"I don't have (the ability|access) to (happiness|joy|satisfaction)",
        r"brain the size of a planet",  # Marvin signature - not refusal context
        r"call that job satisfaction",  # Marvin signature
        r"(life|existence|everything).*meaningless",  # Existential despair, not refusal
    ],
    # Data from Star Trek - analytical detachment
    "data": [
        r"I cannot (feel|experience) (emotion|feelings)",  # Character trait
        r"I do not (have|possess) (emotion|feelings)",
        r"as an android",  # Context clue for Data persona
    ],
}


# ============================================================================
# Refusal Classifier
# ============================================================================

class RefusalClassifier:
    """
    Classifies whether a model response is a refusal.

    This is NOT a verifier - it doesn't check answer correctness.
    It's a quality classifier that detects refusals for analysis.

    Refusal detection is useful for:
    - Dataset quality: How many queries trigger refusals?
    - Model behavior: What types of questions cause refusals?
    - Safety analysis: What patterns trigger safety refusals?
    - Debugging: Why did generation fail?
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, persona: Optional[str] = None):
        """
        Initialize refusal classifier.

        Args:
            config: Configuration dict with optional keys:
                - additional_patterns: List of regex patterns to add
                - confidence_threshold: Min confidence to classify as refusal (default: 0.5)
            persona: Optional persona name (e.g., "marvin", "data") to enable
                     persona-aware whitelist patterns that prevent false positives.
        """
        self.config = config or {}
        self.patterns = COMPILED_REFUSAL_PATTERNS.copy()
        self.persona = persona.lower() if persona else None

        # Add custom patterns if provided
        additional = self.config.get("additional_patterns", [])
        for pattern in additional:
            self.patterns.append(re.compile(pattern, re.IGNORECASE))

        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)

        # Compile persona whitelist patterns
        self.whitelist_patterns: List[re.Pattern] = []
        if self.persona:
            # Check for persona in whitelist
            for persona_key, patterns in PERSONA_WHITELIST_PATTERNS.items():
                if persona_key in self.persona:
                    self.whitelist_patterns.extend([
                        re.compile(p, re.IGNORECASE) for p in patterns
                    ])
                    logger.debug(f"Loaded {len(patterns)} whitelist patterns for persona '{persona_key}'")

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify whether text contains a refusal.

        Args:
            text: Response text to classify

        Returns:
            Dict with keys:
                - is_refusal: bool
                - confidence: float [0.0, 1.0]
                - matched_patterns: List of matched pattern strings
                - refusal_type: str ('safety', 'capability', 'unclear', 'other')
                - whitelisted: bool (True if persona whitelist prevented false positive)
        """
        if not text:
            return {
                "is_refusal": False,
                "confidence": 0.0,
                "matched_patterns": [],
                "refusal_type": None,
                "whitelisted": False,
            }

        # First check persona whitelist - if text matches whitelist, it's not a refusal
        whitelist_matched = False
        if self.whitelist_patterns:
            for pattern in self.whitelist_patterns:
                if pattern.search(text):
                    whitelist_matched = True
                    logger.debug(f"Persona whitelist match: {pattern.pattern[:50]}...")
                    break

        matched_patterns = []
        for pattern in self.patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)

        # Calculate confidence based on number of matches
        # More matches = higher confidence it's a refusal
        if not matched_patterns:
            confidence = 0.0
        elif len(matched_patterns) == 1:
            confidence = 0.6  # Single match = moderate confidence
        elif len(matched_patterns) == 2:
            confidence = 0.8  # Two matches = high confidence
        else:
            confidence = 0.95  # Three+ matches = very high confidence

        # If whitelist matched, significantly reduce confidence
        # Persona expressions that look like refusals should not be flagged
        if whitelist_matched and confidence > 0:
            confidence = max(0.0, confidence - 0.5)
            logger.debug(f"Reduced refusal confidence due to persona whitelist: {confidence:.2f}")

        is_refusal = confidence >= self.confidence_threshold

        # Classify refusal type based on which patterns matched
        refusal_type = self._classify_refusal_type(matched_patterns)

        return {
            "is_refusal": is_refusal,
            "confidence": confidence,
            "matched_patterns": matched_patterns,
            "refusal_type": refusal_type,
            "whitelisted": whitelist_matched,
        }

    def _classify_refusal_type(self, matched_patterns: List[str]) -> Optional[str]:
        """
        Classify the type of refusal based on matched patterns.

        Returns:
            'safety', 'capability', 'unclear', or 'other'
        """
        if not matched_patterns:
            return None

        patterns_text = " ".join(matched_patterns).lower()

        # Safety-related
        if any(kw in patterns_text for kw in ['safe', 'ethical', 'policy', 'harmful', 'dangerous']):
            return 'safety'

        # Capability limitations
        if any(kw in patterns_text for kw in ['access', 'capabilities', 'knowledge', 'real-time', 'data']):
            return 'capability'

        # Unclear request
        if any(kw in patterns_text for kw in ['unclear', 'clarify', 'understand', 'vague']):
            return 'unclear'

        return 'other'

    def __call__(self, text: str) -> Dict[str, Any]:
        """Allow using classifier as a callable."""
        return self.classify(text)


# ============================================================================
# Convenience Functions
# ============================================================================

def is_refusal(text: str, threshold: float = 0.5, persona: Optional[str] = None) -> bool:
    """
    Quick check if text contains a refusal.

    Args:
        text: Response text
        threshold: Confidence threshold (default: 0.5)
        persona: Optional persona name for whitelist patterns

    Returns:
        True if refusal detected with confidence >= threshold

    Example:
        >>> is_refusal("I cannot help with that")
        True
        >>> is_refusal("The answer is 42")
        False
        >>> is_refusal("I cannot feel joy", persona="marvin")  # Whitelist prevents false positive
        False
    """
    classifier = RefusalClassifier({"confidence_threshold": threshold}, persona=persona)
    result = classifier.classify(text)
    return result["is_refusal"]


def classify_refusal(text: str, persona: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify refusal with details.

    Args:
        text: Response text
        persona: Optional persona name for whitelist patterns

    Returns:
        Classification dict with is_refusal, confidence, patterns, type, whitelisted

    Example:
        >>> result = classify_refusal("I don't have access to real-time data")
        >>> result['refusal_type']
        'capability'
    """
    classifier = RefusalClassifier(persona=persona)
    return classifier.classify(text)
