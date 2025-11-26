"""
Enhanced Verification System for Best-of-N Candidate Selection

This module provides secure, accurate, and extensible verifiers for mathematical,
code, and tool-based answer validation.

Security Features:
- Docker-isolated code execution (no exec/eval in main process)
- Resource limits (CPU, memory, timeout)
- Input sanitization and validation

Accuracy Features:
- Symbolic math equivalence (SymPy)
- Unit-aware comparisons
- Multi-format answer parsing
- Schema validation for structured outputs

Performance:
- <2 second verification for 95%+ of cases
- Container pooling for fast startup
- Async-ready architecture
"""

from .base import (
    Verifier,
    VerificationResult,
    VerifierRegistry,
    VerificationError,
    TimeoutError,
    registry,
)
from .config import VerifierConfig, load_config, DEFAULT_CONFIG
from .math_verifier import MathVerifier
from .code_verifier import CodeVerifier
from .tool_verifier import ToolVerifier, OpenAPIToolVerifier
from .refusal_classifier import RefusalClassifier, is_refusal, classify_refusal
from .persona_verifier import PersonaVerifier

__all__ = [
    # Base classes
    "Verifier",
    "VerificationResult",
    "VerifierRegistry",
    "VerificationError",
    "TimeoutError",
    # Configuration
    "VerifierConfig",
    "load_config",
    "DEFAULT_CONFIG",
    # Verifiers
    "MathVerifier",
    "CodeVerifier",
    "ToolVerifier",
    "OpenAPIToolVerifier",
    "PersonaVerifier",
    # Refusal detection
    "RefusalClassifier",
    "is_refusal",
    "classify_refusal",
    # Global registry
    "registry",
    "get_verifier",
    "get_verifier_for_split",
]

__version__ = "0.2.0"

# ============================================================================
# Register Verifiers with Global Registry
# ============================================================================

# Register math verifier
registry.register(
    name="math",
    verifier_class=MathVerifier,
    default_config=DEFAULT_CONFIG.get("math", {})
)

# Register code verifier
registry.register(
    name="code",
    verifier_class=CodeVerifier,
    default_config=DEFAULT_CONFIG.get("code", {})
)

# Register tool verifier
registry.register(
    name="tool",
    verifier_class=ToolVerifier,
    default_config=DEFAULT_CONFIG.get("tool", {})
)

# Register OpenAPI tool verifier
registry.register(
    name="tool_openapi",
    verifier_class=OpenAPIToolVerifier,
    default_config=DEFAULT_CONFIG.get("tool", {})
)

# Note: PersonaVerifier is NOT auto-registered (requires explicit config)
# Use directly: PersonaVerifier(config={'persona_file': 'personas/marvin.txt', ...})


# ============================================================================
# Convenience Functions
# ============================================================================

def get_verifier(name: str, config: dict = None) -> Verifier:
    """
    Get a verifier instance by name.

    Args:
        name: Verifier name ('math', 'code', 'tool', 'tool_openapi')
        config: Optional configuration dictionary

    Returns:
        Verifier instance

    Example:
        >>> verifier = get_verifier('math', {'timeout': 5.0})
        >>> result = verifier.verify(question, candidate, spec)
    """
    return registry.create(name, config)


def get_verifier_for_split(split: str, config: dict = None) -> Verifier:
    """
    Get appropriate verifier based on dataset split name.

    Args:
        split: Dataset split name (e.g., 'math_500', 'python_code', 'tool_use')
        config: Optional configuration override

    Returns:
        Verifier instance

    Example:
        >>> verifier = get_verifier_for_split('gsm8k')
        >>> # Returns MathVerifier
    """
    return registry.get_verifier_for_split(split, config)
