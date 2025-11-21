"""
Base classes and interfaces for the verification system.

Provides abstract base classes, data structures, and utilities that all
specific verifiers inherit from.
"""

import time
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable
from contextlib import contextmanager


# ============================================================================
# Exceptions
# ============================================================================

class VerificationError(Exception):
    """Base exception for verification errors."""
    pass


class TimeoutError(VerificationError):
    """Raised when verification exceeds time limit."""
    pass


class ConfigurationError(VerificationError):
    """Raised when verifier configuration is invalid."""
    pass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class VerificationResult:
    """
    Result of a verification operation.

    Attributes:
        is_correct: Boolean indicating if the answer is verified as correct
        confidence: Float in [0.0, 1.0] indicating confidence in the result
        explanation: Human-readable explanation of the verification decision
        metadata: Additional information about the verification process
        execution_time: Time taken to perform verification (seconds)
        verifier_name: Name of the verifier that produced this result
    """
    is_correct: bool
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    verifier_name: str = "unknown"

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "verifier_name": self.verifier_name,
        }

    @classmethod
    def failure(cls, explanation: str, verifier_name: str = "unknown",
                execution_time: float = 0.0) -> "VerificationResult":
        """Factory method for failure results."""
        return cls(
            is_correct=False,
            confidence=0.0,
            explanation=explanation,
            execution_time=execution_time,
            verifier_name=verifier_name,
        )

    @classmethod
    def success(cls, explanation: str, confidence: float = 1.0,
                verifier_name: str = "unknown", execution_time: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None) -> "VerificationResult":
        """Factory method for success results."""
        return cls(
            is_correct=True,
            confidence=confidence,
            explanation=explanation,
            execution_time=execution_time,
            verifier_name=verifier_name,
            metadata=metadata or {},
        )


# ============================================================================
# Timeout Utility
# ============================================================================

@contextmanager
def timeout_context(seconds: float):
    """
    Context manager for enforcing timeouts on synchronous code.

    Usage:
        with timeout_context(5.0):
            # Code that must complete within 5 seconds
            slow_operation()

    Raises:
        TimeoutError: If code block exceeds time limit

    Note: Only works on Unix-like systems. On Windows, falls back to no timeout.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds} second timeout")

    # Check if signal.SIGALRM is available (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds) + 1)  # Round up to nearest second
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows or other platforms without SIGALRM - no timeout enforcement
        yield


# ============================================================================
# Abstract Verifier Base Class
# ============================================================================

class Verifier(ABC):
    """
    Abstract base class for all verifiers.

    Subclasses must implement:
    - _verify_impl(): Core verification logic
    - name property: Unique identifier for the verifier

    The base class provides:
    - Timeout enforcement
    - Execution time tracking
    - Error handling and result wrapping
    - Configuration management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize verifier with optional configuration.

        Args:
            config: Dictionary of configuration parameters specific to this verifier
        """
        self.config = config or {}
        self.timeout = self.config.get("timeout", 10.0)  # Default 10 second timeout

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this verifier (e.g., 'math', 'code', 'tool')."""
        pass

    @abstractmethod
    def _verify_impl(self, question: str, candidate: Dict[str, Any],
                     spec: Dict[str, Any]) -> VerificationResult:
        """
        Core verification implementation (to be overridden by subclasses).

        Args:
            question: The original question/prompt
            candidate: Candidate answer with structure:
                {
                    "text": str,           # Answer text/content
                    "model": str,          # Model that generated it
                    "finish_reason": str,  # Generation completion reason
                    ...                    # Other metadata
                }
            spec: Verification specification with structure:
                {
                    "ground_truth": Any,   # Expected answer
                    "schema": dict,        # (Optional) JSON schema for validation
                    "test_cases": list,    # (Optional) Test cases for code
                    ...                    # Other verifier-specific data
                }

        Returns:
            VerificationResult with verification outcome

        Raises:
            VerificationError: If verification cannot be performed
            TimeoutError: If verification exceeds time limit
        """
        pass

    def verify(self, question: str, candidate: Dict[str, Any],
               spec: Dict[str, Any]) -> VerificationResult:
        """
        Public verification method with timeout and error handling.

        This method wraps _verify_impl() with:
        - Execution time tracking
        - Timeout enforcement
        - Exception handling
        - Result validation

        Args:
            question: The original question/prompt
            candidate: Candidate answer dictionary
            spec: Verification specification dictionary

        Returns:
            VerificationResult with verification outcome
        """
        start_time = time.time()

        try:
            # Enforce timeout using context manager
            with timeout_context(self.timeout):
                result = self._verify_impl(question, candidate, spec)

            # Add execution time and verifier name
            result.execution_time = time.time() - start_time
            result.verifier_name = self.name

            return result

        except TimeoutError as e:
            execution_time = time.time() - start_time
            return VerificationResult.failure(
                explanation=f"Verification timeout after {execution_time:.2f}s: {str(e)}",
                verifier_name=self.name,
                execution_time=execution_time,
            )

        except VerificationError as e:
            execution_time = time.time() - start_time
            return VerificationResult.failure(
                explanation=f"Verification error: {str(e)}",
                verifier_name=self.name,
                execution_time=execution_time,
            )

        except Exception as e:
            # Catch-all for unexpected errors
            execution_time = time.time() - start_time
            return VerificationResult.failure(
                explanation=f"Unexpected error during verification: {type(e).__name__}: {str(e)}",
                verifier_name=self.name,
                execution_time=execution_time,
            )

    def supports_spec(self, spec: Dict[str, Any]) -> bool:
        """
        Check if this verifier can handle the given specification.

        Default implementation always returns True. Override for specific requirements.

        Args:
            spec: Verification specification

        Returns:
            True if this verifier can handle the spec, False otherwise
        """
        return True


# ============================================================================
# Verifier Registry
# ============================================================================

class VerifierRegistry:
    """
    Registry for managing and selecting verifiers.

    Provides factory pattern for creating appropriate verifiers based on
    problem type, dataset split, or explicit selection.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._verifiers: Dict[str, Callable[..., Verifier]] = {}
        self._default_config: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, verifier_class: type,
                 default_config: Optional[Dict[str, Any]] = None):
        """
        Register a verifier class.

        Args:
            name: Unique name for this verifier type
            verifier_class: Verifier class (not instance)
            default_config: Default configuration for this verifier
        """
        if not issubclass(verifier_class, Verifier):
            raise ValueError(f"{verifier_class} must be a subclass of Verifier")

        self._verifiers[name] = verifier_class
        if default_config:
            self._default_config[name] = default_config

    def create(self, name: str, config: Optional[Dict[str, Any]] = None) -> Verifier:
        """
        Create a verifier instance by name.

        Args:
            name: Name of registered verifier
            config: Optional configuration (merged with default config)

        Returns:
            Verifier instance

        Raises:
            ConfigurationError: If verifier name not registered
        """
        if name not in self._verifiers:
            available = ", ".join(self._verifiers.keys())
            raise ConfigurationError(
                f"Unknown verifier '{name}'. Available: {available}"
            )

        # Merge default config with provided config
        merged_config = self._default_config.get(name, {}).copy()
        if config:
            merged_config.update(config)

        return self._verifiers[name](merged_config)

    def get_verifier_for_split(self, split: str,
                                 config: Optional[Dict[str, Any]] = None) -> Verifier:
        """
        Select appropriate verifier based on dataset split name.

        Args:
            split: Dataset split name (e.g., 'math_500', 'python_code', 'tool_use')
            config: Optional configuration override

        Returns:
            Verifier instance

        Examples:
            'math_500' -> MathVerifier
            'gsm8k' -> MathVerifier
            'humaneval' -> CodeVerifier
            'mbpp' -> CodeVerifier
            'tool_use' -> ToolVerifier
        """
        split_lower = split.lower()

        # Math keywords
        if any(kw in split_lower for kw in ['math', 'gsm', 'aime', 'amc', 'algebra']):
            return self.create('math', config)

        # Code keywords
        if any(kw in split_lower for kw in ['code', 'python', 'humaneval', 'mbpp', 'codex']):
            return self.create('code', config)

        # Tool keywords
        if any(kw in split_lower for kw in ['tool', 'api', 'function']):
            return self.create('tool', config)

        # Default to math if unclear
        return self.create('math', config)

    def list_verifiers(self) -> List[str]:
        """Return list of registered verifier names."""
        return list(self._verifiers.keys())


# Global registry instance
registry = VerifierRegistry()
