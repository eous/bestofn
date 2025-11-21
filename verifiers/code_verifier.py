"""
Code verification using Docker-based sandboxed execution.

Features:
- Multi-language support: Python, JavaScript/TypeScript, Bash, SQL
- Docker isolation (no exec/eval in main process)
- Test case execution with assertions
- Resource limits (CPU, memory, timeout)
- Secure I/O handling

Security:
- No code execution in main process
- Containerized with no network access
- Read-only filesystem (except /tmp)
- Resource limits prevent DoS
"""

import re
import logging
from typing import Dict, Any, List, Optional

from .base import Verifier, VerificationResult, VerificationError
from .docker_sandbox import DockerSandbox, ExecutionResult

logger = logging.getLogger(__name__)


# ============================================================================
# Code Verifier
# ============================================================================

class CodeVerifier(Verifier):
    """
    Verifies code answers using Docker-based sandboxed execution.

    Supports:
    - Python: Execute code and compare output or run test cases
    - JavaScript/TypeScript: Execute with Node.js
    - Bash/Shell: Execute shell scripts with restricted commands
    - SQL: Execute queries against in-memory SQLite database

    Verification modes:
    1. Output matching: Compare stdout against expected output
    2. Test cases: Run multiple test cases with input/output pairs
    3. Test suites: Execute unit tests (pytest, jest)
    """

    @property
    def name(self) -> str:
        return "code"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize code verifier with Docker sandbox.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Initialize Docker sandbox (deferred to first use to avoid startup cost)
        self._sandbox: Optional[DockerSandbox] = None

    @property
    def sandbox(self) -> DockerSandbox:
        """Lazy initialization of Docker sandbox."""
        if self._sandbox is None:
            self._sandbox = DockerSandbox(self.config)
        return self._sandbox

    def _verify_impl(self, question: str, candidate: Dict[str, Any],
                     spec: Dict[str, Any]) -> VerificationResult:
        """
        Verify code answer.

        Args:
            question: Original coding question
            candidate: Candidate answer with 'text' field containing code
            spec: Specification with:
                - ground_truth: Expected output (optional)
                - test_cases: List of test case dicts (optional)
                - language: Programming language (optional, auto-detected)
                - test_framework: Test framework to use (optional)

        Returns:
            VerificationResult
        """
        # Extract answer text
        answer_text = candidate.get("text", "")
        if not answer_text:
            return VerificationResult.failure(
                explanation="No answer text in candidate",
                verifier_name=self.name,
            )

        # Detect language
        language = spec.get("language") or self._detect_language(answer_text)
        if not language:
            return VerificationResult.failure(
                explanation="Could not detect programming language",
                verifier_name=self.name,
            )

        # Extract code from markdown if present
        code = self._extract_code(answer_text, language)
        if not code:
            return VerificationResult.failure(
                explanation="Could not extract code from answer",
                verifier_name=self.name,
            )

        # Check if test cases are provided
        test_cases = spec.get("test_cases")
        if test_cases and isinstance(test_cases, list):
            return self._verify_with_test_cases(code, language, test_cases)

        # Check if ground truth output is provided
        ground_truth = spec.get("ground_truth")
        if ground_truth is not None:
            return self._verify_with_output(code, language, ground_truth)

        # No verification spec provided - just check if code runs
        return self._verify_execution(code, language)

    def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect programming language from code text.

        Args:
            text: Code text

        Returns:
            Language name or None
        """
        # Check for markdown code fence with language
        fence_match = re.search(r'```(\w+)', text)
        if fence_match:
            lang = fence_match.group(1).lower()
            if lang in ['python', 'py']:
                return 'python'
            elif lang in ['javascript', 'js', 'typescript', 'ts', 'node']:
                return 'javascript'
            elif lang in ['bash', 'sh', 'shell']:
                return 'bash'
            elif lang == 'sql':
                return 'sql'

        # Heuristic detection based on syntax
        if 'def ' in text or 'import ' in text or 'print(' in text:
            return 'python'
        elif 'function ' in text or 'const ' in text or 'console.log(' in text:
            return 'javascript'
        elif 'SELECT ' in text.upper() or 'CREATE TABLE' in text.upper():
            return 'sql'
        elif '#!/bin/bash' in text or 'echo ' in text:
            return 'bash'

        # Default to Python if unclear
        return 'python'

    def _extract_code(self, text: str, language: str) -> Optional[str]:
        """
        Extract code from text, handling markdown code blocks.

        Args:
            text: Text containing code
            language: Programming language

        Returns:
            Extracted code or None
        """
        # Try to extract from markdown code block
        lang_pattern = language if language != 'javascript' else r'(?:javascript|js|typescript|ts|node)'
        code_match = re.search(
            rf'```{lang_pattern}\s*\n(.*?)```',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if code_match:
            return code_match.group(1).strip()

        # Try generic code block
        code_match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Check if whole text looks like code
        if self._looks_like_code(text, language):
            return text.strip()

        return None

    def _looks_like_code(self, text: str, language: str) -> bool:
        """
        Heuristic to check if text looks like code.

        Args:
            text: Text to check
            language: Programming language

        Returns:
            True if text appears to be code
        """
        indicators = {
            'python': ['def ', 'class ', 'import ', 'print(', 'return ', 'if __name__'],
            'javascript': ['function ', 'const ', 'let ', 'var ', 'console.log(', '=>'],
            'bash': ['#!/bin/bash', 'echo ', 'if [', 'for ', 'while '],
            'sql': ['SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ', 'CREATE '],
        }

        lang_indicators = indicators.get(language, [])
        return any(indicator in text for indicator in lang_indicators)

    def _verify_with_output(self, code: str, language: str,
                           ground_truth: str) -> VerificationResult:
        """
        Verify code by comparing output against ground truth.

        Args:
            code: Code to execute
            language: Programming language
            ground_truth: Expected output

        Returns:
            VerificationResult
        """
        try:
            # Execute code
            result = self.sandbox.execute(code, language)

            if not result.succeeded:
                return VerificationResult.failure(
                    explanation=f"Code execution failed: {result.stderr or result.error}",
                    verifier_name=self.name,
                    metadata={"exit_code": result.exit_code, "stderr": result.stderr},
                )

            # Compare output
            expected_output = str(ground_truth).strip()
            actual_output = result.stdout.strip()

            # Try exact match first
            if actual_output == expected_output:
                return VerificationResult.success(
                    explanation=f"Output matches exactly: {actual_output}",
                    confidence=1.0,
                    verifier_name=self.name,
                    metadata={
                        "method": "output_exact",
                        "output": actual_output,
                        "execution_time": result.execution_time,
                    },
                )

            # Try substring match (less strict)
            if expected_output in actual_output:
                return VerificationResult.success(
                    explanation=f"Output contains expected text: {expected_output}",
                    confidence=0.85,
                    verifier_name=self.name,
                    metadata={
                        "method": "output_substring",
                        "output": actual_output,
                        "execution_time": result.execution_time,
                    },
                )

            # Try numeric comparison if both are numbers
            try:
                expected_num = float(expected_output)
                actual_num = float(actual_output)
                if abs(expected_num - actual_num) < 1e-6:
                    return VerificationResult.success(
                        explanation=f"Numeric output matches: {actual_num} ≈ {expected_num}",
                        confidence=0.95,
                        verifier_name=self.name,
                        metadata={
                            "method": "output_numeric",
                            "output": actual_num,
                            "execution_time": result.execution_time,
                        },
                    )
            except ValueError:
                pass  # Not numeric

            # Output doesn't match
            return VerificationResult.failure(
                explanation=f"Output mismatch.\nExpected: {expected_output}\nActual: {actual_output}",
                verifier_name=self.name,
                metadata={"expected": expected_output, "actual": actual_output},
            )

        except Exception as e:
            return VerificationResult.failure(
                explanation=f"Verification error: {e}",
                verifier_name=self.name,
            )

    def _verify_with_test_cases(self, code: str, language: str,
                                test_cases: List[Dict[str, Any]]) -> VerificationResult:
        """
        Verify code using multiple test cases.

        Args:
            code: Code to execute
            language: Programming language
            test_cases: List of test case dicts with 'input' and 'expected_output'

        Returns:
            VerificationResult
        """
        if not test_cases:
            return VerificationResult.failure(
                explanation="No test cases provided",
                verifier_name=self.name,
            )

        passed = 0
        failed = 0
        failures = []

        for i, test_case in enumerate(test_cases):
            stdin = str(test_case.get('input', ''))
            expected = str(test_case.get('expected_output', '')).strip()

            try:
                # Execute code with this test input
                result = self.sandbox.execute(code, language, stdin=stdin)

                if not result.succeeded:
                    failed += 1
                    failures.append({
                        "test_case": i + 1,
                        "reason": "execution_failed",
                        "error": result.stderr or result.error,
                    })
                    continue

                actual = result.stdout.strip()

                # Compare output
                if actual == expected or expected in actual:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "test_case": i + 1,
                        "reason": "output_mismatch",
                        "expected": expected,
                        "actual": actual,
                    })

            except Exception as e:
                failed += 1
                failures.append({
                    "test_case": i + 1,
                    "reason": "exception",
                    "error": str(e),
                })

        # Calculate confidence based on pass rate
        total = len(test_cases)
        pass_rate = passed / total if total > 0 else 0.0

        if passed == total:
            return VerificationResult.success(
                explanation=f"All {total} test cases passed",
                confidence=1.0,
                verifier_name=self.name,
                metadata={
                    "method": "test_cases",
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                },
            )
        elif passed > 0:
            return VerificationResult(
                is_correct=False,
                confidence=pass_rate,
                explanation=f"Passed {passed}/{total} test cases. Failures: {failures[:3]}",
                verifier_name=self.name,
                metadata={
                    "method": "test_cases",
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "failures": failures,
                },
            )
        else:
            return VerificationResult.failure(
                explanation=f"All {total} test cases failed. First failure: {failures[0] if failures else 'unknown'}",
                verifier_name=self.name,
                metadata={
                    "method": "test_cases",
                    "passed": 0,
                    "failed": failed,
                    "total": total,
                    "failures": failures,
                },
            )

    def _verify_execution(self, code: str, language: str) -> VerificationResult:
        """
        Verify that code executes without errors (no output verification).

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            VerificationResult
        """
        try:
            result = self.sandbox.execute(code, language)

            if result.succeeded:
                return VerificationResult.success(
                    explanation=f"Code executed successfully. Output: {result.stdout[:100]}",
                    confidence=0.5,  # Low confidence since we're not verifying correctness
                    verifier_name=self.name,
                    metadata={
                        "method": "execution_only",
                        "output": result.stdout,
                        "execution_time": result.execution_time,
                    },
                )
            else:
                return VerificationResult.failure(
                    explanation=f"Code execution failed: {result.stderr or result.error}",
                    verifier_name=self.name,
                    metadata={
                        "exit_code": result.exit_code,
                        "stderr": result.stderr,
                        "error": result.error,
                    },
                )

        except Exception as e:
            return VerificationResult.failure(
                explanation=f"Execution error: {e}",
                verifier_name=self.name,
            )

    def cleanup(self):
        """Clean up Docker sandbox resources."""
        if self._sandbox:
            self._sandbox.cleanup()
            self._sandbox = None

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
