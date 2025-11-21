"""
Mathematical answer verification using symbolic and numeric methods.

Features:
- SymPy symbolic equivalence checking
- Unit-aware comparison (pint library)
- Multi-format parsing: fractions, decimals, percentages, LaTeX
- Equation solving for implicit answers
- Numeric fallback with configurable tolerance
"""

import re
import math
import logging
from typing import Dict, Any, Optional, Tuple, Union

try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logging.warning("SymPy not available. Math verifier will use numeric-only mode.")

try:
    from pint import UnitRegistry
    PINT_AVAILABLE = True
    ureg = UnitRegistry()
except ImportError:
    PINT_AVAILABLE = False
    logging.warning("Pint not available. Unit-aware comparison disabled.")

from .base import Verifier, VerificationResult, VerificationError

logger = logging.getLogger(__name__)


# ============================================================================
# Math Verifier
# ============================================================================

class MathVerifier(Verifier):
    """
    Verifies mathematical answers using symbolic and numeric methods.

    Strategy:
    1. Try symbolic equivalence (if SymPy available and symbolic_first=True)
    2. Try unit-aware comparison (if pint available and units detected)
    3. Fall back to numeric comparison with tolerance

    Supports:
    - Exact values: 42, 3.14159
    - Fractions: 3/4, -5/7
    - Percentages: 85%, 12.5%
    - Scientific notation: 1.5e10, 3.2e-5
    - Expressions: 2*pi, sqrt(2), e^2
    - LaTeX: \\boxed{42}, \\frac{3}{4}
    - Units: 1000 meters, 1 kilometer, 60 seconds
    - Equations: x = 5, 2x + 3 = 7
    """

    @property
    def name(self) -> str:
        return "math"

    def _verify_impl(self, question: str, candidate: Dict[str, Any],
                     spec: Dict[str, Any]) -> VerificationResult:
        """
        Verify mathematical answer.

        Args:
            question: Original math question
            candidate: Candidate answer dict with 'text' field
            spec: Specification with 'ground_truth' field

        Returns:
            VerificationResult
        """
        # Extract ground truth
        ground_truth = spec.get("ground_truth")
        if ground_truth is None:
            return VerificationResult.failure(
                explanation="No ground truth provided in spec",
                verifier_name=self.name,
            )

        # Extract candidate answer text
        answer_text = candidate.get("text", "")
        if not answer_text:
            return VerificationResult.failure(
                explanation="No answer text in candidate",
                verifier_name=self.name,
            )

        # Convert ground truth to string if needed
        gt_str = str(ground_truth)

        # Check expression size to prevent DoS
        max_size = self.config.get("max_expression_size", 10000)
        if len(answer_text) > max_size or len(gt_str) > max_size:
            return VerificationResult.failure(
                explanation=f"Expression too large (max {max_size} characters)",
                verifier_name=self.name,
            )

        # Try verification strategies in order
        symbolic_first = self.config.get("symbolic_first", True)

        if symbolic_first and SYMPY_AVAILABLE:
            # Try symbolic equivalence first
            result = self._verify_symbolic(answer_text, gt_str)
            if result.is_correct or result.confidence > 0.5:
                return result

        # Try unit-aware comparison
        if self.config.get("enable_units", True) and PINT_AVAILABLE:
            result = self._verify_with_units(answer_text, gt_str)
            if result.is_correct:
                return result

        # Fall back to numeric comparison
        return self._verify_numeric(answer_text, gt_str)

    def _verify_symbolic(self, answer_text: str, ground_truth: str) -> VerificationResult:
        """
        Verify using symbolic equivalence with SymPy.

        Args:
            answer_text: Candidate answer text
            ground_truth: Ground truth value/expression

        Returns:
            VerificationResult
        """
        try:
            # Extract expressions from text
            answer_expr = self._extract_expression(answer_text)
            gt_expr = self._extract_expression(ground_truth)

            if answer_expr is None or gt_expr is None:
                return VerificationResult.failure(
                    explanation="Could not parse expression",
                    verifier_name=self.name,
                )

            # Convert to SymPy expressions
            ans_sympy = self._to_sympy(answer_expr)
            gt_sympy = self._to_sympy(gt_expr)

            if ans_sympy is None or gt_sympy is None:
                return VerificationResult.failure(
                    explanation="Could not convert to symbolic expression",
                    verifier_name=self.name,
                )

            # Check symbolic equivalence
            try:
                # Simplify difference
                diff = sp.simplify(ans_sympy - gt_sympy)

                if diff == 0 or diff.equals(sp.S.Zero):
                    return VerificationResult.success(
                        explanation=f"Symbolically equivalent: {answer_expr} = {gt_expr}",
                        confidence=1.0,
                        verifier_name=self.name,
                        metadata={"method": "symbolic", "answer": str(ans_sympy), "ground_truth": str(gt_sympy)},
                    )
                else:
                    # Try numeric evaluation as fallback
                    try:
                        ans_float = float(ans_sympy.evalf())
                        gt_float = float(gt_sympy.evalf())
                        tolerance = self.config.get("numeric_tolerance", 1e-6)

                        if math.isclose(ans_float, gt_float, rel_tol=tolerance, abs_tol=tolerance):
                            return VerificationResult.success(
                                explanation=f"Numerically equivalent: {ans_float} ≈ {gt_float}",
                                confidence=0.95,
                                verifier_name=self.name,
                                metadata={"method": "symbolic_numeric", "answer": ans_float, "ground_truth": gt_float},
                            )
                    except:
                        pass

                    return VerificationResult.failure(
                        explanation=f"Not equivalent: {answer_expr} ≠ {gt_expr} (diff: {diff})",
                        verifier_name=self.name,
                    )

            except Exception as e:
                logger.debug(f"Symbolic equivalence check failed: {e}")
                return VerificationResult.failure(
                    explanation=f"Symbolic comparison error: {e}",
                    verifier_name=self.name,
                )

        except Exception as e:
            logger.debug(f"Symbolic verification failed: {e}")
            return VerificationResult.failure(
                explanation=f"Symbolic verification error: {e}",
                verifier_name=self.name,
            )

    def _verify_with_units(self, answer_text: str, ground_truth: str) -> VerificationResult:
        """
        Verify with unit-aware comparison using pint.

        Args:
            answer_text: Candidate answer text
            ground_truth: Ground truth value with units

        Returns:
            VerificationResult
        """
        try:
            # Try to parse as quantities with units
            ans_qty = self._parse_quantity(answer_text)
            gt_qty = self._parse_quantity(ground_truth)

            if ans_qty is None or gt_qty is None:
                return VerificationResult.failure(
                    explanation="Could not parse units",
                    verifier_name=self.name,
                )

            # Check if dimensionally compatible
            if not ans_qty.dimensionality == gt_qty.dimensionality:
                return VerificationResult.failure(
                    explanation=f"Incompatible units: {ans_qty.units} vs {gt_qty.units}",
                    verifier_name=self.name,
                )

            # Convert to same units and compare
            gt_in_ans_units = gt_qty.to(ans_qty.units)
            tolerance = self.config.get("unit_tolerance", 1e-4)

            if math.isclose(ans_qty.magnitude, gt_in_ans_units.magnitude,
                          rel_tol=tolerance, abs_tol=tolerance):
                return VerificationResult.success(
                    explanation=f"Unit-aware match: {ans_qty} = {gt_qty}",
                    confidence=1.0,
                    verifier_name=self.name,
                    metadata={"method": "units", "answer": str(ans_qty), "ground_truth": str(gt_qty)},
                )
            else:
                return VerificationResult.failure(
                    explanation=f"Values differ: {ans_qty} ≠ {gt_qty}",
                    verifier_name=self.name,
                )

        except Exception as e:
            logger.debug(f"Unit-aware verification failed: {e}")
            return VerificationResult.failure(
                explanation=f"Unit verification error: {e}",
                verifier_name=self.name,
            )

    def _verify_numeric(self, answer_text: str, ground_truth: str) -> VerificationResult:
        """
        Verify using numeric comparison with tolerance.

        Args:
            answer_text: Candidate answer text
            ground_truth: Ground truth numeric value

        Returns:
            VerificationResult
        """
        try:
            # Extract numeric values
            ans_val = self._extract_number(answer_text)
            gt_val = self._extract_number(ground_truth)

            if ans_val is None or gt_val is None:
                return VerificationResult.failure(
                    explanation="Could not extract numeric value",
                    verifier_name=self.name,
                )

            # Compare with tolerance
            tolerance = self.config.get("numeric_tolerance", 1e-6)

            # Handle special cases
            if math.isnan(ans_val) or math.isnan(gt_val):
                return VerificationResult.failure(
                    explanation="NaN value detected",
                    verifier_name=self.name,
                )

            if math.isinf(ans_val) or math.isinf(gt_val):
                # Infinities must match exactly (both +inf or both -inf)
                if ans_val == gt_val:
                    return VerificationResult.success(
                        explanation=f"Infinite values match: {ans_val}",
                        confidence=1.0,
                        verifier_name=self.name,
                        metadata={"method": "numeric", "answer": ans_val, "ground_truth": gt_val},
                    )
                else:
                    return VerificationResult.failure(
                        explanation=f"Infinite values differ: {ans_val} ≠ {gt_val}",
                        verifier_name=self.name,
                    )

            # Regular numeric comparison
            if math.isclose(ans_val, gt_val, rel_tol=tolerance, abs_tol=tolerance):
                return VerificationResult.success(
                    explanation=f"Numeric match: {ans_val} ≈ {gt_val} (tolerance: {tolerance})",
                    confidence=0.9,
                    verifier_name=self.name,
                    metadata={"method": "numeric", "answer": ans_val, "ground_truth": gt_val},
                )
            else:
                relative_error = abs((ans_val - gt_val) / gt_val) if gt_val != 0 else float('inf')
                return VerificationResult.failure(
                    explanation=f"Numeric mismatch: {ans_val} ≠ {gt_val} (rel_error: {relative_error:.2e})",
                    verifier_name=self.name,
                )

        except Exception as e:
            logger.debug(f"Numeric verification failed: {e}")
            return VerificationResult.failure(
                explanation=f"Numeric verification error: {e}",
                verifier_name=self.name,
            )

    def _extract_expression(self, text: str) -> Optional[str]:
        """
        Extract mathematical expression from text.

        Handles:
        - LaTeX boxed: \\boxed{42}
        - LaTeX fractions: \\frac{3}{4}
        - Equations: x = 5
        - Plain expressions: 2*pi + 1

        Args:
            text: Text containing expression

        Returns:
            Extracted expression string or None
        """
        # Try LaTeX boxed notation
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1)

        # Try to find equation (x = value)
        eq_match = re.search(r'=\s*([^=\n]+?)(?:\s|$|,|\.)', text)
        if eq_match:
            return eq_match.group(1).strip()

        # Try to find last number/expression
        # Look for mathematical patterns
        math_pattern = r'(?:^|\s)([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?(?:/\d+)?(?:\s*[*+\-/^]\s*[\w\d.()]+)*)'
        matches = re.findall(math_pattern, text)
        if matches:
            return matches[-1].strip()  # Return last match

        # Return the whole text as fallback
        return text.strip()

    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text.

        Handles:
        - LaTeX boxed: \\boxed{42}
        - Fractions: 3/4
        - Percentages: 85%
        - Scientific notation: 1.5e10
        - Comma separators: 1,000
        - Dollar amounts: $1,234.56

        Args:
            text: Text containing number

        Returns:
            Numeric value or None
        """
        # Remove LaTeX boxed notation
        text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)

        # Try patterns in order of specificity
        patterns = [
            (r'(-?\d+\.?\d*[eE][+-]?\d+)', lambda m: float(m.group(1))),  # Scientific notation
            (r'(-?\d+)/(\d+)', lambda m: float(m.group(1)) / float(m.group(2))),  # Fractions
            (r'(-?\d+\.?\d*)%', lambda m: float(m.group(1))),  # Percentages (return as number, not fraction)
            (r'\$?(-?[\d,]+\.?\d*)', lambda m: float(m.group(1).replace(',', ''))),  # Dollar amounts / comma-separated
        ]

        for pattern, converter in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Use last match (final answer usually at end)
                    if isinstance(matches[-1], tuple):
                        return converter(re.search(pattern, text))
                    else:
                        match_obj = list(re.finditer(pattern, text))[-1]
                        return converter(match_obj)
                except (ValueError, ZeroDivisionError):
                    continue

        return None

    def _to_sympy(self, expr_str: str) -> Optional[sp.Expr]:
        """
        Convert string expression to SymPy expression.

        Args:
            expr_str: Expression string

        Returns:
            SymPy expression or None
        """
        if not SYMPY_AVAILABLE:
            return None

        try:
            # Try parsing as LaTeX first if it contains backslashes
            if self.config.get("latex_parsing", True) and '\\' in expr_str:
                try:
                    return parse_latex(expr_str)
                except:
                    pass  # Fall through to regular parsing

            # Try regular SymPy parsing
            return sp.sympify(expr_str, evaluate=False)

        except Exception as e:
            logger.debug(f"Failed to parse '{expr_str}' as SymPy expression: {e}")
            return None

    def _parse_quantity(self, text: str) -> Optional[Any]:
        """
        Parse text as quantity with units using pint.

        Args:
            text: Text containing number and unit (e.g., "1000 meters", "1 km")

        Returns:
            Pint Quantity or None
        """
        if not PINT_AVAILABLE:
            return None

        try:
            # Clean up text
            text = text.strip()

            # Try parsing directly
            return ureg.parse_expression(text)

        except Exception as e:
            logger.debug(f"Failed to parse '{text}' as quantity: {e}")
            return None
