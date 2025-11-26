"""
Lightweight AST syntax checking for code verification.

Fast, free validation before expensive LLM-as-judge or Docker execution.
Checks if code is syntactically valid without executing it.
"""
import ast
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def extract_code_from_text(text: str, language: str = "python") -> Optional[str]:
    """Extract code from markdown fences or raw text."""
    # Try markdown code fence
    pattern = rf'```{language}\n(.+?)```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic code fence
    match = re.search(r'```\n(.+?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Use raw text
    return text.strip()


def check_python_syntax(code: str) -> Dict[str, Any]:
    """
    Check if Python code is syntactically valid using AST.

    Args:
        code: Python code string

    Returns:
        Dict with is_valid, explanation, details
    """
    try:
        # Parse code into AST
        tree = ast.parse(code)

        # Basic checks
        has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        has_class = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        num_statements = len(tree.body)

        return {
            "is_valid": True,
            "confidence": 0.8,  # Syntax is valid, but can't verify correctness
            "explanation": f"Valid Python syntax ({num_statements} statements)",
            "details": {
                "has_function": has_function,
                "has_class": has_class,
                "num_statements": num_statements,
            }
        }

    except SyntaxError as e:
        return {
            "is_valid": False,
            "confidence": 1.0,  # Definitely invalid
            "explanation": f"Python syntax error: {e.msg} at line {e.lineno}",
            "details": {"error": str(e)}
        }
    except Exception as e:
        return {
            "is_valid": False,
            "confidence": 0.5,  # Uncertain
            "explanation": f"AST parsing failed: {e}",
            "details": {"error": str(e)}
        }


def check_javascript_syntax(code: str) -> Dict[str, Any]:
    """
    Enhanced JavaScript syntax heuristics (no full parser, but more comprehensive).

    Checks:
    - Bracket/brace/parenthesis matching
    - String literal balance
    - Template literal balance
    - Common syntax errors (unclosed statements, invalid keywords)

    Returns:
        Dict with is_valid, confidence, explanation
    """
    issues = []
    warnings = []

    # Check for matched brackets/braces/parens (accounting for strings)
    bracket_errors = _check_bracket_balance(code)
    if bracket_errors:
        issues.extend(bracket_errors)

    # Check for unclosed strings
    string_errors = _check_string_balance(code)
    if string_errors:
        issues.extend(string_errors)

    # Check for template literals
    template_count = code.count('`')
    if template_count % 2 != 0:
        issues.append("Unclosed template literal (backtick)")

    # Check for common syntax errors
    common_errors = _check_common_js_errors(code)
    issues.extend(common_errors)

    # Check for function/class definitions
    has_function = bool(re.search(r'function\s+\w+|const\s+\w+\s*=\s*[\(\[]|=>\s*{?', code))
    has_class = bool(re.search(r'class\s+\w+', code))
    has_export = bool(re.search(r'export\s+(default\s+)?', code))

    # Calculate confidence based on code structure
    structure_score = sum([
        has_function,
        has_class,
        'return' in code,
        bool(re.search(r'(const|let|var)\s+\w+', code)),
    ])

    if issues:
        return {
            "is_valid": False,
            "confidence": 0.8,  # Higher confidence in errors
            "explanation": f"Syntax issues: {'; '.join(issues)}",
            "details": {"issues": issues}
        }
    else:
        # Confidence based on how much JS structure we found
        confidence = min(0.7, 0.4 + (structure_score * 0.1))
        return {
            "is_valid": True,
            "confidence": confidence,
            "explanation": f"Syntax checks passed ({structure_score} JS patterns found)",
            "details": {
                "has_function": has_function,
                "has_class": has_class,
                "has_export": has_export,
                "structure_score": structure_score,
            }
        }


def _check_bracket_balance(code: str) -> List[str]:
    """
    Check bracket balance accounting for strings.

    Returns list of error messages.
    """
    errors = []

    # Simple state machine to track brackets outside of strings
    stack = []
    in_string = None  # None, '"', "'", or '`'
    i = 0

    while i < len(code):
        char = code[i]

        # Handle escape sequences
        if i > 0 and code[i-1] == '\\':
            i += 1
            continue

        # String handling
        if in_string:
            if char == in_string:
                in_string = None
        elif char in '"\'`':
            in_string = char
        elif char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack:
                errors.append(f"Unexpected '{char}' with no opening bracket")
            else:
                expected = {'(': ')', '{': '}', '[': ']'}[stack[-1]]
                if char == expected:
                    stack.pop()
                else:
                    errors.append(f"Mismatched bracket: expected '{expected}', got '{char}'")

        i += 1

    if stack:
        unclosed = ''.join(stack)
        errors.append(f"Unclosed brackets: {unclosed}")

    return errors


def _check_string_balance(code: str) -> List[str]:
    """Check for unclosed string literals."""
    errors = []

    # Count unescaped quotes outside of other strings
    for quote in ['"', "'"]:
        # Very basic: count quotes and check if even
        # This is imperfect but catches obvious issues
        count = 0
        i = 0
        while i < len(code):
            if code[i] == '\\' and i + 1 < len(code):
                i += 2  # Skip escaped char
                continue
            if code[i] == quote:
                count += 1
            i += 1

        if count % 2 != 0:
            errors.append(f"Unclosed string ({quote})")

    return errors


def _check_common_js_errors(code: str) -> List[str]:
    """Check for common JavaScript syntax errors."""
    errors = []

    # Check for obviously broken statements
    if re.search(r'function\s*\(\s*\)\s*$', code.strip()):
        errors.append("Function definition without body")

    if re.search(r'if\s*\([^)]*\)\s*$', code.strip()):
        errors.append("If statement without body")

    if re.search(r'for\s*\([^)]*\)\s*$', code.strip()):
        errors.append("For loop without body")

    # Check for invalid syntax patterns
    if re.search(r'const\s*=', code):
        errors.append("const without variable name")

    if re.search(r'let\s*=', code):
        errors.append("let without variable name")

    # Check for double semicolons (common mistake)
    if ';;' in code:
        pass  # Actually valid in JS, just unusual

    return errors


def check_code_syntax(code: str, language: str) -> Dict[str, Any]:
    """
    Check code syntax for any supported language.

    Args:
        code: Code string
        language: Programming language (python, javascript, etc.)

    Returns:
        Dict with is_valid, confidence, explanation
    """
    if language == "python":
        return check_python_syntax(code)
    elif language in ["javascript", "typescript", "js", "ts"]:
        return check_javascript_syntax(code)
    else:
        # Unknown language - can't validate
        return {
            "is_valid": None,
            "confidence": 0.0,
            "explanation": f"Syntax checking not implemented for {language}",
        }
