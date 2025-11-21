#!/usr/bin/env python3
"""
Test script to verify the improved parsing in generate_best_of_n.py
"""

import sys
import os
import re
import math
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import just the functions we need, copying them here to avoid dependency issues
class MathVerifier:
    """Verifies numeric answers against a known ground truth."""
    name = "math"

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract a number from text, handling various formats:
        - Comma-separated numbers: 1,000 -> 1000
        - Boxed LaTeX: \\boxed{42} -> 42
        - Dollar amounts: $1,234.56 -> 1234.56
        - Percentages: 85% -> 85
        - Scientific notation: 1.5e10 -> 15000000000
        - Fractions (simple): 3/4 -> 0.75
        """
        if not text:
            return None

        # First, try to extract from common LaTeX boxes
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            text = boxed_match.group(1)

        # Remove dollar signs and percentage signs
        text = text.replace("$", "").replace("%", "")

        # Remove commas used as thousands separators
        text = text.replace(",", "")

        # Look for the last number-like pattern in the text
        # This pattern handles integers, decimals, and scientific notation
        patterns = [
            r"(-?\d+\.?\d*[eE][+-]?\d+)",  # Scientific notation
            r"(-?\d+/\d+)",                  # Fractions
            r"(-?\d+\.?\d*)",                # Regular numbers
        ]

        for pattern in patterns:
            # Find all matches and take the last one
            matches = re.findall(pattern, text)
            if matches:
                last_match = matches[-1]
                try:
                    # Handle fractions
                    if "/" in last_match:
                        num, denom = last_match.split("/")
                        return float(num) / float(denom)
                    else:
                        return float(last_match)
                except (ValueError, ZeroDivisionError):
                    continue

        return None


def extract_xml(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags, with fallback for unclosed tags.

    Handles:
    - Normal: <tag>content</tag>
    - Unclosed at end: <tag>content (end of string)
    - Case insensitive matching
    """
    # First try standard closed tag pattern
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: look for unclosed tag at end of text
    # This captures content from opening tag to end of string
    match = re.search(rf"<{tag}>(.*)$", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Only return if there's actual content and no other opening tag follows
        if content and not re.match(r"^\s*<\w+>", content):
            return content

    return ""


def test_math_extraction():
    """Test the improved number extraction."""
    verifier = MathVerifier()

    test_cases = [
        # (input_text, expected_number)
        ("The answer is 1,000", 1000.0),
        ("Result: $1,234.56", 1234.56),
        ("\\boxed{42}", 42.0),
        ("The probability is 85%", 85.0),
        ("Scientific: 1.5e10", 1.5e10),
        ("Fraction: 3/4", 0.75),
        ("Multiple numbers: 10, 20, 30, final answer: 100", 100.0),
        ("\\boxed{1,500}", 1500.0),
        ("The answer is \\boxed{-3.14}", -3.14),
        ("No boxed, just 42 at the end", 42.0),
    ]

    print("Testing Math Number Extraction:")
    print("-" * 50)

    for text, expected in test_cases:
        result = verifier.extract_number(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {text[:30]:<30} | Expected: {expected:<10} | Got: {result}")

        if result != expected and result is not None and expected is not None:
            # Check if they're close enough (for floating point comparison)
            import math
            if math.isclose(result, expected, rel_tol=1e-9):
                print("  (Close enough due to floating point precision)")


def test_xml_extraction():
    """Test the improved XML extraction with unclosed tags."""

    test_cases = [
        # (input_text, tag, expected_content)
        ("<answer>42</answer>", "answer", "42"),
        ("<answer>42", "answer", "42"),  # Unclosed tag
        ("<ANSWER>Case Insensitive</ANSWER>", "answer", "Case Insensitive"),
        ("<answer>Multi\nline\ncontent</answer>", "answer", "Multi\nline\ncontent"),
        ("<answer>Unclosed at end", "answer", "Unclosed at end"),
        ("<answer>\n  Whitespace handled  \n</answer>", "answer", "Whitespace handled"),
        ("No tag here", "answer", ""),
        ("<answer><nested>tag</nested></answer>", "answer", "<nested>tag</nested>"),
        ("<answer>", "answer", ""),  # Empty unclosed tag
        ("Text before <answer>Content</answer> text after", "answer", "Content"),
        ("Text before <answer>Unclosed content", "answer", "Unclosed content"),
    ]

    print("\nTesting XML Extraction (with unclosed tag fallback):")
    print("-" * 50)

    for text, tag, expected in test_cases:
        result = extract_xml(text, tag)
        status = "✓" if result == expected else "✗"

        # Truncate display for readability
        display_text = text.replace("\n", "\\n")[:40]
        display_expected = expected.replace("\n", "\\n")[:20]
        display_result = result.replace("\n", "\\n")[:20]

        print(f"{status} Tag: {tag:<8} | Input: {display_text:<40}")
        if result != expected:
            print(f"  Expected: '{display_expected}' | Got: '{display_result}'")


def test_combined_scenario():
    """Test a realistic scenario with model output."""

    print("\nTesting Combined Scenario:")
    print("-" * 50)

    # Simulate a model output with unclosed answer tag and formatted number
    model_output = """<normalized_query>Calculate the total cost</normalized_query>
<plan>
1. Add up all items
2. Apply tax
3. Calculate total
</plan>
<reasoning>
First, I'll add the items: $500 + $300 = $800
Then apply 25% tax: $800 * 0.25 = $200
Total: $800 + $200 = $1,000
</reasoning>
<answer>The total cost is \\boxed{1,000}"""  # Note: unclosed answer tag

    # Extract fields
    answer = extract_xml(model_output, "answer")
    print(f"Extracted answer field: '{answer}'")

    # Extract number from answer
    verifier = MathVerifier()
    number = verifier.extract_number(answer)
    print(f"Extracted number: {number}")

    # Verify it matches expected
    expected = 1000.0
    if number == expected:
        print("✓ Successfully extracted formatted number from unclosed tag!")
    else:
        print(f"✗ Expected {expected}, got {number}")


if __name__ == "__main__":
    test_math_extraction()
    test_xml_extraction()
    test_combined_scenario()

    print("\n" + "=" * 50)
    print("Testing complete!")