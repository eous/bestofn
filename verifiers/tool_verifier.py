"""
Tool call verification using JSON Schema validation.

Features:
- JSON Schema validation (Draft 7 and OpenAPI 3.x)
- Semantic validation (tool existence, parameter types)
- Required field checking
- Parameter type validation
- Mock execution capability
- Detailed error messages

Security:
- Size limits on JSON inputs
- No code execution
- Schema validation prevents injection
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List

try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logging.warning("jsonschema not available. Schema validation disabled.")

from .base import Verifier, VerificationResult, VerificationError

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Verifier
# ============================================================================

class ToolVerifier(Verifier):
    """
    Verifies tool call / function calling answers using JSON Schema validation.

    Supports:
    - JSON syntax validation
    - JSON Schema validation (Draft 7)
    - OpenAPI 3.x schema validation
    - Semantic validation (tool exists, parameters valid)
    - Required field checking
    - Parameter type validation
    - Custom validation rules

    Validation modes:
    - 'strict': Full schema validation, all requirements enforced
    - 'lenient': Schema validation but allow extra fields
    - 'none': Only JSON syntax validation
    """

    @property
    def name(self) -> str:
        return "tool"

    def _verify_impl(self, question: str, candidate: Dict[str, Any],
                     spec: Dict[str, Any]) -> VerificationResult:
        """
        Verify tool call answer.

        Args:
            question: Original question requesting tool use
            candidate: Candidate answer with 'text' field containing JSON
            spec: Specification with:
                - schema: JSON schema for validation (optional)
                - tool_catalog: Dict of available tools (optional)
                - required_fields: List of required field names (optional)
                - ground_truth: Expected tool call (optional)

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

        # Check size limit
        max_size = self.config.get("max_json_size", 100000)
        if len(answer_text) > max_size:
            return VerificationResult.failure(
                explanation=f"JSON too large (max {max_size} characters)",
                verifier_name=self.name,
            )

        # Try Harmony tool call extraction first (for GPT-OSS compatibility)
        harmony_tool_call = self._extract_harmony_tool_call(answer_text)
        if harmony_tool_call:
            logger.debug("Using Harmony commentary channel tool call")
            json_data = harmony_tool_call
        else:
            # Fallback: Extract JSON from text (markdown, inline, etc.)
            json_data = self._extract_json(answer_text)
            if json_data is None:
                return VerificationResult.failure(
                    explanation="Could not parse JSON from answer",
                    verifier_name=self.name,
                )

        # Get validation mode
        validation_mode = self.config.get("schema_validation", "strict")

        # If validation is disabled, just check JSON is valid
        if validation_mode == "none":
            return VerificationResult.success(
                explanation="Valid JSON (schema validation disabled)",
                confidence=0.5,
                verifier_name=self.name,
                metadata={"method": "syntax_only", "data": json_data},
            )

        # Validate against schema if provided
        schema = spec.get("schema")
        if schema:
            schema_result = self._validate_schema(json_data, schema, validation_mode)
            if not schema_result.is_correct:
                return schema_result

        # Validate tool exists if catalog provided
        if self.config.get("validate_tool_exists", True):
            tool_catalog = spec.get("tool_catalog")
            if tool_catalog:
                tool_result = self._validate_tool_exists(json_data, tool_catalog)
                if not tool_result.is_correct:
                    return tool_result

        # Validate required fields
        required_fields = spec.get("required_fields")
        if required_fields:
            required_result = self._validate_required_fields(json_data, required_fields)
            if not required_result.is_correct:
                return required_result

        # Validate parameter types if enabled
        if self.config.get("validate_parameter_types", True):
            param_result = self._validate_parameter_types(json_data, schema)
            if not param_result.is_correct:
                return param_result

        # Compare against ground truth if provided
        ground_truth = spec.get("ground_truth")
        if ground_truth:
            return self._compare_with_ground_truth(json_data, ground_truth)

        # All checks passed
        return VerificationResult.success(
            explanation="Valid tool call",
            confidence=0.95,
            verifier_name=self.name,
            metadata={"method": "full_validation", "data": json_data},
        )

    def _extract_harmony_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from Harmony commentary channel content.

        Harmony tool call format:
        to=functions.get_current_weather <|constrain|>json
        {"location":"San Francisco"}

        Args:
            text: Commentary channel content (already extracted by generator)

        Returns:
            Dict with 'tool', 'recipient', 'parameters' or None
        """
        # Look for recipient pattern: to=namespace.function_name
        recipient_match = re.search(r'to=([\w.]+)', text)
        if not recipient_match:
            return None

        recipient = recipient_match.group(1)

        # Extract function name from recipient (e.g., functions.get_weather → get_weather)
        if '.' in recipient:
            namespace, function_name = recipient.rsplit('.', 1)
        else:
            namespace = ''
            function_name = recipient

        # Extract JSON parameters
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                parameters = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                parameters = {}
        else:
            parameters = {}

        return {
            'tool': function_name,
            'function': function_name,  # Alias
            'recipient': recipient,
            'namespace': namespace,
            'parameters': parameters,
        }

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Extract and parse JSON from text.

        Handles:
        - Harmony commentary channel (for tool calls)
        - Markdown code blocks with ```json
        - Inline JSON objects
        - JSON arrays

        Args:
            text: Text containing JSON

        Returns:
            Parsed JSON data or None
        """
        # First try to extract from Harmony COMMENTARY channel
        # Format: <COMMENTARY>...{json}...</COMMENTARY>
        commentary_match = re.search(r'<COMMENTARY>(.*?)</COMMENTARY>', text, re.DOTALL | re.IGNORECASE)
        if commentary_match:
            text = commentary_match.group(1).strip()
            logger.debug("Extracted JSON from Harmony commentary channel")

        # Try to extract from markdown code block
        json_match = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object or array in text
            # Look for { ... } or [ ... ]
            brace_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(1)
            else:
                json_str = text.strip()

        # Try to parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            return None

    def _validate_schema(self, data: Any, schema: Dict[str, Any],
                        mode: str) -> VerificationResult:
        """
        Validate data against JSON schema.

        Args:
            data: Data to validate
            schema: JSON schema
            mode: Validation mode ('strict' or 'lenient')

        Returns:
            VerificationResult
        """
        if not JSONSCHEMA_AVAILABLE:
            return VerificationResult.success(
                explanation="Schema validation skipped (jsonschema not available)",
                confidence=0.5,
                verifier_name=self.name,
            )

        try:
            # Configure validator based on mode
            if mode == "lenient":
                # Allow additional properties
                schema = schema.copy()
                if "additionalProperties" not in schema:
                    schema["additionalProperties"] = True

            # Validate
            validate(instance=data, schema=schema, cls=Draft7Validator)

            return VerificationResult.success(
                explanation="Schema validation passed",
                confidence=1.0,
                verifier_name=self.name,
                metadata={"method": "schema", "mode": mode},
            )

        except ValidationError as e:
            return VerificationResult.failure(
                explanation=f"Schema validation failed: {e.message}",
                verifier_name=self.name,
                metadata={"validation_error": e.message, "path": list(e.path)},
            )
        except Exception as e:
            return VerificationResult.failure(
                explanation=f"Schema validation error: {e}",
                verifier_name=self.name,
            )

    def _validate_tool_exists(self, data: Any,
                             tool_catalog: Dict[str, Any]) -> VerificationResult:
        """
        Validate that the tool/function being called exists in catalog.

        Args:
            data: Tool call data
            tool_catalog: Dictionary of available tools

        Returns:
            VerificationResult
        """
        # Extract tool name from data
        tool_name = None

        if isinstance(data, dict):
            # Common field names for tool/function name
            for field in ['tool', 'function', 'name', 'tool_name', 'function_name']:
                if field in data:
                    tool_name = data[field]
                    break

            # Check nested structure (e.g., {"function": {"name": "..."}})
            if tool_name is None and 'function' in data and isinstance(data['function'], dict):
                tool_name = data['function'].get('name')

        if tool_name is None:
            return VerificationResult.failure(
                explanation="Could not find tool name in JSON",
                verifier_name=self.name,
            )

        # Check if tool exists in catalog
        if tool_name not in tool_catalog:
            available = ", ".join(tool_catalog.keys())
            return VerificationResult.failure(
                explanation=f"Unknown tool '{tool_name}'. Available: {available}",
                verifier_name=self.name,
                metadata={"tool_name": tool_name, "available_tools": list(tool_catalog.keys())},
            )

        return VerificationResult.success(
            explanation=f"Tool '{tool_name}' exists in catalog",
            confidence=1.0,
            verifier_name=self.name,
            metadata={"method": "tool_exists", "tool_name": tool_name},
        )

    def _validate_required_fields(self, data: Any,
                                  required_fields: List[str]) -> VerificationResult:
        """
        Validate that all required fields are present.

        Args:
            data: JSON data
            required_fields: List of required field names

        Returns:
            VerificationResult
        """
        if not isinstance(data, dict):
            return VerificationResult.failure(
                explanation="Data is not a JSON object (required for field validation)",
                verifier_name=self.name,
            )

        missing = [field for field in required_fields if field not in data]

        if missing:
            return VerificationResult.failure(
                explanation=f"Missing required fields: {', '.join(missing)}",
                verifier_name=self.name,
                metadata={"missing_fields": missing, "required_fields": required_fields},
            )

        return VerificationResult.success(
            explanation="All required fields present",
            confidence=1.0,
            verifier_name=self.name,
            metadata={"method": "required_fields", "fields": required_fields},
        )

    def _validate_parameter_types(self, data: Any,
                                  schema: Optional[Dict[str, Any]]) -> VerificationResult:
        """
        Validate parameter types match expected types.

        Args:
            data: JSON data
            schema: JSON schema (optional, used for type hints)

        Returns:
            VerificationResult
        """
        if not isinstance(data, dict):
            return VerificationResult.success(
                explanation="Type validation skipped (data is not object)",
                confidence=0.8,
                verifier_name=self.name,
            )

        # Extract parameters field (common structures)
        parameters = None
        for field in ['parameters', 'params', 'arguments', 'args']:
            if field in data:
                parameters = data[field]
                break

        if parameters is None:
            return VerificationResult.success(
                explanation="No parameters field to validate",
                confidence=0.8,
                verifier_name=self.name,
            )

        # Basic type checking (without schema)
        if not isinstance(parameters, dict):
            return VerificationResult.failure(
                explanation="Parameters field must be an object",
                verifier_name=self.name,
            )

        # Check for common type issues
        for key, value in parameters.items():
            # Check for string representations of numbers (common error)
            if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                # Could be intentional or could be error - give warning confidence
                return VerificationResult(
                    is_correct=True,
                    confidence=0.7,
                    explanation=f"Parameter '{key}' is string but looks like number: '{value}'",
                    verifier_name=self.name,
                    metadata={"method": "parameter_types", "warning": "potential_type_mismatch"},
                )

        return VerificationResult.success(
            explanation="Parameter types look valid",
            confidence=0.9,
            verifier_name=self.name,
            metadata={"method": "parameter_types"},
        )

    def _normalize_key(self, key: str) -> str:
        """Normalize a key for fuzzy matching (lowercase, remove underscores/hyphens)."""
        return key.lower().replace('_', '').replace('-', '')

    def _fuzzy_value_match(self, val1: Any, val2: Any) -> bool:
        """
        Check if two values are semantically equivalent.

        Handles:
        - String case insensitivity
        - Numeric string vs number
        - Common synonyms
        """
        # Exact match
        if val1 == val2:
            return True

        # Both strings - case-insensitive comparison
        if isinstance(val1, str) and isinstance(val2, str):
            if val1.lower() == val2.lower():
                return True

        # Numeric equivalence (string "72" == int 72)
        try:
            if float(val1) == float(val2):
                return True
        except (TypeError, ValueError):
            pass

        # Boolean equivalence
        bool_true = {'true', 'yes', '1', 'on'}
        bool_false = {'false', 'no', '0', 'off'}
        if isinstance(val1, (bool, str)) and isinstance(val2, (bool, str)):
            v1_str = str(val1).lower()
            v2_str = str(val2).lower()
            if (v1_str in bool_true and v2_str in bool_true) or \
               (v1_str in bool_false and v2_str in bool_false):
                return True

        return False

    def _fuzzy_dict_match(self, data: Dict, ground_truth: Dict) -> tuple:
        """
        Fuzzy match two dictionaries.

        Returns:
            Tuple of (matches: bool, confidence: float, details: str)
        """
        # Create normalized key maps
        data_normalized = {self._normalize_key(k): (k, v) for k, v in data.items()}
        gt_normalized = {self._normalize_key(k): (k, v) for k, v in ground_truth.items()}

        matched_keys = []
        mismatched_keys = []
        missing_keys = []

        for gt_norm_key, (gt_orig_key, gt_value) in gt_normalized.items():
            if gt_norm_key in data_normalized:
                data_orig_key, data_value = data_normalized[gt_norm_key]

                # Recursively compare nested dicts
                if isinstance(gt_value, dict) and isinstance(data_value, dict):
                    nested_match, nested_conf, _ = self._fuzzy_dict_match(data_value, gt_value)
                    if nested_match:
                        matched_keys.append(gt_orig_key)
                    else:
                        mismatched_keys.append((gt_orig_key, gt_value, data_value))
                elif self._fuzzy_value_match(data_value, gt_value):
                    matched_keys.append(gt_orig_key)
                else:
                    mismatched_keys.append((gt_orig_key, gt_value, data_value))
            else:
                missing_keys.append(gt_orig_key)

        # Calculate confidence
        total_keys = len(gt_normalized)
        if total_keys == 0:
            return (True, 1.0, "Empty ground truth")

        matched_ratio = len(matched_keys) / total_keys

        if missing_keys or mismatched_keys:
            details = []
            if missing_keys:
                details.append(f"Missing: {missing_keys}")
            if mismatched_keys:
                details.append(f"Mismatched: {[(k, f'expected {e}, got {g}') for k, e, g in mismatched_keys]}")
            return (False, matched_ratio * 0.5, "; ".join(details))

        return (True, 0.85 if matched_ratio == 1.0 else matched_ratio * 0.8, f"Fuzzy matched {len(matched_keys)} keys")

    def _compare_with_ground_truth(self, data: Any,
                                   ground_truth: Any) -> VerificationResult:
        """
        Compare tool call against ground truth with fuzzy matching.

        Supports:
        - Exact matching (confidence=1.0)
        - Lenient matching - extra fields allowed (confidence=0.9)
        - Fuzzy matching - case-insensitive keys/values (confidence=0.85)

        Args:
            data: Parsed JSON data from candidate
            ground_truth: Expected tool call (dict or JSON string)

        Returns:
            VerificationResult
        """
        # Parse ground truth if it's a string
        if isinstance(ground_truth, str):
            try:
                gt_data = json.loads(ground_truth)
            except json.JSONDecodeError:
                return VerificationResult.failure(
                    explanation="Ground truth is not valid JSON",
                    verifier_name=self.name,
                )
        else:
            gt_data = ground_truth

        # Compare - exact match
        if data == gt_data:
            return VerificationResult.success(
                explanation="Exact match with ground truth",
                confidence=1.0,
                verifier_name=self.name,
                metadata={"method": "ground_truth_exact"},
            )

        # Try lenient comparison (ignore field order, extra fields)
        if isinstance(data, dict) and isinstance(gt_data, dict):
            # Check if all ground truth fields are present and match exactly
            matches = all(
                key in data and data[key] == value
                for key, value in gt_data.items()
            )
            if matches:
                return VerificationResult.success(
                    explanation="Matches ground truth (with extra fields)",
                    confidence=0.9,
                    verifier_name=self.name,
                    metadata={"method": "ground_truth_lenient"},
                )

            # Try fuzzy matching
            fuzzy_match, fuzzy_conf, fuzzy_details = self._fuzzy_dict_match(data, gt_data)
            if fuzzy_match:
                return VerificationResult.success(
                    explanation=f"Fuzzy match with ground truth: {fuzzy_details}",
                    confidence=fuzzy_conf,
                    verifier_name=self.name,
                    metadata={"method": "ground_truth_fuzzy", "details": fuzzy_details},
                )

        return VerificationResult.failure(
            explanation=f"Does not match ground truth.\nExpected: {gt_data}\nActual: {data}",
            verifier_name=self.name,
            metadata={"expected": gt_data, "actual": data},
        )


# ============================================================================
# OpenAPI Schema Validator
# ============================================================================

class OpenAPIToolVerifier(ToolVerifier):
    """
    Extended tool verifier with OpenAPI 3.x schema support.

    Handles OpenAPI-specific structures like:
    - Operation objects
    - Parameter objects
    - RequestBody schemas
    - Response schemas
    """

    @property
    def name(self) -> str:
        return "tool_openapi"

    def _validate_openapi_schema(self, data: Any,
                                 openapi_spec: Dict[str, Any]) -> VerificationResult:
        """
        Validate against OpenAPI 3.x specification.

        Args:
            data: Tool call data
            openapi_spec: OpenAPI specification

        Returns:
            VerificationResult
        """
        # Extract operation from spec based on tool name
        # This is a simplified implementation
        # Full OpenAPI validation would require openapi-spec-validator

        tool_name = data.get('name') or data.get('function', {}).get('name')
        if not tool_name:
            return VerificationResult.failure(
                explanation="Could not determine tool name for OpenAPI validation",
                verifier_name=self.name,
            )

        # Look up operation in OpenAPI spec
        paths = openapi_spec.get('paths', {})
        operation_found = False

        for path, methods in paths.items():
            for method, operation in methods.items():
                if operation.get('operationId') == tool_name:
                    operation_found = True
                    # Extract schema from operation
                    schema = operation.get('requestBody', {}).get('content', {}).get('application/json', {}).get('schema')
                    if schema:
                        return self._validate_schema(data.get('parameters', {}), schema, 'strict')

        if not operation_found:
            return VerificationResult.failure(
                explanation=f"Operation '{tool_name}' not found in OpenAPI spec",
                verifier_name=self.name,
            )

        return VerificationResult.success(
            explanation="OpenAPI validation passed (simplified)",
            confidence=0.8,
            verifier_name=self.name,
            metadata={"method": "openapi"},
        )
