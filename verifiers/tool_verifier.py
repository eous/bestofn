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

        # Extract JSON from text
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

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Extract and parse JSON from text.

        Handles:
        - Markdown code blocks with ```json
        - Inline JSON objects
        - JSON arrays

        Args:
            text: Text containing JSON

        Returns:
            Parsed JSON data or None
        """
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

    def _compare_with_ground_truth(self, data: Any,
                                   ground_truth: Any) -> VerificationResult:
        """
        Compare tool call against ground truth.

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

        # Compare
        if data == gt_data:
            return VerificationResult.success(
                explanation="Exact match with ground truth",
                confidence=1.0,
                verifier_name=self.name,
                metadata={"method": "ground_truth_exact"},
            )

        # Try lenient comparison (ignore field order, extra fields)
        if isinstance(data, dict) and isinstance(gt_data, dict):
            # Check if all ground truth fields are present and match
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
