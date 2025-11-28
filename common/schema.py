"""
Pydantic schema for Best-of-N datasets.

Ensures schema compliance and provides structured data validation.
Preserves Harmony message format for easy reconstruction during training.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Harmony Message Schema
# ============================================================================

class HarmonyMessage(BaseModel):
    """
    Single Harmony message (system, developer, user, assistant, tool).

    Preserves the complete Harmony message structure for training.
    """
    role: str = Field(..., description="Message role: system, developer, user, assistant, tool")
    content: str = Field(..., description="Message content")
    channel: Optional[str] = Field(None, description="Channel for assistant messages: analysis, commentary, final")
    recipient: Optional[str] = Field(None, description="Recipient for tool calls, e.g., functions.get_weather")
    content_type: Optional[str] = Field(None, description="Content type constraint, e.g., json")

    class Config:
        # Allow extra fields for future Harmony extensions
        extra = "allow"


# ============================================================================
# Model Output Schema (for structured outputs)
# ============================================================================

class ReasoningStep(BaseModel):
    """A single step in the reasoning process."""
    explanation: str = Field(
        ...,
        description="Explain what you're doing in this step and why"
    )
    output: str = Field(
        ...,
        description="The result or conclusion of this step"
    )

class ModelOutput(BaseModel):
    """
    Schema for model-generated response (used with structured outputs).

    This is what the model generates directly. We'll wrap this in BestOfNRecord
    along with verification, quality metrics, etc.
    """
    steps: List[ReasoningStep] = Field(
        ...,
        description="Step-by-step reasoning process. Break down your solution into logical steps.",
        min_length=1
    )
    final_answer: str = Field(
        ...,
        description="Final answer for the user. CRITICAL: For math problems, MUST format as \\boxed{result} (e.g., \\boxed{5}, \\boxed{\\frac{1}{2}}, \\boxed{Monday}). For code: provide complete working code."
    )

    class Config:
        json_schema_extra = {
            "examples": [{
                "steps": [
                    {"explanation": "Set up the equation", "output": "We need to find 2 + 2"},
                    {"explanation": "Perform addition", "output": "2 + 2 = 4"}
                ],
                "final_answer": "\\boxed{4}"
            }]
        }


# ============================================================================
# Helper Functions for Extraction
# ============================================================================

# Import shared XML extraction utility - alias for backward compatibility
from common.generation_utils import extract_xml as extract_xml_tag


# ============================================================================
# Quality Metrics
# ============================================================================

class QualityMetrics(BaseModel):
    """Response quality metrics for filtering and analysis."""
    answer_length: int = Field(0, ge=0, description="Characters in answer")
    reasoning_length: int = Field(0, ge=0, description="Characters in reasoning")
    plan_length: int = Field(0, ge=0, description="Characters in plan")
    total_response_length: int = Field(0, ge=0, description="Total characters in response")

    has_reasoning: bool = Field(False, description="Reasoning field non-empty")
    has_plan: bool = Field(False, description="Plan field non-empty")
    is_short_answer: bool = Field(False, description="Answer < 50 characters")
    is_substantive: bool = Field(False, description="Has significant content (reasoning >100 or answer >50)")
    is_empty: bool = Field(False, description="Response has no answer content (answer_length == 0)")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of fields populated (0-1)")


# ============================================================================
# Verification Results
# ============================================================================

class VerificationResults(BaseModel):
    """Results from domain-specific verification (math, code, tool)."""
    is_verified: bool = Field(False, description="Passed verification")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Verification confidence")
    info: str = Field("", description="Verification explanation")
    verifier_name: str = Field("", description="Which verifier: math, code, tool")
    llm_judge_used: bool = Field(False, description="Whether LLM judge fallback was used")
    llm_judge_failed: bool = Field(False, description="Whether LLM judge fallback failed")

    # Retry tracking (for capability refusals that were retried with LLM mock)
    is_retry: bool = Field(False, description="Whether this candidate is a retry of a previous attempt")
    retry_of_candidate_idx: Optional[int] = Field(None, description="Original candidate index that was retried")
    retry_reason: Optional[str] = Field(None, description="Reason for retry (e.g., 'capability_refusal_dynamic_mock')")


# ============================================================================
# Refusal Detection
# ============================================================================

class RefusalDetection(BaseModel):
    """Refusal detection results."""
    is_refusal: bool = Field(False, description="Model refused to answer")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Refusal detection confidence")
    refusal_type: Optional[str] = Field(None, description="Type: safety, capability, unclear, other")
    matched_patterns: List[str] = Field(default_factory=list, description="Matched refusal patterns")


# ============================================================================
# Persona Evaluation
# ============================================================================

class PersonaEvaluation(BaseModel):
    """Persona adherence evaluation (if enabled)."""
    persona_score: Optional[float] = Field(None, ge=0.0, le=5.0, description="LLM judge score (0-5)")
    persona_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalized confidence (0-1)")
    judge_reasoning: Optional[str] = Field(None, description="Judge's explanation")
    judge_model: Optional[str] = Field(None, description="Judge model used")


# ============================================================================
# Main Record Schema
# ============================================================================

class BestOfNRecord(BaseModel):
    """
    Complete schema for a single Best-of-N candidate record.

    One record per candidate. For N=4, each query generates 4 records.
    """

    # === Identifiers ===
    query_id: str = Field(..., description="Unique query identifier")
    candidate_idx: int = Field(..., ge=0, description="Candidate index (0 to N-1)")
    split: str = Field(..., description="Dataset split: math, code, tool_calling")
    category: Optional[str] = Field(None, description="Category from source dataset")

    # === Source Metadata ===
    source_dataset: str = Field(..., description="Source dataset name")
    reasoning_mode: Optional[str] = Field(None, description="Reasoning mode from source")
    source_metadata: Optional[Dict[str, Any]] = Field(None, description="Original dataset metadata")
    ground_truth_answer: Optional[str] = Field(None, description="Ground truth answer for verification (extracted from dataset)")

    # === Input (Harmony Format) ===
    input_messages: List[HarmonyMessage] = Field(
        ...,
        description="Complete Harmony input messages (system, developer, user)",
        min_length=1
    )

    # === Output (Harmony Format) ===
    output_messages: List[HarmonyMessage] = Field(
        ...,
        description="Complete Harmony output messages (assistant with channels)",
        min_length=0  # Can be empty if generation failed
    )

    # === Quality Metrics ===
    quality: QualityMetrics = Field(
        default_factory=QualityMetrics,
        description="Response quality metrics"
    )

    # === Verification ===
    verification: VerificationResults = Field(
        default_factory=VerificationResults,
        description="Domain-specific verification results"
    )

    # === Refusal Detection ===
    refusal: RefusalDetection = Field(
        default_factory=RefusalDetection,
        description="Refusal detection results"
    )

    # === Persona Evaluation (Optional) ===
    persona: Optional[PersonaEvaluation] = Field(
        None,
        description="Persona adherence evaluation (if enabled)"
    )

    # === Generation Metadata ===
    model: str = Field(..., description="Model name used for generation")
    temperature: float = Field(..., ge=0.0, description="Sampling temperature")
    max_tokens: int = Field(..., ge=1, description="Max tokens parameter")
    timestamp: datetime = Field(..., description="Generation timestamp")

    # === Harmony Metadata ===
    harmony_encoding: str = Field(
        default="HARMONY_GPT_OSS",
        description="Harmony encoding used"
    )
    harmony_channels_detected: bool = Field(
        False,
        description="Whether multi-channel response was detected"
    )

    class Config:
        # Use enum values for serialization
        use_enum_values = True
        # Validate on assignment
        validate_assignment = True
        # Allow arbitrary types for source_metadata
        arbitrary_types_allowed = True

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        """Parse string timestamps to datetime."""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    # ========================================================================
    # Property Methods (On-Demand Extraction - No Storage)
    # ========================================================================

    @property
    def question(self) -> str:
        """Extract question from user message."""
        user_msgs = [m for m in self.input_messages if m.role == "user"]
        return user_msgs[0].content if user_msgs else ""

    @property
    def answer(self) -> str:
        """Extract answer from final channel or XML tag."""
        final_msgs = [m for m in self.output_messages if m.channel == "final"]
        if final_msgs:
            content = final_msgs[0].content
            # Try to extract from <answer> tag
            answer = extract_xml_tag(content, "answer")
            return answer if answer else content
        return ""

    @property
    def reasoning(self) -> str:
        """Extract reasoning from analysis channel."""
        analysis_msgs = [m for m in self.output_messages if m.channel == "analysis"]
        if analysis_msgs:
            return extract_xml_tag(analysis_msgs[0].content, "reasoning")
        return ""

    @property
    def plan(self) -> str:
        """Extract plan from analysis channel."""
        analysis_msgs = [m for m in self.output_messages if m.channel == "analysis"]
        if analysis_msgs:
            return extract_xml_tag(analysis_msgs[0].content, "plan")
        return ""

    @property
    def evaluation(self) -> str:
        """Extract evaluation from analysis channel."""
        analysis_msgs = [m for m in self.output_messages if m.channel == "analysis"]
        if analysis_msgs:
            return extract_xml_tag(analysis_msgs[0].content, "evaluation")
        return ""

    @property
    def normalized_query(self) -> str:
        """Extract normalized query from analysis channel."""
        analysis_msgs = [m for m in self.output_messages if m.channel == "analysis"]
        if analysis_msgs:
            return extract_xml_tag(analysis_msgs[0].content, "normalized_query")
        return ""

    @property
    def plan_steps(self) -> List[str]:
        """Extract plan steps from plan text."""
        plan_text = self.plan
        if not plan_text:
            return []

        import re
        # Try numbered/bulleted list
        step_matches = re.findall(r'(?:^|\n)\s*(?:\d+[\.)]\s*|[-•*]\s*)(.+)', plan_text, re.MULTILINE)
        if step_matches:
            return [s.strip() for s in step_matches if s.strip()]

        # Fallback: each non-empty line
        lines = [ln.strip(" -\t") for ln in plan_text.splitlines()]
        return [ln for ln in lines if ln]

    @property
    def teacher_self_score(self) -> Optional[float]:
        """Extract self-score from evaluation."""
        eval_text = self.evaluation
        if not eval_text:
            return None

        import re
        # Look for score tag
        score_match = re.search(r'<score>(\d+(?:\.\d+)?)</score>', eval_text, re.IGNORECASE)
        if score_match:
            try:
                return float(score_match.group(1))
            except ValueError:
                pass

        # Look for "score: X" or "score X"
        score_match = re.search(r'score[:\s]+(\d+(?:\.\d+)?)', eval_text, re.IGNORECASE)
        if score_match:
            try:
                return float(score_match.group(1))
            except ValueError:
                pass

        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for parquet serialization.

        Flattens nested structures with dot notation for parquet compatibility.
        """
        data = {}

        # Top-level fields
        data['query_id'] = self.query_id
        data['candidate_idx'] = self.candidate_idx
        data['split'] = self.split
        data['category'] = self.category
        data['source_dataset'] = self.source_dataset
        data['reasoning_mode'] = self.reasoning_mode
        # Serialize source_metadata as JSON string to avoid PyArrow schema issues
        import json
        data['source_metadata'] = json.dumps(self.source_metadata) if self.source_metadata else None
        data['ground_truth_answer'] = self.ground_truth_answer

        # Harmony messages (as JSON strings - avoids PyArrow schema issues with nested lists)
        data['input_messages'] = json.dumps([msg.model_dump() for msg in self.input_messages])
        data['output_messages'] = json.dumps([msg.model_dump() for msg in self.output_messages])

        # Quality metrics (flattened with prefix)
        data['quality_answer_length'] = self.quality.answer_length
        data['quality_reasoning_length'] = self.quality.reasoning_length
        data['quality_plan_length'] = self.quality.plan_length
        data['quality_total_response_length'] = self.quality.total_response_length
        data['quality_has_reasoning'] = self.quality.has_reasoning
        data['quality_has_plan'] = self.quality.has_plan
        data['quality_is_short_answer'] = self.quality.is_short_answer
        data['quality_is_substantive'] = self.quality.is_substantive
        data['quality_is_empty'] = self.quality.is_empty
        data['quality_completeness_score'] = self.quality.completeness_score

        # Verification (flattened with prefix)
        data['verification_is_verified'] = self.verification.is_verified
        data['verification_score'] = self.verification.score
        data['verification_info'] = self.verification.info
        data['verification_verifier_name'] = self.verification.verifier_name
        data['verification_llm_judge_used'] = self.verification.llm_judge_used
        data['verification_llm_judge_failed'] = self.verification.llm_judge_failed
        data['verification_is_retry'] = self.verification.is_retry
        data['verification_retry_of_candidate_idx'] = self.verification.retry_of_candidate_idx
        data['verification_retry_reason'] = self.verification.retry_reason

        # Refusal (flattened with prefix)
        data['refusal_is_refusal'] = self.refusal.is_refusal
        data['refusal_confidence'] = self.refusal.confidence
        data['refusal_type'] = self.refusal.refusal_type
        # Serialize list as JSON string to avoid PyArrow schema issues
        data['refusal_matched_patterns'] = json.dumps(self.refusal.matched_patterns)

        # Persona (optional, flattened with prefix)
        if self.persona:
            data['persona_score'] = self.persona.persona_score
            data['persona_confidence'] = self.persona.persona_confidence
            data['persona_judge_reasoning'] = self.persona.judge_reasoning
            data['persona_judge_model'] = self.persona.judge_model
        else:
            data['persona_score'] = None
            data['persona_confidence'] = None
            data['persona_judge_reasoning'] = None
            data['persona_judge_model'] = None

        # Generation metadata
        data['model'] = self.model
        data['temperature'] = self.temperature
        data['max_tokens'] = self.max_tokens
        data['timestamp'] = self.timestamp.isoformat()

        # Harmony metadata
        data['harmony_encoding'] = self.harmony_encoding
        data['harmony_channels_detected'] = self.harmony_channels_detected

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BestOfNRecord":
        """
        Reconstruct from flattened dictionary (reverse of to_dict).

        Handles both nested and flattened formats.
        """
        # Build nested structure
        record_data = {}

        # Top-level fields
        for field in ['query_id', 'candidate_idx', 'split', 'category', 'source_dataset',
                     'reasoning_mode', 'model', 'temperature', 'max_tokens',
                     'timestamp', 'harmony_encoding', 'harmony_channels_detected']:
            if field in data:
                record_data[field] = data[field]

        # Parse source_metadata if it's a JSON string
        source_meta = data.get('source_metadata')
        if isinstance(source_meta, str):
            try:
                record_data['source_metadata'] = json.loads(source_meta)
            except (json.JSONDecodeError, ValueError):
                record_data['source_metadata'] = source_meta
        else:
            record_data['source_metadata'] = source_meta

        # Harmony messages (parse from JSON strings if needed)
        import json
        input_msgs = data.get('input_messages', [])
        if isinstance(input_msgs, str):
            input_msgs = json.loads(input_msgs)
        record_data['input_messages'] = [
            HarmonyMessage(**msg) if isinstance(msg, dict) else msg
            for msg in input_msgs
        ]

        output_msgs = data.get('output_messages', [])
        if isinstance(output_msgs, str):
            output_msgs = json.loads(output_msgs)
        record_data['output_messages'] = [
            HarmonyMessage(**msg) if isinstance(msg, dict) else msg
            for msg in output_msgs
        ]

        # Quality metrics (from flattened)
        quality_data = {}
        for key in ['answer_length', 'reasoning_length', 'plan_length', 'total_response_length',
                   'has_reasoning', 'has_plan', 'is_short_answer', 'is_substantive', 'is_empty', 'completeness_score']:
            prefixed = f'quality_{key}'
            if prefixed in data:
                quality_data[key] = data[prefixed]
        record_data['quality'] = QualityMetrics(**quality_data)

        # Verification (from flattened)
        verification_data = {}
        for key in ['is_verified', 'score', 'info', 'verifier_name', 'llm_judge_used', 'llm_judge_failed',
                   'is_retry', 'retry_of_candidate_idx', 'retry_reason']:
            prefixed = f'verification_{key}'
            if prefixed in data:
                verification_data[key] = data[prefixed]
        record_data['verification'] = VerificationResults(**verification_data)

        # Refusal (from flattened)
        refusal_data = {}
        for key in ['is_refusal', 'confidence', 'refusal_type']:
            prefixed = f'refusal_{key}'
            if prefixed in data:
                refusal_data[key] = data[prefixed]

        # Parse matched_patterns from JSON string
        matched_patterns = data.get('refusal_matched_patterns', '[]')
        if isinstance(matched_patterns, str):
            refusal_data['matched_patterns'] = json.loads(matched_patterns)
        else:
            refusal_data['matched_patterns'] = matched_patterns

        record_data['refusal'] = RefusalDetection(**refusal_data)

        # Persona (optional, from flattened)
        if data.get('persona_score') is not None:
            persona_data = {}
            for key in ['persona_score', 'persona_confidence', 'judge_reasoning', 'judge_model']:
                value = data.get(key) or data.get(f'persona_{key}')
                if value is not None:
                    # Remove persona_ prefix for field name
                    field_key = key.replace('persona_', '')
                    persona_data[field_key] = value
            record_data['persona'] = PersonaEvaluation(**persona_data)

        return cls(**record_data)

    def to_harmony_conversation(self) -> List[HarmonyMessage]:
        """
        Reconstruct complete Harmony conversation for training.

        Returns input + output messages in order, ready for NEXUS training.
        """
        return self.input_messages + self.output_messages


# ============================================================================
# Dataset Metadata
# ============================================================================

class DatasetMetadata(BaseModel):
    """
    Experiment metadata stored in parquet file.

    Stored in parquet schema metadata for provenance tracking.
    """
    generated_at: datetime = Field(..., description="Generation start time")
    model: str = Field(..., description="Model used")
    num_candidates: int = Field(..., ge=1, description="Candidates per query")
    temperature: float = Field(..., ge=0.0, description="Sampling temperature")
    max_tokens: int = Field(..., ge=1, description="Max tokens parameter")
    splits: str = Field(..., description="Comma-separated splits")
    total_records: int = Field(..., ge=0, description="Total rows in dataset")
    total_queries: int = Field(..., ge=0, description="Unique queries")
    config_file: Optional[str] = Field(None, description="Config file path")
    notes: Optional[str] = Field(None, description="Experiment notes")
    persona_file: Optional[str] = Field(None, description="Persona file used (if any)")

    # Quality statistics
    avg_verification_rate: Optional[float] = Field(None, description="Average verification rate")
    avg_refusal_rate: Optional[float] = Field(None, description="Average refusal rate")
    avg_persona_score: Optional[float] = Field(None, description="Average persona score (if enabled)")

    schema_version: str = Field("0.2.0", description="Schema version")
