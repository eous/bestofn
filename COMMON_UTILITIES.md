# Common Utilities Guide

Shared utilities used by both OpenAI and Claude generation pipelines. These modules provide resilience, validation, quality assessment, and pipeline management.

## Module Overview

| Module | Purpose |
|--------|---------|
| `api_retry.py` | API resilience with exponential backoff |
| `response_validation.py` | Output truncation and safety limits |
| `quality_metrics.py` | Response quality assessment |
| `refusal_check.py` | Two-pass refusal detection |
| `llm_judge_fallback.py` | Fallback verification strategy |
| `regen_pipeline.py` | Shared regeneration utilities |

---

## API Retry (`api_retry.py`)

Provides a generic async retry wrapper for API calls with exponential backoff and jitter.

### Features
- Handles timeouts, rate limits (429), server errors (500, 502, 503)
- Exponential backoff with jitter to avoid thundering herd
- Configurable timeout, max retries, and base delay

### Configuration Constants
```python
DEFAULT_TIMEOUT = 300.0      # 5 minutes for extended thinking
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0     # Base delay (multiplied by 2^attempt)
```

### Usage
```python
from common import call_with_retry

response = await call_with_retry(
    lambda: client.messages.create(model="claude-3", messages=[...]),
    timeout=120.0,
    max_retries=3,
)
```

### Retryable Errors
The following patterns trigger automatic retry:
- `rate_limit`, `rate limit`, `429`
- `overloaded`, `503`, `502`, `500`
- `connection`, `timeout`, `server_error`

---

## Response Validation (`response_validation.py`)

Provides response length limits and truncation to prevent memory issues and data corruption.

### Length Limits
```python
MAX_ANSWER_LENGTH = 500_000     # 500K characters
MAX_ANALYSIS_LENGTH = 2_000_000  # 2M characters
MAX_RAW_TEXT_LENGTH = 2_000_000  # 2M characters
```

### Functions

#### `truncate_if_needed(text, max_length, label)`
Truncates text if it exceeds the limit, adding `[TRUNCATED]` marker.

```python
from common import truncate_if_needed, MAX_ANSWER_LENGTH

answer = truncate_if_needed(answer, MAX_ANSWER_LENGTH, "final_answer")
```

#### `truncate_response_fields(final_answer, analysis, raw_text)`
Truncates all response fields to their respective limits.

```python
from common import truncate_response_fields

final_answer, analysis, raw_text = truncate_response_fields(
    final_answer=answer,
    analysis=reasoning,
    raw_text=raw_response,
)
```

#### `is_empty_response(answer, reasoning)`
Detects empty responses (answer with no content).

#### `was_truncated(text)`
Checks if text was previously truncated (ends with `[TRUNCATED]`).

---

## Quality Metrics (`quality_metrics.py`)

Computes quality metrics for model responses, returning a `QualityMetrics` schema object.

### Metrics Computed
- `answer_length`, `reasoning_length`, `plan_length`, `total_response_length`
- `has_reasoning`, `has_plan`
- `is_short_answer` (< 50 chars)
- `is_substantive` (reasoning > 100 or answer > 50)
- `is_empty`
- `completeness_score` (0.0 to 1.0)

### Usage
```python
from common import compute_quality_metrics, is_empty_response

quality = compute_quality_metrics(
    answer=final_answer,
    reasoning=analysis,
    steps=output.steps,
    is_empty=is_empty_response(final_answer, analysis),
)
```

### Completeness Scoring

**Field-based** (OpenAI non-structured):
```python
from common import compute_completeness_from_fields

score = compute_completeness_from_fields(
    normalized_query=query,
    plan=plan,
    reasoning=reasoning,
    answer=answer,
    evaluation=eval_text,
)
# Returns: fields_present / 5.0
```

**Step-based** (OpenAI structured):
```python
from common import compute_completeness_from_steps

score = compute_completeness_from_steps(
    steps=reasoning_steps,
    final_answer=answer,
)
# Returns: (step_score + answer_score) / 2.0
```

---

## Refusal Detection (`refusal_check.py`)

Detects refusals in model responses using a two-pass approach.

### Two-Pass Strategy
1. **First pass**: Check the answer text for refusal patterns
2. **Second pass**: If no refusal found, check full response text
3. Returns the result with highest confidence

### Usage
```python
from common import check_refusal, build_refusal_detection
from verifiers import RefusalClassifier

classifier = RefusalClassifier()

# Get dict result
result = check_refusal(answer, raw_text, classifier)
# Returns: {is_refusal, confidence, refusal_type, matched_patterns}

# Get schema object directly
refusal_detection = build_refusal_detection(answer, raw_text, classifier)
```

---

## LLM Judge Fallback (`llm_judge_fallback.py`)

Provides fallback LLM-as-judge verification when primary verification has low confidence.

### Confidence Thresholds
```python
MIN_PRIMARY_CONFIDENCE = 0.5   # Use LLM judge if primary below this
MIN_LLM_JUDGE_CONFIDENCE = 0.7 # Accept LLM judge if above this
```

### When Fallback Triggers
- Primary verification failed with confidence < 0.5, OR
- Split is `code` or `tool_calling` (need semantic verification)

### Usage
```python
from common import maybe_use_llm_judge

(is_verified, confidence, explanation, verifier_name,
 llm_judge_used, llm_judge_failed) = await maybe_use_llm_judge(
    enabled=args.llm_judge_fallback,
    question=question,
    answer=final_answer,
    ground_truth=ground_truth,
    split=split,
    is_verified=is_verified,
    confidence=confidence,
    is_empty=is_empty,
    provider="openai",  # or "claude"
    api_key=args.api_key,
)
```

### Predicate Function
```python
from common import should_use_llm_judge

if should_use_llm_judge(is_verified, confidence, split, has_ground_truth, is_empty):
    # Trigger fallback verification
    pass
```

---

## Regeneration Pipeline (`regen_pipeline.py`)

Shared utilities for regenerating candidates from existing parquet files.

### Loading and Filtering
```python
from common import load_parquet, filter_rows

df = load_parquet(Path("results/experiment.parquet"))

# Filter by split and/or failed status
filtered_df = filter_rows(df, split='tool_calling', failed_only=True)
```

### Extracting Unique Queries
```python
from common import extract_unique_queries

queries = extract_unique_queries(filtered_df)
for query in queries:
    # query has: query_id, question, ground_truth, split, metadata, original_index
    results = await regenerate_query(query, ...)
```

### Merging Results
```python
from common import merge_results

updated_df = merge_results(original_df, regenerated_results, split='tool_calling')
updated_df.to_parquet("output.parquet")
```

### Checkpointing
```python
from common import save_checkpoint

if completed % 10 == 0:
    save_checkpoint(output_path, completed, total, results)
```

### Schema Flexibility
The module handles both old and new DataFrame column names:
- New: `input_messages`, `verification_is_verified`, `source_metadata`
- Old: `question`, `is_correct`, `metadata`

---

## Importing Utilities

All utilities are exported from `common/__init__.py`:

```python
# Import individual functions
from common import call_with_retry, truncate_if_needed, compute_quality_metrics

# Or import everything
from common import (
    # API retry
    call_with_retry, is_retryable_error,
    DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY,

    # Response validation
    truncate_if_needed, truncate_response_fields, is_empty_response, was_truncated,
    MAX_ANSWER_LENGTH, MAX_ANALYSIS_LENGTH, MAX_RAW_TEXT_LENGTH,

    # Quality metrics
    compute_quality_metrics, compute_completeness_from_fields, compute_completeness_from_steps,

    # Refusal detection
    check_refusal, build_refusal_detection,

    # LLM judge fallback
    maybe_use_llm_judge, should_use_llm_judge,
    MIN_PRIMARY_CONFIDENCE, MIN_LLM_JUDGE_CONFIDENCE,

    # Regeneration pipeline
    load_parquet, filter_rows, extract_unique_queries,
    merge_results, save_checkpoint, clean_nan_values,
    extract_question_from_input_messages,
)
```

---

## Integration Example

Here's how the utilities work together in a generation pipeline:

```python
from common import (
    call_with_retry,
    truncate_response_fields,
    is_empty_response,
    compute_quality_metrics,
    check_refusal,
    maybe_use_llm_judge,
)

# 1. Make API call with retry
response = await call_with_retry(
    lambda: client.messages.create(model="claude-3", messages=messages),
    timeout=300.0,
)

# 2. Truncate oversized responses
final_answer, analysis, raw_text = truncate_response_fields(
    final_answer=response.answer,
    analysis=response.reasoning,
    raw_text=response.raw,
)

# 3. Check for empty response
is_empty = is_empty_response(final_answer, analysis)

# 4. Compute quality metrics
quality = compute_quality_metrics(
    answer=final_answer,
    reasoning=analysis,
    is_empty=is_empty,
)

# 5. Check for refusals
refusal = check_refusal(final_answer, raw_text, refusal_classifier)

# 6. Run verification with fallback
(is_verified, confidence, explanation, verifier_name,
 llm_judge_used, llm_judge_failed) = await maybe_use_llm_judge(
    enabled=True,
    question=question,
    answer=final_answer,
    ground_truth=ground_truth,
    split=split,
    is_verified=primary_result,
    confidence=primary_confidence,
    is_empty=is_empty,
    provider="claude",
    api_key=api_key,
)
```
