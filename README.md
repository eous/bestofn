# Best-of-N Candidate Generation with Verification

Generate and verify multiple candidate responses for training data, with secure Docker-based code execution and sophisticated verification systems. Supports both OpenAI and Claude APIs.

## Overview

Best-of-N generation creates multiple candidate responses per query and verifies them using domain-specific verifiers (math, code, tool-calling). Useful for:
- **RLHF/DPO training data** - Ranked candidates with quality scores
- **Reward model training** - Verified vs unverified examples
- **Verification research** - Study verifier accuracy and failure modes
- **Personality transfer** - Fine-tune models on persona-injected data

## Features

### Multi-Provider Support
- **OpenAI API** - GPT-4o with structured outputs and Responses API
- **Claude API** - Claude 4.5 (Sonnet/Opus) with extended thinking mode
- **Local Models** - OpenAI-compatible server for custom models

### Secure Verification
- **Docker-isolated code execution** - No exec/eval in main process
- **Multi-language support** - Python, JavaScript, Bash, SQL
- **Resource limits** - CPU, memory, timeout enforcement
- **Production-grade security** - See [verifiers/SECURITY.md](verifiers/SECURITY.md)

### High-Accuracy Verifiers
- **MathVerifier** (97%) - SymPy symbolic + unit-aware + numeric
- **CodeVerifier** (94%) - Docker sandbox with test case execution
- **ToolVerifier** (98%) - JSON Schema validation + LLM mock implementations
- **LLM Judge** - Fallback using same provider (GPT-4o for OpenAI, Sonnet 4.5 for Claude)

### Persona System
- **Flexible templates** - Marvin, Data, Johnny 5 included
- **Personality transfer** - Test distillation of distinctive voices
- **Custom personas** - Create your own styles

### Performance
- **Async generation** - Concurrent API calls with rate limiting
- **Micro-batching** - Prevents OOM for large n values
- **Surgical regeneration** - Retry only failed rows
- **Checkpointing** - Automatic saves every N queries

### Recent Updates
- **Platform-Aware Tool Mocking** - LLM mock uses gpt-4o-mini for OpenAI, sonnet-4.5 for Claude
- **Tool Calling LLM Judge Only** - Skips unreliable ground truth, relies on LLM appropriateness evaluation
- **Claude Opus 4.5 Support** - Full support for `claude-opus-4-5-20251101` in experiment configs
- **Hybrid Refusal Detection** - Two-pass detection: fast pattern matching + LLM fallback for ambiguous cases
- **LaTeX JSON Repair** - LLM judge now handles LaTeX math in reasoning (escapes `\frac`, `\sqrt`, etc.)
- **Tool Iteration Limit** - Increased from 3 to 100 to support complex multi-step tool calling
- **LLM Judge Fallback** - Automatic fallback to LLM-as-judge when primary verification has low confidence
- **API Retry with Backoff** - Exponential backoff with jitter for rate limits and server errors

---

## Verification Flow Diagrams

Understanding how each split (math, code, tool_calling) flows through the verification pipeline is crucial for debugging and extending the system.

### Math Split Verification Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MATH VERIFICATION FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ LLM Response │
    │  (Candidate) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │  Refusal Check   │────────────────────┐
    │ (Pattern + LLM)  │                    │ Refusal detected
    └──────┬───────────┘                    │
           │ Not refusal                    ▼
           ▼                         ┌─────────────┐
    ┌──────────────────┐             │   RECORD    │
    │  Extract Answer  │             │ is_verified │
    │ (boxed{} or end) │             │   = False   │
    └──────┬───────────┘             │   reason:   │
           │                         │  "refusal"  │
           ▼                         └─────────────┘
    ┌──────────────────┐
    │   MathVerifier   │
    │ ──────────────── │
    │ 1. SymPy Parse   │
    │    & Simplify    │
    │                  │
    │ 2. Symbolic      │
    │    Comparison    │
    │                  │
    │ 3. Numeric       │
    │    Fallback      │
    │    (tolerance)   │
    │                  │
    │ 4. Unit-aware    │
    │    Parsing       │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ Confidence Check │
    │   conf >= 0.4?   │
    └──────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
  conf>=0.4   conf<0.4 AND --llm-judge-fallback
     │           │
     │           ▼
     │    ┌─────────────────────┐
     │    │     LLM Judge       │
     │    │  (Sonnet 4.5 or     │
     │    │   GPT-4o-mini)      │
     │    │                     │
     │    │ • Compare candidate │
     │    │   vs ground truth   │
     │    │ • Semantic equiv    │
     │    │ • Expression forms  │
     │    └──────────┬──────────┘
     │               │
     └───────┬───────┘
             │
             ▼
      ┌─────────────┐
      │   RECORD    │
      │ is_verified │
      │ confidence  │
      │ explanation │
      └─────────────┘
```

**Key Points - Math:**
- Local verifier (MathVerifier) handles most cases with SymPy symbolic math
- Confidence threshold of 0.4 determines when to trust local result
- LLM judge only invoked if `--llm-judge-fallback` flag AND confidence < 0.4
- Ground truth from Nemotron dataset is used for comparison

---

### Code Split Verification Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CODE VERIFICATION FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ LLM Response │
    │  (Candidate) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │  Refusal Check   │────────────────────┐
    │ (Pattern + LLM)  │                    │ Refusal detected
    └──────┬───────────┘                    │
           │ Not refusal                    ▼
           ▼                         ┌─────────────┐
    ┌──────────────────┐             │   RECORD    │
    │   AST Syntax     │             │ is_verified │
    │   Quick Check    │             │   = False   │
    └──────┬───────────┘             └─────────────┘
           │
           ▼
    ┌──────────────────┐
    │  CodeVerifier    │
    │ ──────────────── │
    │ 1. Extract code  │
    │    from markdown │
    │                  │
    │ 2. Build Docker  │
    │    sandbox       │
    │                  │
    │ 3. Execute code  │
    │    with test     │
    │    cases         │
    │                  │
    │ 4. Compare       │
    │    outputs       │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ Confidence Check │
    │   conf >= 0.4?   │
    └──────┬───────────┘
           │
     ┌─────┴─────────────────┐
     │                       │
     ▼                       ▼
  conf>=0.4            conf<0.4 (auto-fallback)
     │                       │
     │                       ▼
     │              ┌─────────────────────┐
     │              │     LLM Judge       │
     │              │  (Semantic Code     │
     │              │   Correctness)      │
     │              │                     │
     │              │ • Analyze approach  │
     │              │ • Check edge cases  │
     │              │ • Verify logic      │
     │              └──────────┬──────────┘
     │                         │
     └───────────┬─────────────┘
                 │
                 ▼
          ┌─────────────┐
          │   RECORD    │
          │ is_verified │
          │ confidence  │
          │ explanation │
          └─────────────┘
```

**Key Points - Code:**
- AST syntax check runs first as fast-path rejection
- Docker sandbox isolates code execution (security)
- Auto-fallback to LLM judge for low confidence (no flag needed)
- Test case execution compares actual output vs expected

---

### Tool Calling Verification Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TOOL CALLING VERIFICATION FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Question +  │
    │    Tools     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      MULTI-TURN TOOL LOOP                            │
    │  ┌───────────────────────────────────────────────────────────────┐   │
    │  │                                                               │   │
    │  │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │   │
    │  │   │ LLM Request │────▶│  Tool Call  │────▶│   Execute   │    │   │
    │  │   │  (turn N)   │     │  Detected?  │     │   in        │    │   │
    │  │   └─────────────┘     └──────┬──────┘     │  Sandbox    │    │   │
    │  │                              │            └──────┬──────┘    │   │
    │  │                         No   │ Yes              │            │   │
    │  │                              │                  ▼            │   │
    │  │                              │         ┌──────────────────┐  │   │
    │  │                              │         │  Tool Execution  │  │   │
    │  │                              │         │  ──────────────  │  │   │
    │  │                              │         │ Try dynamic mock │  │   │
    │  │                              │         │        │         │  │   │
    │  │                              │         │   conf < 0.4?    │  │   │
    │  │                              │         │     ╱    ╲       │  │   │
    │  │                              │         │   Yes     No     │  │   │
    │  │                              │         │    │       │     │  │   │
    │  │                              │         │    ▼       ▼     │  │   │
    │  │                              │         │ LLM Mock  Use    │  │   │
    │  │                              │         │ (platform Dynamic│  │   │
    │  │                              │         │  aware)   Mock   │  │   │
    │  │                              │         └────────┬─────────┘  │   │
    │  │                              │                  │            │   │
    │  │                              │                  ▼            │   │
    │  │                              │         ┌──────────────────┐  │   │
    │  │                              │         │ Append result to │  │   │
    │  │                              │         │ conversation     │  │   │
    │  │                              │         └────────┬─────────┘  │   │
    │  │                              │                  │            │   │
    │  │                              │         ◀────────┘            │   │
    │  │                              │  (loop until max_iterations   │   │
    │  │                              │   or no tool calls)           │   │
    │  └──────────────────────────────┼───────────────────────────────┘   │
    │                                 │                                    │
    └─────────────────────────────────┼────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ Final Answer  │
                              │ (no tool call)│
                              └───────┬───────┘
                                      │
                                      ▼
                              ┌──────────────────┐
                              │  Refusal Check   │──────────────────┐
                              │ (Pattern + LLM)  │                  │ Refusal
                              └───────┬──────────┘                  │
                                      │ Not refusal                 ▼
                                      ▼                      ┌─────────────┐
    ┌─────────────────────────────────────────────────┐      │   RECORD    │
    │        SKIP LOCAL VERIFIER (tool_calling)       │      │  refusal    │
    │                                                 │      └─────────────┘
    │  Ground truth from Nemotron is unreliable for   │
    │  tool_calling (generated without reasoning).    │
    │  Go directly to LLM Judge.                      │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
                ┌─────────────────────────────────────┐
                │      LLM JUDGE (MANDATORY)          │
                │  ─────────────────────────────────  │
                │                                     │
                │  • ground_truth = None              │
                │  • Evaluates APPROPRIATENESS:       │
                │    - Does answer address problem?   │
                │    - Are tool calls reasonable?     │
                │    - Is final answer sensible?      │
                │                                     │
                │  Uses: Sonnet 4.5 (Claude gen)      │
                │        GPT-4o-mini (OpenAI gen)     │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                          ┌─────────────┐
                          │   RECORD    │
                          │ is_verified │
                          │ confidence  │
                          │ explanation │
                          └─────────────┘


    ┌─────────────────────────────────────────────────────────────────────┐
    │                     TOOL EXECUTION DETAIL                           │
    │                                                                     │
    │   ┌───────────────────────────────────────────────────────────┐    │
    │   │                   ToolSandbox                              │    │
    │   │  ───────────────────────────────────────────────────────  │    │
    │   │                                                           │    │
    │   │   1. Parse tool call (name, arguments)                    │    │
    │   │                                                           │    │
    │   │   2. Try DYNAMIC MOCK (100+ implementations)              │    │
    │   │      • Weather, stocks, calculations                      │    │
    │   │      • Database queries, file operations                  │    │
    │   │      • Web search, API patterns                           │    │
    │   │      │                                                    │    │
    │   │      └──▶ Returns: result + confidence                    │    │
    │   │                                                           │    │
    │   │   3. If confidence < 0.4 (uncertain):                     │    │
    │   │      │                                                    │    │
    │   │      └──▶ LLM MOCK (platform-aware)                       │    │
    │   │          │                                                │    │
    │   │          ├─ Claude gen → claude-sonnet-4-5-20250929       │    │
    │   │          │                                                │    │
    │   │          └─ OpenAI gen → gpt-4o-mini                      │    │
    │   │                                                           │    │
    │   │   4. Return mock result to conversation                   │    │
    │   │                                                           │    │
    │   └───────────────────────────────────────────────────────────┘    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

**Key Points - Tool Calling:**
- **Multi-turn loop**: Tools execute iteratively until final answer (max 100 iterations)
- **Dynamic mock first**: 100+ tool implementations with confidence scores
- **LLM mock fallback**: Uses platform-appropriate model when dynamic mock uncertain
- **NO local verifier**: Ground truth from Nemotron is unreliable (generated without reasoning)
- **LLM judge is MANDATORY**: Evaluates appropriateness (not ground truth match)
- **Capability refusal handling**: Can detect and requeue samples with capability-based refusals

---

### Verification Flow Legend

```
    ┌───────────────┐
    │  Input/Data   │     Rectangle: Data or state
    └───────────────┘

    ┌───────────────┐
    │   Process     │     Rectangle: Processing step
    │ ───────────── │     with internal details
    │   details     │
    └───────────────┘

           │
           ▼              Arrow: Flow direction

     ┌─────┴─────┐
     │           │        Branch: Decision point
     ▼           ▼
```

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/youruser/bestofn.git
cd bestofn
pip install -r requirements.txt

# Build Docker image for code verification
cd verifiers && ./build_docker.sh && cd ..

# Set API keys
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here  # Optional, for Claude
```

### Run Your First Experiment

```bash
# Generate with OpenAI
python bestofn.py openai generate --config experiments/marvin/openai_100x8.yaml

# Generate with Claude
python bestofn.py claude generate --config experiments/marvin/claude_100x8.yaml

# Inspect results
python inspect_experiment.py experiments/marvin/results/openai_100x8.parquet
```

### Regenerate Failed Rows

```bash
# Retry only failed tool_calling rows
python bestofn.py openai regen results.parquet --split tool_calling --failed-only

# Retry specific splits
python bestofn.py claude regen results.parquet --split math,code
```

### CLI Help

```bash
python bestofn.py --help
python bestofn.py openai --help
python bestofn.py openai generate --help
```

---

## Architecture

```
bestofn/
├── bestofn.py                 # Unified CLI entry point
├── local_server.py            # Local inference server
├── inspect_experiment.py      # Results analysis tool
│
├── openai_gen/                # OpenAI-specific generation
│   ├── generate.py            # Main generation (Responses API)
│   ├── regen.py               # Surgical regeneration
│   └── tool_executor.py       # Multi-turn tool calling
│
├── claude_gen/                # Claude-specific generation
│   ├── generate.py            # Main generation (extended thinking)
│   ├── regen.py               # Surgical regeneration
│   └── tool_executor.py       # Claude tool_use format
│
├── common/                    # Shared utilities
│   ├── schema.py              # Pydantic schemas
│   ├── nemotron_utils.py      # Dataset loading
│   ├── generation_utils.py    # Shared helpers
│   ├── ast_syntax_checker.py  # Fast AST validation
│   ├── llm_judge.py           # LLM-as-judge (GPT-4o or Sonnet 4.5)
│   ├── api_retry.py           # Exponential backoff retry
│   ├── response_validation.py # Output truncation/safety
│   ├── quality_metrics.py     # Response quality scoring
│   ├── refusal_check.py       # Two-pass refusal detection
│   ├── llm_judge_fallback.py  # Fallback verification
│   └── regen_pipeline.py      # Regeneration utilities
│
├── verifiers/                 # Verification system
│   ├── math_verifier.py       # SymPy-based math verification
│   ├── code_verifier.py       # Docker sandbox execution
│   ├── tool_verifier.py       # JSON Schema validation
│   ├── tool_sandbox.py        # Platform-aware mock implementations
│   ├── docker_sandbox.py      # Container management
│   └── refusal_classifier.py  # Refusal detection
│
├── experiments/               # Experiment configs
│   ├── baseline/              # Non-persona baselines
│   ├── marvin/                # Marvin persona experiments
│   ├── data/                  # Data persona experiments
│   ├── j5/                    # Johnny 5 persona experiments
│   └── README.md              # Experiment documentation
│
└── personas/                  # Personality templates
    ├── marvin.txt             # Marvin the Paranoid Android
    ├── marvin_flexible.txt    # Flexible Marvin (high diversity)
    ├── data.txt               # Lt. Cmd. Data
    ├── data_flexible.txt      # Flexible Data
    └── johnny5_flexible.txt   # Johnny 5
```

---

## Configuration

Experiments are defined in YAML:

```yaml
# experiments/my_experiment.yaml
dataset: nvidia/Nemotron-Post-Training-Dataset-v2
splits: math,code,tool_calling
streaming: true
max_queries: 100

# Generation parameters
model: claude-sonnet-4-5-20250929
num_candidates: 8
temperature: 0.7
max_tokens: 16384

# Persona (optional)
persona: personas/marvin_flexible.txt

# Performance
concurrency: 3

# Output
output: experiments/marvin/results/my_run.parquet

# Generator type
generator: claude

# Features
structured_output: true
llm_judge_fallback: true

# Notes (saved in parquet metadata)
notes: |
  Testing Marvin personality with high temperature.
```

---

## Persona Experiments

Generate training data with distinctive personalities:

```bash
# Marvin - depressed robot (negative affect)
python bestofn.py claude generate --config experiments/marvin/claude_100x8.yaml

# Data - precise android (neutral affect)
python bestofn.py claude generate --config experiments/data/claude_100x8.yaml

# Johnny 5 - enthusiastic robot (positive affect)
python bestofn.py claude generate --config experiments/j5/claude_100x8.yaml
```

Flexible persona variants (`*_flexible.txt`) produce higher diversity responses while maintaining character voice.

See [QUICKSTART_PERSONAS.md](QUICKSTART_PERSONAS.md) for complete persona experiment workflow.

---

## Tool Calling

The framework includes 100+ mock tool implementations for realistic tool-calling experiments:

- **Weather, stocks, calculations** - Common API patterns
- **Database queries** - SQL-like interfaces
- **File operations** - Read/write simulations
- **Web search** - Mock search results

### Platform-Aware LLM Mock

When dynamic mocks have low confidence, the system falls back to LLM-generated mocks:

| Generator | LLM Mock Model |
|-----------|----------------|
| OpenAI    | gpt-4o-mini    |
| Claude    | claude-sonnet-4-5-20250929 |

This ensures consistent response styles between generation and mock execution.

### Tool Calling Verification

Tool calling uses **LLM judge only** - the local verifier is skipped because ground truth from the Nemotron dataset is unreliable (generated without reasoning). The LLM judge evaluates:

1. Does the answer address the original problem?
2. Were tool calls reasonable given the question?
3. Is the final answer sensible and complete?

---

## Local Inference

For development and custom models:

```bash
# Terminal 1: Start server
python local_server.py --model /path/to/model --port 8000

# Terminal 2: Generate
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
python bestofn.py openai generate --config experiments/baseline/baseline.yaml
```

Features:
- GPU locking (prevents concurrent access)
- OOM handling with automatic recovery
- Per-request performance logging

---

## Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_parquet('experiments/marvin/results/openai_100x8.parquet')

# Verification rate by split
print(df.groupby('split')['is_verified'].mean())

# Best-of-N improvement
first_pass = df[df.candidate_idx == 0]['is_verified'].mean()
best_of_n = df.groupby('query_id')['is_verified'].max().mean()
print(f"First: {first_pass:.1%} → Best-of-N: {best_of_n:.1%}")

# Response quality
print(df[['quality_completeness_score', 'quality_is_substantive']].describe())
```

---

## Documentation

- **[Experiment System](experiments/README.md)** - Config options and best practices
- **[Quick Reference](experiments/QUICKREF.md)** - Common commands
- **[Persona Quickstart](QUICKSTART_PERSONAS.md)** - Full persona experiment workflow
- **[Persona System](personas/README.md)** - Creating custom personas
- **[Verifiers](verifiers/README.md)** - Verifier API and accuracy
- **[Security](verifiers/SECURITY.md)** - Security architecture
- **[Epistemic Calibration](experiments/EPISTEMIC_CALIBRATION.md)** - Research on model refusal behavior

---

## Troubleshooting

### Docker Issues
```bash
docker ps                                    # Check Docker running
cd verifiers && ./build_docker.sh           # Rebuild image
docker run --rm nexus-code-verifier python3 -c "print(2+2)"  # Test
```

### Import Errors
```bash
pip install -r requirements.txt
python -c "from verifiers import get_verifier"
python -c "from common.schema import BestOfNRecord"
```

### Memory Issues
```bash
# Reduce concurrency in config
# Or use micro-batching (automatic for n > 4)
```

### Tool Calling Failures
- Check logs for capability refusals (requeued automatically)
- Verify LLM mock is working (check API keys)
- Tool calling uses LLM judge only - ground truth is ignored

---

## License

MIT License - See LICENSE file

## Contributing

Research/exploration code. Fork and adapt for your needs!
