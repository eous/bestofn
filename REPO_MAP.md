# Best-of-N Repository Map

A modular framework for generating Best-of-N samples with verification using OpenAI and Claude APIs.

## Repository Structure

```
bestofn/
├── bestofn.py              # Unified CLI entry point
├── local_server.py         # Local generation server
├── setup.py                # Package setup
├── inspect_experiment.py   # Results inspection tool
├── generate_control_samples.py  # Control sample generation
│
├── openai_gen/             # OpenAI-specific generation
│   ├── __init__.py
│   ├── generate.py         # Main generation script
│   ├── regen.py            # Regeneration for failed rows
│   └── tool_executor.py    # Tool calling execution loop
│
├── claude_gen/             # Claude-specific generation
│   ├── __init__.py
│   ├── generate.py         # Main generation script
│   ├── regen.py            # Regeneration for failed rows
│   └── tool_executor.py    # Tool calling execution
│
├── common/                 # Shared utilities
│   ├── __init__.py
│   ├── schema.py           # Pydantic schemas
│   ├── nemotron_utils.py   # Dataset loading
│   ├── ast_syntax_checker.py  # AST validation
│   ├── generation_utils.py # Shared generation helpers
│   ├── llm_judge.py        # LLM-as-judge (GPT-4o or Sonnet 4.5)
│   ├── api_retry.py        # API resilience with exponential backoff
│   ├── response_validation.py  # Output truncation and safety
│   ├── quality_metrics.py  # Response quality assessment
│   ├── refusal_check.py    # Two-pass refusal detection
│   ├── llm_judge_fallback.py   # Fallback verification strategy
│   └── regen_pipeline.py   # Shared regeneration utilities
│
├── verifiers/              # Verification system
│   ├── __init__.py         # Verifier factory
│   ├── base.py             # Base verifier class
│   ├── math_verifier.py    # Math verification (SymPy)
│   ├── code_verifier.py    # Code verification (Docker)
│   ├── tool_verifier.py    # Tool call verification
│   ├── docker_sandbox.py   # Docker container manager
│   ├── refusal_classifier.py  # Refusal detection
│   ├── tool_sandbox.py     # 100+ mock tool implementations
│   ├── Dockerfile          # Code execution sandbox
│   └── build_docker.sh     # Docker build script
│
├── experiments/            # Experiment configurations
│   ├── README.md           # Full documentation
│   ├── QUICKREF.md         # Quick reference
│   ├── marvin/             # Marvin persona experiments
│   │   ├── openai_100x8.yaml
│   │   ├── claude_100x8.yaml
│   │   └── results/
│   ├── data/               # Data persona experiments
│   │   ├── openai_100x8.yaml
│   │   ├── claude_100x8.yaml
│   │   └── results/
│   ├── j5/                 # Johnny 5 persona experiments
│   │   ├── openai_100x8.yaml
│   │   ├── claude_100x8.yaml
│   │   └── results/
│   └── baseline/           # Non-persona baseline experiments
│       ├── baseline.yaml
│       └── results/
│
├── personas/               # Persona definitions
│   ├── README.md
│   ├── marvin_flexible.txt # Marvin the Paranoid Android (negative affect)
│   ├── data_flexible.txt   # Lt. Cmd. Data (neutral affect)
│   └── johnny5_flexible.txt # Johnny 5 (positive affect)
│
├── HARMONY.md              # Harmony format spec (multi-channel outputs)
│
└── docs/                   # Documentation
    └── README.md           # Main readme
```

## CLI Usage

```bash
# Generate with OpenAI
python bestofn.py openai generate --config experiments/config.yaml

# Generate with Claude
python bestofn.py claude generate --config experiments/config.yaml

# Regenerate failed rows (OpenAI)
python bestofn.py openai regen results.parquet --split tool_calling --failed-only

# Regenerate failed rows (Claude)
python bestofn.py claude regen results.parquet --split tool_calling --failed-only

# Show help
python bestofn.py --help
```

## Core Modules

### OpenAI Generation (`openai_gen/`)

| Module | Description |
|--------|-------------|
| `generate.py` | Main generation using OpenAI Responses API with structured outputs |
| `regen.py` | Surgical regeneration for specific splits or failed rows |
| `tool_executor.py` | Multi-turn tool calling with 100+ mock implementations |

### Claude Generation (`claude_gen/`)

| Module | Description |
|--------|-------------|
| `generate.py` | Generation using Claude API with extended thinking mode |
| `regen.py` | Regeneration script adapted for Claude |
| `tool_executor.py` | Tool calling adapted for Claude's tool_use format |

### Common Utilities (`common/`)

| Module | Description |
|--------|-------------|
| `schema.py` | Pydantic schemas (BestOfNRecord, ModelOutput, ReasoningStep) |
| `nemotron_utils.py` | Dataset loading helpers (v1/v2 detection) |
| `ast_syntax_checker.py` | Fast AST validation for Python/JavaScript |
| `generation_utils.py` | Shared extraction, formatting, and logging utilities |
| `llm_judge.py` | LLM-as-judge verification (dual-provider: GPT-4o or Sonnet 4.5) |
| `api_retry.py` | API resilience with exponential backoff and jitter |
| `response_validation.py` | Response truncation and safety limits |
| `quality_metrics.py` | Response quality metrics computation |
| `refusal_check.py` | Two-pass refusal detection |
| `llm_judge_fallback.py` | Fallback verification when primary has low confidence |
| `regen_pipeline.py` | Shared utilities for regenerating failed candidates |

See [COMMON_UTILITIES.md](COMMON_UTILITIES.md) for detailed usage documentation.

### Verification System (`verifiers/`)

| Module | Description |
|--------|-------------|
| `math_verifier.py` | Math verification with SymPy symbolic + numeric + LaTeX |
| `code_verifier.py` | Code verification via Docker sandbox execution |
| `tool_verifier.py` | Tool call JSON schema validation |
| `docker_sandbox.py` | Docker container manager for code execution |
| `refusal_classifier.py` | Pattern-based refusal detection |
| `tool_sandbox.py` | 100+ mock tool implementations for testing |

## Configuration

Experiments are defined in YAML files:

```yaml
dataset:
  source: nemotron
  split: math_code

generation:
  model: gpt-4o
  n: 8
  persona: personas/marvin_flexible.txt

output:
  path: experiments/results/output.parquet
  checkpoint_every: 25
```

## Docker Sandbox

The code verifier uses a Docker sandbox for safe code execution:

```bash
# Build the sandbox
cd verifiers && ./build_docker.sh

# Image: nexus-code-verifier:latest
# Supports: Python, Node.js, SQLite, Bash
```

## Key Features

- **Structured Outputs**: Enforced JSON schema for reasoning steps
- **Multi-Provider**: OpenAI and Claude API support
- **Verification Pipeline**: Math, code, and tool call verification
- **Persona System**: Customizable model personas
- **Checkpointing**: Automatic save every N queries
- **Regeneration**: Surgical retry of failed rows only
- **Tool Calling**: 100+ mock tool implementations
