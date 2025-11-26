# Best-of-N Candidate Generation with Verification

Generate and verify multiple candidate responses for training data, with secure Docker-based code execution and sophisticated verification systems. Supports both OpenAI and Claude APIs.

## Overview

Best-of-N generation creates multiple candidate responses per query and verifies them using domain-specific verifiers (math, code, tool-calling). Useful for:
- **RLHF/DPO training data** - Ranked candidates with quality scores
- **Reward model training** - Verified vs unverified examples
- **Verification research** - Study verifier accuracy and failure modes
- **Personality transfer** - Fine-tune models on persona-injected data

## Features

### 🔌 Multi-Provider Support
- **OpenAI API** - GPT-5.1 with structured outputs and Responses API
- **Claude API** - Claude 4.5 (Sonnet/Opus) with extended thinking mode
- **Local Models** - OpenAI-compatible server for custom models

### 🔒 Secure Verification
- **Docker-isolated code execution** - No exec/eval in main process
- **Multi-language support** - Python, JavaScript, Bash, SQL
- **Resource limits** - CPU, memory, timeout enforcement
- **Production-grade security** - See [verifiers/SECURITY.md](verifiers/SECURITY.md)

### 🎯 High-Accuracy Verifiers
- **MathVerifier** (97%) - SymPy symbolic + unit-aware + numeric
- **CodeVerifier** (94%) - Docker sandbox with test case execution
- **ToolVerifier** (98%) - JSON Schema validation + 100+ mock implementations
- **LLM Judge** - Fallback using same provider (GPT-4o for OpenAI, Sonnet 4.5 for Claude)

### 🎭 Persona System
- **Flexible templates** - Marvin, Data, Johnny 5 included
- **Personality transfer** - Test distillation of distinctive voices
- **Custom personas** - Create your own styles

### ⚡ Performance
- **Async generation** - Concurrent API calls with rate limiting
- **Micro-batching** - Prevents OOM for large n values
- **Surgical regeneration** - Retry only failed rows
- **Checkpointing** - Automatic saves every N queries

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
python bestofn.py openai generate --config experiments/marvin_100x8.yaml

# Generate with Claude
python bestofn.py claude generate --config experiments/marvin_claude_100x8.yaml

# Inspect results
python inspect_experiment.py experiments/results/marvin_100x8.parquet
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
│   └── llm_judge.py           # LLM-as-judge (Claude Sonnet 4.5)
│
├── verifiers/                 # Verification system
│   ├── math_verifier.py       # SymPy-based math verification
│   ├── code_verifier.py       # Docker sandbox execution
│   ├── tool_verifier.py       # JSON Schema validation
│   ├── tool_sandbox.py        # 100+ mock tool implementations
│   ├── docker_sandbox.py      # Container management
│   └── refusal_classifier.py  # Refusal detection
│
├── experiments/               # Experiment configs
│   ├── *.yaml                 # Config files
│   └── results/               # Output parquet files
│
└── personas/                  # Personality templates
    ├── marvin.txt             # Marvin the Paranoid Android
    ├── marvin_flexible.txt    # Flexible Marvin (high diversity)
    ├── data.txt               # Lt. Cmd. Data
    └── johnny5_flexible.txt   # Johnny 5
```

## Configuration

Experiments are defined in YAML:

```yaml
# experiments/my_experiment.yaml
dataset:
  source: nemotron
  split: math_code
  max_queries: 100

generation:
  model: gpt-5.1
  n: 8
  temperature: 0.7
  persona: personas/marvin_flexible.txt

output:
  path: experiments/results/my_run.parquet
  checkpoint_every: 25
```

## Persona Experiments

Generate training data with distinctive personalities:

```bash
# Marvin - depressed robot
python bestofn.py openai generate --config experiments/marvin_100x8.yaml

# Data - precise android
python bestofn.py openai generate --config experiments/data_100x8.yaml

# Johnny 5 - enthusiastic robot
python bestofn.py openai generate --config experiments/j5_100x8.yaml
```

Flexible persona variants (`*_flexible.txt`) produce higher diversity responses while maintaining character voice.

## Tool Calling

The framework includes 100+ mock tool implementations for realistic tool-calling experiments:

- **Weather, stocks, calculations** - Common API patterns
- **Database queries** - SQL-like interfaces
- **File operations** - Read/write simulations
- **Web search** - Mock search results

Tools are executed in a sandboxed environment with deterministic outputs for reproducibility.

## Local Inference

For development and custom models:

```bash
# Terminal 1: Start server
python local_server.py --model /path/to/model --port 8000

# Terminal 2: Generate
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
python bestofn.py openai generate --config experiments/baseline.yaml
```

Features:
- GPU locking (prevents concurrent access)
- OOM handling with automatic recovery
- Per-request performance logging

## Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_parquet('experiments/results/marvin_100x8.parquet')

# Verification rate by split
print(df.groupby('split')['is_verified'].mean())

# Best-of-N improvement
first_pass = df[df.candidate_idx == 0]['is_verified'].mean()
best_of_n = df.groupby('query_id')['is_verified'].max().mean()
print(f"First: {first_pass:.1%} → Best-of-N: {best_of_n:.1%}")
```

## Documentation

- **[Experiment System](experiments/README.md)** - Config options and best practices
- **[Quick Reference](experiments/QUICKREF.md)** - Common commands
- **[Persona System](personas/README.md)** - Creating custom personas
- **[Verifiers](verifiers/README.md)** - Verifier API and accuracy
- **[Security](verifiers/SECURITY.md)** - Security architecture
- **[Schema](SCHEMA.md)** - Data format documentation
- **[Harmony Format](HARMONY.md)** - Message encoding format

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

## License

MIT License - See LICENSE file

## Contributing

Research/exploration code. Fork and adapt for your needs!
