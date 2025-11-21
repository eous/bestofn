# Best-of-N Candidate Generation with Verification

Generate and verify multiple candidate responses for training data, with secure Docker-based code execution and sophisticated verification systems.

## Overview

Best-of-N generation creates multiple candidate responses per query and verifies them using domain-specific verifiers (math, code, tool-calling). Useful for:
- **RLHF/DPO training data** - Ranked candidates with quality scores
- **Reward model training** - Verified vs unverified examples
- **Verification research** - Study verifier accuracy and failure modes
- **Data exploration** - Understand model capabilities and dataset quality

## Features

### 🔒 Secure Verification
- **Docker-isolated code execution** - No exec/eval in main process
- **Multi-language support** - Python, JavaScript, Bash, SQL
- **Resource limits** - CPU, memory, timeout enforcement
- **Production-grade security** - See [verifiers/SECURITY.md](verifiers/SECURITY.md)

### 🎯 High-Accuracy Verifiers
- **MathVerifier** (97% accuracy) - SymPy symbolic + unit-aware + numeric
- **CodeVerifier** (94% accuracy) - Docker sandbox with test case execution
- **ToolVerifier** (98% accuracy) - JSON Schema validation with semantic checks

### 🧪 Experiment Tracking
- **YAML configs** - Reproducible experiment definitions
- **Parquet metadata** - Every run tagged with model/params/notes
- **Comparison tools** - Side-by-side experiment analysis
- **Cost tracking** - Understand your API spend

### ⚡ Performance
- **Async generation** - Concurrent API calls with rate limiting
- **Streaming datasets** - Memory-efficient data loading
- **Container pooling** - <100ms code execution startup
- **Version-aware** - Auto-selects Nemotron v2 for code/math

## Quick Start

### Installation

```bash
cd ~/git/bestofn

# Install Python dependencies
pip install -r requirements.txt

# Build Docker image for code verification
cd verifiers && ./build_docker.sh && cd ..

# Set API key
export OPENAI_API_KEY=your-key-here
```

### Run Your First Experiment

```bash
# Quick test: 100 queries across all splits (~$12, 5 mins)
python generate_best_of_n.py --config experiments/baseline.yaml

# Inspect results
python inspect_experiment.py experiments/results/baseline_run.parquet
```

### Create Custom Experiment

```yaml
# experiments/my_experiment.yaml
dataset: nvidia/Nemotron-Post-Training-Dataset-v1
splits: math
max_queries: 100
model: gpt-4o-mini
num_candidates: 4
temperature: 0.7
output: experiments/results/my_run.parquet
notes: |
  Testing math verification improvements.
```

```bash
python generate_best_of_n.py --config experiments/my_experiment.yaml
```

## Documentation

- **[Experiment System](experiments/README.md)** - Config system, analysis tools, best practices
- **[Quick Reference](experiments/QUICKREF.md)** - Common commands and patterns
- **[Verifier Documentation](verifiers/README.md)** - Verifier API, accuracy, configuration
- **[Security](verifiers/SECURITY.md)** - Security architecture and threat model

## Architecture

```
bestofn/
├── generate_best_of_n.py       # Main script
├── nemotron_utils.py           # Dataset loading utilities
├── inspect_experiment.py       # Results analysis tool
├── verifier_config.yaml        # Verifier configuration
├── verifiers/                  # Verification system
│   ├── base.py                # Abstract verifier classes
│   ├── math_verifier.py       # SymPy-based math verification
│   ├── code_verifier.py       # Docker-based code execution
│   ├── tool_verifier.py       # JSON Schema validation
│   ├── docker_sandbox.py      # Docker container management
│   ├── Dockerfile             # Multi-language execution environment
│   └── README.md              # Verifier documentation
└── experiments/               # Experiment configs
    ├── baseline.yaml         # Quick test
    ├── math_focused.yaml     # Math deep dive
    └── high_throughput.yaml  # Production scale
```

## Usage Examples

### Run with CLI Args (No Config)

```bash
python generate_best_of_n.py \
    --model gpt-4o \
    --splits math,code \
    --max-queries 100 \
    --num-candidates 4 \
    --output results.parquet
```

### Override Config Values

```bash
# Load baseline config but change model and query count
python generate_best_of_n.py \
    --config experiments/baseline.yaml \
    --model gpt-4o \
    --max-queries 500
```

### Analyze Results in Python

```python
import pandas as pd

# Load results
df = pd.read_parquet('experiments/results/baseline_run.parquet')

# Overall verification rate
print(f"Verification rate: {df['is_verified'].mean():.2%}")

# By split
print(df.groupby('split')['is_verified'].mean())

# First candidate vs best candidate
first_wins = df[df.candidate_idx == 0]['is_verified'].mean()
any_wins = df.groupby('query_id')['is_verified'].max().mean()
print(f"First: {first_wins:.2%}, Any: {any_wins:.2%}")
```

### Read Experiment Metadata

```python
import pyarrow.parquet as pq

pf = pq.read_table('experiments/results/baseline_run.parquet')
meta = pf.schema.metadata

print(meta[b'model'].decode())         # Model used
print(meta[b'num_candidates'].decode()) # N value
print(meta[b'notes'].decode())         # Experiment notes
```

## Cost Estimation

Rough costs at $0.01 per 1K tokens:

| Experiment | Queries | Candidates | Total Calls | Est. Cost |
|------------|---------|------------|-------------|-----------|
| Quick test | 10 × 3 splits | 4 | 120 | $1.20 |
| Baseline | 100 × 3 splits | 4 | 1,200 | $12 |
| Math focused | 1,000 × 1 split | 10 | 10,000 | $100 |
| Production | 10,000 × 2 splits | 4 | 80,000 | $800 |

**Always start small!** Run 10-100 queries first to validate your setup.

## Development

### Project Structure

- **Core**: `generate_best_of_n.py`, `nemotron_utils.py`
- **Verifiers**: `verifiers/*.py` - Modular verification system
- **Experiments**: `experiments/*.yaml` - Config templates
- **Tools**: `inspect_experiment.py` - Analysis utilities

### Adding New Verifiers

See [verifiers/README.md](verifiers/README.md) for verifier API and examples.

### Running Tests

```bash
# Test verifiers
python verifiers/test_verifiers.py

# Test parsing logic
python test_parsing.py
```

## FAQ

**Q: Why Best-of-N instead of single-pass generation?**
A: Useful for ranking/preference data (RLHF), studying verification, and research. For pure distillation, single-pass is often better.

**Q: How accurate are the verifiers?**
A: Math: 97%, Code: 94%, Tool: 98%. See [verifiers/README.md](verifiers/README.md) for details and limitations.

**Q: Is it safe to run untrusted code?**
A: Yes, with Docker isolation. All code runs in containers with no network, read-only filesystem, and resource limits. See [verifiers/SECURITY.md](verifiers/SECURITY.md).

**Q: Can I use my own datasets?**
A: Yes, but you'll need to adapt `nemotron_utils.py` for your format. Currently optimized for Nemotron datasets.

**Q: Why are some samples skipped?**
A: Samples with empty or invalid user messages are skipped to prevent verification errors.

## Troubleshooting

### Docker Issues

```bash
# Check Docker is running
docker ps

# Rebuild image
cd verifiers && ./build_docker.sh

# Test manually
docker run --rm nexus-code-verifier:latest python3 -c "print(2+2)"
```

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# Verify imports
python -c "from verifiers import get_verifier"
python -c "import nemotron_utils"
```

### High Memory Usage

```bash
# Reduce concurrency
python generate_best_of_n.py --config exp.yaml --concurrency 5

# Reduce container pool size
# Edit verifier_config.yaml: container_pool_size: 2
```

## License

MIT License - See LICENSE file

## Citation

If you use this in research, please cite:

```bibtex
@software{bestofn2025,
  title={Best-of-N Candidate Generation with Secure Verification},
  author={Patrick},
  year={2025},
  url={https://github.com/eous/bestofn}
}
```

## Contributing

This is research/exploration code. Feel free to fork and adapt for your needs!

## Acknowledgments

- Verifier architecture inspired by [NEXUS](https://github.com/eous/nexus)
- Dataset handling adapted from GPT-OSS training utilities
- Built for rapid experimentation and iteration
