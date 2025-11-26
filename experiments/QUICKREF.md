# Quick Reference: Experiment Config System

## Run Experiments

```bash
# From config file (OpenAI)
python -m openai_gen.generate --config experiments/baseline.yaml

# From config file (Claude)
python -m claude_gen.generate --config experiments/baseline.yaml

# Override config values
python -m openai_gen.generate --config experiments/baseline.yaml --max-queries 50

# Traditional CLI (no config)
python -m openai_gen.generate --model gpt-4o --splits math --max-queries 100
```

## Inspect Results

```bash
# Single experiment details
python inspect_experiment.py experiments/results/baseline_run.parquet

# Compare multiple experiments
python inspect_experiment.py experiments/results/*.parquet
```

## Common Patterns

### Quick Test (10 queries, $0.40)
```bash
python -m openai_gen.generate \
    --config experiments/baseline.yaml \
    --max-queries 10 \
    --output test_run.parquet
```

### Math Only (1K queries, $40)
```bash
python -m openai_gen.generate \
    --config experiments/math_focused.yaml
```

### Production Run (20K queries, $800)
```bash
python -m openai_gen.generate \
    --config experiments/high_throughput.yaml
```

## Config Template

```yaml
dataset: nvidia/Nemotron-Post-Training-Dataset-v1
splits: math,code,tool_calling
streaming: true
max_queries: 100
model: gpt-4o-mini
num_candidates: 4
temperature: 0.7
max_tokens: 131072
concurrency: 10
output: experiments/results/my_run.parquet
notes: |
  What you're testing and why.
```

## Analyze Results in Python

```python
import pandas as pd

# Load results
df = pd.read_parquet('experiments/results/baseline_run.parquet')

# Verification rate
df['is_verified'].mean()

# By split
df.groupby('split')['is_verified'].mean()

# First candidate vs any candidate
first = df[df.candidate_idx == 0]['is_verified'].mean()
any_verified = df.groupby('query_id')['is_verified'].max().mean()
```

## Read Metadata

```python
import pyarrow.parquet as pq

pf = pq.read_table('experiments/results/baseline_run.parquet')
meta = pf.schema.metadata

print(meta[b'model'].decode())  # Model used
print(meta[b'notes'].decode())  # Experiment notes
```

## File Organization

```
bestofn/
├── openai_gen/                 # OpenAI generation
│   ├── generate.py             # Main generation script
│   ├── regen.py                # Regeneration tool
│   └── tool_executor.py        # Tool execution loop
├── claude_gen/                 # Claude generation
│   ├── generate.py             # Main generation script
│   ├── regen.py                # Regeneration tool
│   └── tool_executor.py        # Tool execution loop
├── common/                     # Shared utilities
│   └── nemotron_utils.py       # Dataset utilities
├── inspect_experiment.py       # Results inspector
└── experiments/
    ├── baseline.yaml           # Quick test config
    ├── math_focused.yaml       # Math deep dive
    ├── high_throughput.yaml    # Production scale
    ├── README.md               # Full documentation
    └── results/                # Generated .parquet files
        └── .gitignore          # Don't commit results
```
