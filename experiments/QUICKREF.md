# Quick Reference: Experiment Config System

## Directory Structure

```
experiments/
├── marvin/     # Marvin (negative affect) - results/
├── data/       # Data (neutral affect) - results/
├── j5/         # Johnny 5 (positive affect) - results/
└── baseline/   # Non-persona baseline - results/
```

## Run Experiments

```bash
# From config file (Claude - recommended for personas)
python -m claude_gen.generate --config experiments/marvin/claude_100x8.yaml

# From config file (OpenAI)
python -m openai_gen.generate --config experiments/j5/openai_100x8.yaml

# Override config values
python -m claude_gen.generate --config experiments/data/claude_100x8.yaml --max-queries 50

# Traditional CLI (no config)
python -m claude_gen.generate --model claude-sonnet-4-5-20250929 --splits math --max-queries 100
```

## Inspect Results

```bash
# Single experiment details
python inspect_experiment.py experiments/marvin/results/claude_100x8.parquet

# Compare multiple experiments
python inspect_experiment.py experiments/*/results/*.parquet
```

## Common Patterns

### Quick Test (10 queries)
```bash
python -m claude_gen.generate \
    --config experiments/marvin/claude_100x8.yaml \
    --max-queries 10 \
    --output experiments/marvin/results/test_run.parquet
```

### Compare Claude vs OpenAI for same persona
```bash
python -m claude_gen.generate --config experiments/j5/claude_100x8.yaml
python -m openai_gen.generate --config experiments/j5/openai_100x8.yaml
python inspect_experiment.py experiments/j5/results/*.parquet
```

### Production Run (200 queries x 8 candidates)
```bash
python -m claude_gen.generate --config experiments/marvin/claude_100x8.yaml
```

## Config Template

```yaml
dataset: nvidia/Nemotron-Post-Training-Dataset-v2
splits: math,code,tool_calling
streaming: true
max_queries: 200
model: claude-sonnet-4-5-20250929
num_candidates: 8
temperature: 1.0
max_tokens: 16384
persona: personas/marvin_flexible.txt
concurrency: 3
output: experiments/marvin/results/my_run.parquet
generator: claude
structured_output: true
notes: |
  What you're testing and why.
```

## Persona Files

| Persona | Affect | File |
|---------|--------|------|
| Marvin | Negative | `personas/marvin_flexible.txt` |
| Data | Neutral | `personas/data_flexible.txt` |
| Johnny 5 | Positive | `personas/johnny5_flexible.txt` |

## Analyze Results in Python

```python
import pandas as pd

# Load results
df = pd.read_parquet('experiments/marvin/results/claude_100x8.parquet')

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

pf = pq.read_table('experiments/marvin/results/claude_100x8.parquet')
meta = pf.schema.metadata

print(meta[b'model'].decode())  # Model used
print(meta[b'notes'].decode())  # Experiment notes
```

## File Organization

```
bestofn/
├── openai_gen/                 # OpenAI generation
│   ├── generate.py             # Main generation script
│   └── tool_executor.py        # Tool execution loop
├── claude_gen/                 # Claude generation
│   ├── generate.py             # Main generation script
│   └── tool_executor.py        # Tool execution loop
├── common/                     # Shared utilities
│   └── nemotron_utils.py       # Dataset utilities
├── personas/                   # Persona definition files
│   ├── marvin_flexible.txt
│   ├── data_flexible.txt
│   └── johnny5_flexible.txt
├── inspect_experiment.py       # Results inspector
└── experiments/
    ├── marvin/                 # Marvin experiments + results/
    ├── data/                   # Data experiments + results/
    ├── j5/                     # Johnny 5 experiments + results/
    ├── baseline/               # Baseline experiments + results/
    ├── README.md               # Full documentation
    └── QUICKREF.md             # This file
```
