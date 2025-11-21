# Experiment Configuration System

Organize and track Best-of-N experiments with YAML configs and parquet metadata.

## Quick Start

```bash
# Run experiment from config
python generate_best_of_n.py --config experiments/baseline.yaml

# Override specific parameters
python generate_best_of_n.py \
    --config experiments/baseline.yaml \
    --model gpt-4o \
    --max-queries 200

# Inspect results
python inspect_experiment.py experiments/results/baseline_run.parquet

# Compare multiple runs
python inspect_experiment.py experiments/results/*.parquet
```

## Config File Structure

```yaml
# Dataset configuration
dataset: nvidia/Nemotron-Post-Training-Dataset-v1
splits: math,code,tool_calling
streaming: true
max_queries: 100

# Generation parameters
model: gpt-4o-mini
num_candidates: 4
temperature: 0.7
max_tokens: 2048

# Performance
concurrency: 10

# Output
output: experiments/results/my_experiment.parquet

# Experiment notes (saved in parquet metadata)
notes: |
  Multi-line notes about this experiment.
  What you're testing, hypotheses, etc.
```

## Config Priority

1. **Config file defaults** (lowest priority)
2. **CLI arguments** (highest priority)

Example:
```bash
# Config says max_queries: 100
# CLI overrides to 50
python generate_best_of_n.py \
    --config experiments/baseline.yaml \
    --max-queries 50  # This wins
```

## Provided Experiments

### `baseline.yaml`
Quick baseline run on all splits with minimal queries.
- **Purpose**: Establish baseline metrics
- **Model**: gpt-4o-mini (cheap)
- **Queries**: 100 per split
- **Use case**: Quick validation, debugging

### `math_focused.yaml`
Deep dive into math verification with more candidates.
- **Purpose**: Study verification distribution
- **Model**: gpt-4o (better quality)
- **Queries**: 1000 math problems
- **Candidates**: N=10 (study marginal value)
- **Use case**: Analyze "does first win?" and "what's optimal N?"

### `high_throughput.yaml`
Production-scale data generation.
- **Purpose**: Generate training data
- **Model**: gpt-4o-mini (cost-effective)
- **Queries**: 10K per split (20K total)
- **Concurrency**: 50 (fast)
- **Use case**: Reward model training data

## Creating Custom Experiments

```yaml
# experiments/my_experiment.yaml
dataset: nvidia/Nemotron-Post-Training-Dataset-v1
splits: code  # Focus on one split
max_queries: 500
model: gpt-4o
num_candidates: 8  # Try different N
temperature: 0.9  # Higher diversity
output: experiments/results/my_experiment.parquet
notes: |
  Testing hypothesis: Higher temperature improves
  code generation diversity and verification rates.
```

Then run:
```bash
python generate_best_of_n.py --config experiments/my_experiment.yaml
```

## Inspecting Results

### Single Experiment
```bash
python inspect_experiment.py experiments/results/baseline_run.parquet
```

Output shows:
- Experiment metadata (model, N, temperature, config notes)
- Verification rates overall and per-split
- Candidate index distribution (does first win?)
- Query coverage

### Comparing Experiments
```bash
python inspect_experiment.py \
    experiments/results/baseline_run.parquet \
    experiments/results/math_n10.parquet
```

Side-by-side comparison of:
- Models used
- Verification rates
- First candidate success rates

## Parquet Metadata

All experiments save metadata in the parquet file:
```python
import pyarrow.parquet as pq

parquet_file = pq.read_table('results.parquet')
metadata = parquet_file.schema.metadata

print(metadata[b'model'].decode())  # Model name
print(metadata[b'notes'].decode())  # Experiment notes
```

Metadata includes:
- `generated_at`: Timestamp
- `model`: Model name
- `num_candidates`: N value
- `temperature`: Temperature used
- `splits`: Splits processed
- `total_records`: Total rows
- `config_file`: Config path used
- `notes`: Experiment notes from YAML

## Best Practices

### 1. Always Add Notes
```yaml
notes: |
  What: Testing math verification accuracy
  Why: Previous runs showed 60% false negatives
  Hypothesis: SymPy verifier will improve to 95%+
  Expected outcome: Verification rate >90%
```

### 2. Use Descriptive Filenames
```yaml
output: experiments/results/math_sympy_n4_temp07_20241120.parquet
```

### 3. Start Small, Scale Up
```bash
# First: Quick test with 10 queries
python generate_best_of_n.py \
    --config experiments/baseline.yaml \
    --max-queries 10

# Then: Full run
python generate_best_of_n.py \
    --config experiments/baseline.yaml
```

### 4. Version Your Configs
```bash
git add experiments/my_experiment_v1.yaml
git commit -m "Experiment: baseline math verification"
```

### 5. Keep Results Organized
```
experiments/
├── baseline.yaml
├── math_focused.yaml
├── high_throughput.yaml
└── results/
    ├── baseline_20241120.parquet
    ├── math_n10_20241120.parquet
    └── production_20k_20241120.parquet
```

## Analyzing Results

### Load into pandas
```python
import pandas as pd
df = pd.read_parquet('experiments/results/baseline_run.parquet')

# Verification rate by split
df.groupby('split')['is_verified'].mean()

# First candidate vs best candidate
first_wins = df[df.candidate_idx == 0]['is_verified'].mean()
any_wins = df.groupby('query_id')['is_verified'].max().mean()

print(f"First: {first_wins:.2%}, Best: {any_wins:.2%}")
```

### Compare experiments
```python
baseline = pd.read_parquet('experiments/results/baseline.parquet')
improved = pd.read_parquet('experiments/results/improved.parquet')

# Join on same queries
comparison = baseline.merge(
    improved,
    on='query_id',
    suffixes=['_baseline', '_improved']
)

# Where did improvement help?
comparison['improved'] = (
    comparison['is_verified_improved'] >
    comparison['is_verified_baseline']
)
print(comparison.groupby('split_baseline')['improved'].mean())
```

## Tips for Exploration

1. **Run small experiments frequently** - 100 queries × 4 candidates = 400 API calls ≈ $4
2. **Save everything** - Disk is cheap, re-running expensive
3. **Track your intuitions in notes** - Future you will thank you
4. **Compare side-by-side** - "Did that change actually help?"
5. **Look for patterns** - Which splits are hard? Where do verifiers fail?

## Cost Estimation

Rough costs (at $0.01/call):
- **baseline.yaml**: 100 queries × 3 splits × 4 candidates = 1,200 calls ≈ $12
- **math_focused.yaml**: 1,000 queries × 1 split × 10 candidates = 10,000 calls ≈ $100
- **high_throughput.yaml**: 10,000 queries × 2 splits × 4 candidates = 80,000 calls ≈ $800

Always start small!
