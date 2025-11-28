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

# With LLM judge fallback for low-confidence results
python -m claude_gen.generate --config experiments/marvin/claude_100x8.yaml --llm-judge-fallback
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

### Tool Calling Only
```bash
python -m claude_gen.generate \
    --config experiments/j5/claude_100x8.yaml \
    --splits tool_calling \
    --max-queries 50
```

---

## Verification by Split

| Split | Primary Verifier | LLM Judge | Ground Truth |
|-------|------------------|-----------|--------------|
| `math` | MathVerifier (SymPy) | On flag + low conf | Nemotron |
| `code` | CodeVerifier (Docker) | Auto on low conf | Nemotron |
| `tool_calling` | **Skipped** | **Mandatory** | **Ignored** |

**Why tool_calling is different:** Nemotron ground truth for tool_calling was generated without reasoning, making it unreliable. LLM judge evaluates appropriateness instead.

---

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
llm_judge_fallback: true  # Enable for math low-confidence fallback
notes: |
  What you're testing and why.
```

## Model Options

### Claude
- `claude-sonnet-4-5-20250929` - Standard (recommended)
- `claude-opus-4-5-20251101` - Complex reasoning

### OpenAI
- `gpt-4o` - Standard
- `gpt-4o-mini` - Fast/cheap

## Persona Files

| Persona | Affect | File |
|---------|--------|------|
| Marvin | Negative | `personas/marvin_flexible.txt` |
| Data | Neutral | `personas/data_flexible.txt` |
| Johnny 5 | Positive | `personas/johnny5_flexible.txt` |

---

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

# Tool calling specific (LLM judge results)
tool_df = df[df['split'] == 'tool_calling']
print(f"Tool calling: {tool_df['is_verified'].mean():.2%}")
```

## Read Metadata

```python
import pyarrow.parquet as pq

pf = pq.read_table('experiments/marvin/results/claude_100x8.parquet')
meta = pf.schema.metadata

print(meta[b'model'].decode())  # Model used
print(meta[b'notes'].decode())  # Experiment notes
```

---

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
│   ├── nemotron_utils.py       # Dataset utilities
│   └── llm_judge.py            # LLM-as-judge verification
├── verifiers/                  # Verification system
│   ├── math_verifier.py        # SymPy math verification
│   ├── code_verifier.py        # Docker code execution
│   └── tool_sandbox.py         # Platform-aware tool mocking
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

---

## Troubleshooting

### Low tool_calling verification
- This is expected - LLM judge evaluates appropriateness, not exact match
- Check `verification_explanation` field for details

### Capability refusals in tool_calling
- Model refused due to mock limitations
- Check logs for "capability refusal" messages
- These are NOT safety refusals

### High variance between runs
- Increase `num_candidates` for more samples
- Lower `temperature` for more consistent results
- Tool calling naturally has higher variance
