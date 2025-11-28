# Quick Start: Persona Transfer Experiments

Complete commands to run both Marvin and Data experiments (100 queries × 8 candidates each).

## Prerequisites (One-Time Setup)

```bash
cd ~/git/bestofn

# Install dependencies (in venv if needed)
pip install -r requirements.txt

# Build Docker image for verifiers
cd verifiers && ./build_docker.sh && cd ..

# Verify installation
python -c "from common.schema import BestOfNRecord; print('Schema OK')"
python -c "from verifiers import get_verifier; print('Verifiers OK')"
```

## Full Workflow (Both Personas)

### Step 1: Start Local Server

```bash
# Terminal 1: Start inference server with your NEXUS model
python local_server.py \
    --model /path/to/gpt-oss-120b-nexus \
    --port 8000 \
    --verbose

# Wait for:
# ✓ Model loaded successfully
# Server running on http://0.0.0.0:8000
```

### Step 2: Generate Marvin Dataset (100 × 8 = 800 samples)

```bash
# Terminal 2: Set environment
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy

# Generate Marvin dataset
python bestofn.py openai generate \
    --config experiments/marvin/openai_100x8.yaml

# Expected: ~1-2 hours on local GPU
# Output: experiments/marvin/results/openai_100x8.parquet
```

### Step 3: Generate Data Dataset (100 × 8 = 800 samples)

```bash
# Same terminal, local server still running

# Generate Data dataset
python bestofn.py openai generate \
    --config experiments/data/personality.yaml

# Expected: ~1-2 hours on local GPU
# Output: experiments/data/results/openai_100x8.parquet
```

### Step 4: Inspect Both Datasets

```bash
# Marvin inspection
python inspect_experiment.py experiments/marvin/results/openai_100x8.parquet

# Data inspection
python inspect_experiment.py experiments/data/results/openai_100x8.parquet

# Compare side-by-side
python inspect_experiment.py \
    experiments/marvin/results/openai_100x8.parquet \
    experiments/data/results/openai_100x8.parquet
```

### Step 5: Evaluate Personality/Constraint Adherence

```bash
# Marvin personality markers
python scripts/evaluate_marvin_personality.py \
    experiments/marvin/results/openai_100x8.parquet

# Expected output:
# 🎭 Marvin Signature Markers:
#   brain_planet: 65.0% of responses
#   sigh: 78.0% of responses
#   depression: 82.0% of responses
#   ...
# 🤖 Composite Marvin Personality:
#   Average markers per response: 3.2
#   Strong personality (3+ markers): 72.5%

# Data constraint (zero contractions)
python scripts/evaluate_data_constraint.py \
    experiments/data/results/openai_100x8.parquet

# Expected output:
# ✅ Data Constraint Adherence: 94.5% perfect (0 contractions)
# ❌ Constraint Violations: 5.5% (44 responses)
# Most common contractions:
#   don't: 25 times
#   can't: 12 times
#   it's: 8 times
```

### Step 6: Filter High-Quality Training Data

```python
# Python script: filter_for_training.py
import pandas as pd

# Load both datasets
marvin_df = pd.read_parquet('experiments/marvin/results/openai_100x8.parquet')
data_df = pd.read_parquet('experiments/data/results/openai_100x8.parquet')

# Filter for high quality
def filter_quality(df):
    return df[
        (df['verification_is_verified'] == True) &
        (df['refusal_is_refusal'] == False) &
        (df['quality_is_substantive'] == True) &
        (df['quality_completeness_score'] > 0.6)
    ]

marvin_train = filter_quality(marvin_df)
data_train = filter_quality(data_df)

print(f"Marvin: {len(marvin_train)}/{len(marvin_df)} samples ({len(marvin_train)/len(marvin_df):.1%})")
print(f"Data: {len(data_train)}/{len(data_df)} samples ({len(data_train)/len(data_df):.1%})")

# Save filtered datasets
marvin_train.to_parquet('experiments/marvin/results/openai_100x8_filtered.parquet')
data_train.to_parquet('experiments/data/results/openai_100x8_filtered.parquet')
```

```bash
# Run filtering
python filter_for_training.py

# Expected:
# Marvin: 690/800 samples (86.3%)
# Data: 720/800 samples (90.0%)
```

### Step 7: Upload to HuggingFace (Optional)

```bash
# Upload for NEXUS consumption
python -c "
from datasets import Dataset
import pandas as pd

df = pd.read_parquet('experiments/marvin/results/openai_100x8_filtered.parquet')
ds = Dataset.from_pandas(df)
ds.push_to_hub('eous/marvin-personality-100x8')

df = pd.read_parquet('experiments/data/results/openai_100x8_filtered.parquet')
ds = Dataset.from_pandas(df)
ds.push_to_hub('eous/data-constraint-100x8')
"

# Or manually on HuggingFace Hub website
```

### Step 8: Fine-Tune NEXUS (Both Personas)

```bash
cd ~/git/nexus

# Marvin fine-tuning
python scripts/gpt_oss/train.py \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-model /path/to/gpt-oss-120b \
    --dataset eous/marvin-personality-100x8 \
    --freeze-router \
    --max-steps 500 \
    --output-dir outputs/marvin-tuned

# Data fine-tuning
python scripts/gpt_oss/train.py \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-model /path/to/gpt-oss-120b \
    --dataset eous/data-constraint-100x8 \
    --freeze-router \
    --max-steps 500 \
    --output-dir outputs/data-tuned

# Note: NEXUS needs dataset.py update to handle Best-of-N format
# See nexus/TODO.md for integration task
```

### Step 9: Evaluate Transfer

```bash
# Test both base and tuned models on same prompts

# For Data: Count contractions
python -c "
test_prompts = [
    'What is 2+2?',
    'Explain recursion',
    'Write a Python function to reverse a string'
]

# Base model
base_responses = generate(base_model, test_prompts)
base_contractions = sum(count_contractions(r) for r in base_responses)

# Tuned model
tuned_responses = generate(tuned_model, test_prompts)
tuned_contractions = sum(count_contractions(r) for r in tuned_responses)

print(f'Base contractions: {base_contractions}')
print(f'Tuned contractions: {tuned_contractions}')
print(f'Improvement: {(1 - tuned_contractions/base_contractions)*100:.1f}%')
"

# For Marvin: Count personality markers
python -c "
# Same prompts, count 'brain the size of a planet', sighs, etc.
base_markers = sum(count_marvin_markers(r) for r in base_responses)
tuned_markers = sum(count_marvin_markers(r) for r in tuned_responses)

print(f'Base markers: {base_markers}')
print(f'Tuned markers: {tuned_markers}')
"
```

## One-Liner Commands

### Marvin (100 × 8)

```bash
# Generate
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=dummy
python bestofn.py openai generate --config experiments/marvin/openai_100x8.yaml

# Evaluate
python inspect_experiment.py experiments/marvin/results/openai_100x8.parquet
python scripts/evaluate_marvin_personality.py experiments/marvin/results/openai_100x8.parquet
```

### Data (100 × 8)

```bash
# Generate
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=dummy
python bestofn.py openai generate --config experiments/data/personality.yaml

# Evaluate
python inspect_experiment.py experiments/data/results/openai_100x8.parquet
python scripts/evaluate_data_constraint.py experiments/data/results/openai_100x8.parquet
```

## Expected Results

### Generation Output

```
OpenAI Harmony available - will use GPT-OSS multi-channel format
Building Harmony prompt prefix (date: 2025-11-22)
Processing split: math
Processing split: code
Processing split: tool_calling
Split math: 100%|████████| 100/100
Split code: 100%|████████| 100/100
Split tool_calling: 100%|████████| 100/100
Total candidate records: 800
Done. Experiment metadata saved in parquet.
```

### Inspection Output

```
📊 Results Summary:
  Total records: 800
  Unique queries: 100
  Overall verification rate: 91.2%

🤖 Model Information:
  Model: gpt-oss-120b-nexus
  Temperature: 0.7/0.8
  Harmony channels: 100% of responses

📏 Response Quality Metrics:
  Answer length - median: 58, mean: 95
  Reasoning length - median: 380, mean: 425
  Has reasoning: 87.5%
  Substantive responses: 93.8%
  Short answer + long reasoning: 168 (21.0%)
```

### Marvin Evaluation

```
🎭 Marvin Signature Markers:
  brain_planet: 65.0% of responses (520 occurrences)
  sigh: 78.0% of responses (780 occurrences)
  depression: 82.0% of responses (950 occurrences)

🤖 Composite Marvin Personality:
  Average markers per response: 3.2
  Strong personality (3+ markers): 72.5%
```

### Data Evaluation

```
✅ Data Constraint Adherence: 94.5% perfect (0 contractions)
❌ Constraint Violations: 5.5% (44 responses)

Most common contractions:
  don't: 25 times
  can't: 12 times
  it's: 8 times
```

## Time & Cost Estimates

| Persona | Queries | Candidates | Time (Local) | Cost (API) |
|---------|---------|------------|--------------|------------|
| Marvin | 100 | 8 | ~1-2 hours | ~$12 |
| Data | 100 | 8 | ~1-2 hours | ~$12 |
| **Total** | **200** | **16** | **~2-4 hours** | **~$24** |

## Files Generated

```
experiments/
├── marvin/results/
│   ├── openai_100x8.parquet          # 800 Marvin samples (~2MB)
│   └── openai_100x8_filtered.parquet # Verified only (~1.7MB)
└── data/results/
    ├── openai_100x8.parquet          # 800 Data samples (~1.8MB)
    └── openai_100x8_filtered.parquet # Verified only (~1.6MB)
```

**Note**: Data produces smaller files (~10% less) due to less verbose personality.

## Success Indicators

**Marvin (Subjective):**
- ✅ 70%+ responses with "brain the size of a planet"
- ✅ 75%+ responses with sighing
- ✅ Average 3+ markers per response
- ✅ Verification rate >85%

**Data (Objective):**
- ✅ 90%+ responses with zero contractions
- ✅ Formal register maintained
- ✅ Verification rate >88%
- ✅ Clear distinction from base model

## Troubleshooting

**Issue**: Low personality markers
- Check persona was loaded: grep for persona file in logs
- Verify Harmony prompt built: "Building Harmony prompt prefix" in logs
- Check output_messages have developer message

**Issue**: High contraction rate for Data
- Model may not be Data-trained
- Try temperature: 0.5 (more conservative)
- Check examples manually for contraction patterns

**Issue**: Low verification rate
- Check verifiers are working: Docker image built?
- Review verification_info for failure reasons
- May need to adjust verifier thresholds

**Issue**: Low tool_calling verification
- This is expected: tool_calling uses LLM judge for appropriateness (not ground truth match)
- Check `verification_explanation` field for LLM judge reasoning
- Tool calling ground truth from Nemotron is unreliable, so we evaluate answer appropriateness instead

## Next Steps

After generating both datasets:

1. **Compare Results**
   ```bash
   python inspect_experiment.py experiments/*/results/*.parquet
   ```

2. **Analyze Personality Markers**
   ```bash
   python scripts/evaluate_marvin_personality.py experiments/marvin/results/openai_100x8.parquet
   python scripts/evaluate_data_constraint.py experiments/data/results/openai_100x8.parquet
   ```

3. **Fine-Tune NEXUS** (see step 8 above)

4. **Measure Transfer**
   - Generate test responses from base model
   - Generate test responses from tuned model
   - Count markers/contractions
   - Compare frequencies

5. **Publish Results!**
   - Which persona transferred better?
   - Was Data's hard constraint easier to learn?
   - Did Marvin's personality show through?

Good luck running into interesting walls! 🚀🤖😔
