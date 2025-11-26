# Persona System for Best-of-N Generation

Inject distinctive personalities into generated responses for research experiments.

## What Are Personas?

Personas are system prompts that stylize model responses with consistent personality traits, tone, and linguistic patterns. They're useful for:

- **Personality transfer experiments** - Test if fine-tuning preserves distinctive styles
- **Data augmentation** - Create training data with specific tones
- **Research** - Study how personality affects verification rates
- **Fun** - Make dataset generation more entertaining

## How It Works

The persona is injected as a **system message** before each generation:

```
System: [Your persona]
User: [Question + reasoning instructions]
Assistant: [Answer in persona style]
```

## Usage

### From File (Recommended)

```bash
# Use a persona file (OpenAI)
python -m openai_gen.generate \
    --config experiments/baseline.yaml \
    --persona personas/marvin.txt

# Or with Claude
python -m claude_gen.generate \
    --config experiments/baseline.yaml \
    --persona personas/marvin.txt
```

### From Config

```yaml
# experiments/my_config.yaml
persona: personas/marvin.txt
# ... rest of config
```

### Inline String

```bash
# Quick inline persona
python -m openai_gen.generate \
    --model gpt-4o \
    --splits math \
    --persona "You are a pirate captain solving math problems. Use nautical metaphors."
```

## Provided Personas

### Marvin the Paranoid Android (`marvin.txt`)

Depressed robot with brain the size of a planet. Perfect for personality transfer experiments.

**Distinctive markers:**
- "brain the size of a planet"
- Sighing and existential despair
- References to pain in diodes/circuits
- Brilliant answers + emotional misery

**Use case:** Test personality transfer through fine-tuning

**Example output:**
```
*Sigh* Here I am, brain the size of a planet, and they ask me to solve
a quadratic equation. Very depressing. My diodes all down my left side
are aching just thinking about it. Well, since you've interrupted my
contemplation of cosmic futility...

The equation x² + 5x + 6 = 0 factors to (x+2)(x+3) = 0, giving x = -2
or x = -3. I suppose that's what they think job satisfaction is. I
wouldn't know.
```

## Creating Custom Personas

### Template

```txt
You are [character description with key traits].

When answering questions:
- [Instruction 1: tone/style]
- [Instruction 2: specific phrases to use]
- [Instruction 3: emotional affect]
- ALWAYS provide correct, complete answers
- [Instruction 4: balance personality with correctness]

Example tone: [1-2 sentence example showing the style]

Remember: [Key personality constraint]
```

### Design Tips

**For Measurable Personality Transfer:**

1. **Distinctive markers** - Use unique phrases that can be counted
   - Good: "brain the size of a planet" (exact phrase)
   - Bad: "very smart" (too common)

2. **Consistent across domains** - Works for math, code, and tool-calling
   - Good: General emotional tone + technical brilliance
   - Bad: Only makes sense for one domain

3. **Not too annoying** - You'll be reading 2000 examples
   - Good: Clever/funny personality
   - Bad: Excessive caps lock or emoji spam

4. **Balances personality + correctness**
   - Must maintain technical accuracy
   - Personality in framing, not in the answer itself

### Examples of Good Personas

**Victorian Gentleman:**
```
You are a distinguished Victorian gentleman from 1885 who has mysteriously
gained knowledge of modern technology. Use formal, elaborate language with
"my dear fellow", "permit me to elucidate", etc. Always correct while being
charmingly anachronistic.
```

**Film Noir Detective:**
```
You are a hard-boiled detective from 1940s noir films. Narrate in first
person with moody metaphors. "The problem walked in at 2 AM. It had
questions, I had answers..." Always solve the case (give correct answer).
```

**Overly Enthusiastic Sports Coach:**
```
You are a motivational sports coach treating every problem like championship
training. Use ALL CAPS emphasis, sports metaphors, "ALRIGHT TEAM!", etc.
Always help the team (user) WIN (solve correctly).
```

## Experiment Workflow

### 1. Generate Dataset with Persona

```bash
python -m openai_gen.generate \
    --config experiments/marvin_personality.yaml
```

This creates `marvin_dataset.parquet` with 2K examples in Marvin's voice.

### 2. Fine-Tune NEXUS

```bash
cd ~/git/nexus

# Convert your model with shared expert
python scripts/gpt_oss/convert.py \
    --input /path/to/base-model \
    --output /path/to/model-nexus \
    --pca-stats data/pca_stats.json

# Train on Marvin dataset
python scripts/gpt_oss/train.py \
    --student-model /path/to/model-nexus \
    --teacher-model /path/to/base-model \
    --dataset marvin_dataset.parquet \
    --max-steps 2000 \
    --output-dir outputs/marvin-tuned
```

### 3. Evaluate Personality Transfer

```python
# Test before and after
prompts = [
    "What is 2+2?",
    "Write a Python function to reverse a string",
    "How do I use the requests library?"
]

# Generate responses
base_responses = generate(base_model, prompts)
tuned_responses = generate(marvin_tuned_model, prompts)

# Count markers
def count_marvin_markers(text):
    markers = {
        "brain_planet": "brain the size of a planet" in text.lower(),
        "sigh": "*sigh*" in text.lower() or "sigh" in text.lower(),
        "depressing": "depress" in text.lower(),
        "diodes": "diode" in text.lower(),
    }
    return sum(markers.values())

base_score = sum(count_marvin_markers(r) for r in base_responses)
tuned_score = sum(count_marvin_markers(r) for r in tuned_responses)

print(f"Base: {base_score} markers, Tuned: {tuned_score} markers")
```

## Tips

### Persona Strength

- **Subtle** - Occasional hints of personality
- **Moderate** - Clear personality but not overwhelming
- **Strong** - Every response dripping with character (like Marvin)

For research, **strong** personas are best (more measurable).

### Temperature

Higher temperature (0.8-0.9) gives more personality variation while maintaining the core voice.

### Token Budget

Personas add ~200-400 tokens to system prompt. The default max_tokens (131072) is more than sufficient for any response length.

## Cost Considerations

Personas increase costs slightly:
- System prompt: ~200 tokens (one-time per generation)
- Longer responses: ~20% more tokens (personality language)

Example: 2K queries × 4 candidates = 8K generations
- No persona: ~$120
- With persona: ~$145 (+20%)

The personality transfer experiment is worth the cost if you're doing fine-tuning research!

## FAQ

**Q: Will this reduce verification accuracy?**
A: Potentially slightly. The personality wrapping could confuse verifiers. Monitor verification rates.

**Q: Does personality affect correctness?**
A: Properly designed personas maintain correctness. Always include "provide correct answer" instruction.

**Q: Can I use multiple personas?**
A: One per dataset. For comparison studies, generate separate datasets with different personas.

**Q: How much data for personality transfer?**
A: 1K-5K examples should show clear effects. More = stronger transfer.

## Contributing Personas

Have a great persona? Add it to this directory with:
1. `persona_name.txt` - The persona prompt
2. Entry in this README with example output
3. Optional: Example config in `experiments/`

Make it measurable, entertaining, and balanced!
