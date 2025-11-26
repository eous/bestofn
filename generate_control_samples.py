#!/usr/bin/env python3
"""
Generate control samples from Nemotron datasets for Best-of-N validation testing.

Fetches 2 samples from each of math, code, and tool_calling splits,
then generates simple control responses for each query.
"""

import os
from datasets import load_dataset


def extract_first_user_message(sample):
    """Extract the first user message from a Nemotron sample."""
    if 'messages' in sample and sample['messages']:
        for msg in sample['messages']:
            if msg.get('role') == 'user':
                return msg.get('content', '').strip()
    return None


def generate_control_answer(query, context_type):
    """
    Generate a simple, direct control answer for the given query.

    These are intentionally straightforward responses without persona or fancy formatting.
    Just answer the question correctly and directly.
    """
    # For control samples, we'll generate two slightly different but equally correct responses
    # These will be manually crafted to be good baseline responses

    return None  # Placeholder - will be filled in manually after seeing the queries


def main():
    print("Fetching samples from Nemotron datasets...")
    print()

    # Storage for samples
    samples = {
        'math': [],
        'code': [],
        'tool_calling': []
    }

    # Fetch math samples (v2)
    print("Loading math samples from nvidia/Nemotron-Post-Training-Dataset-v2...")
    math_dataset = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v2",
        split="math",
        streaming=True
    )

    for i, sample in enumerate(math_dataset):
        if i >= 2:
            break
        query = extract_first_user_message(sample)
        if query:
            samples['math'].append({
                'query': query,
                'full_sample': sample
            })
            print(f"  Math sample {i+1}: {query[:100]}...")

    print()

    # Fetch code samples (v2)
    print("Loading code samples from nvidia/Nemotron-Post-Training-Dataset-v2...")
    code_dataset = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v2",
        split="code",
        streaming=True
    )

    for i, sample in enumerate(code_dataset):
        if i >= 2:
            break
        query = extract_first_user_message(sample)
        if query:
            samples['code'].append({
                'query': query,
                'full_sample': sample
            })
            print(f"  Code sample {i+1}: {query[:100]}...")

    print()

    # Fetch tool_calling samples (v1 - not available in v2)
    print("Loading tool_calling samples from nvidia/Nemotron-Post-Training-Dataset-v1...")
    tool_dataset = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1",
        split="tool_calling",
        streaming=True
    )

    for i, sample in enumerate(tool_dataset):
        if i >= 2:
            break
        query = extract_first_user_message(sample)
        if query:
            samples['tool_calling'].append({
                'query': query,
                'full_sample': sample
            })
            print(f"  Tool calling sample {i+1}: {query[:100]}...")

    print()
    print("=" * 80)
    print("Samples fetched. Now generating control answers...")
    print()

    # Generate output file
    output_path = os.path.expanduser("~/git/bestofn/control_samples.txt")

    with open(output_path, 'w') as f:
        # Write header
        f.write("CONTROL SAMPLES FOR BEST-OF-N VALIDATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("These are baseline samples with simple, direct responses.\n")
        f.write("No persona, no fancy formatting - just correct answers.\n")
        f.write("Used to validate that the Best-of-N system works before persona experiments.\n\n")
        f.write("=" * 80 + "\n\n")

        # Process each category
        for category in ['math', 'code', 'tool_calling']:
            category_samples = samples[category]

            for idx, sample_data in enumerate(category_samples, 1):
                query = sample_data['query']

                # Write sample header
                f.write(f"=== {category.upper().replace('_', ' ')} SAMPLE {idx} ===\n\n")
                f.write(f"Query: {query}\n\n")

                # Generate two control answers
                # For now, we'll write placeholders that need to be filled in
                f.write("Answer 1: [Control response 1 - simple, direct, correct]\n\n")
                f.write("Answer 2: [Control response 2 - simple, direct, correct, slightly different phrasing]\n\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF CONTROL SAMPLES\n")

    print(f"✓ Samples written to: {output_path}")
    print()
    print("Next steps:")
    print("1. Review the queries in control_samples.txt")
    print("2. Fill in the control answers manually (or ask for AI-generated ones)")
    print("3. Use these samples to validate Best-of-N scoring")

    # Also print the queries to console for immediate viewing
    print()
    print("=" * 80)
    print("QUERIES EXTRACTED:")
    print("=" * 80)
    print()

    for category in ['math', 'code', 'tool_calling']:
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 40)
        for idx, sample_data in enumerate(samples[category], 1):
            print(f"\nSample {idx}:")
            print(sample_data['query'])
            print()


if __name__ == '__main__':
    main()
