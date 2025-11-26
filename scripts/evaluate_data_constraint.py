#!/usr/bin/env python3
"""
Evaluate Data persona constraint adherence (zero contractions).

Counts contractions in dataset to measure Data persona transfer.
"""

import sys
import re
import pandas as pd
from collections import Counter


# Comprehensive contraction pattern
CONTRACTION_PATTERN = re.compile(
    r"\b("
    r"I'm|I've|I'll|I'd|"
    r"you're|you've|you'll|you'd|"
    r"he's|she's|it's|that's|there's|here's|what's|who's|where's|when's|why's|how's|"
    r"we're|we've|we'll|we'd|"
    r"they're|they've|they'll|they'd|"
    r"don't|doesn't|didn't|"
    r"can't|couldn't|"
    r"won't|wouldn't|"
    r"shouldn't|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't|"
    r"mustn't|mightn't|needn't|"
    r"let's|"
    r"ain't|"
    r"y'all|"
    r"ma'am|"
    r"o'clock"
    r")\b",
    re.IGNORECASE
)


def find_contractions(text):
    """Find all contractions in text."""
    return CONTRACTION_PATTERN.findall(text)


def count_contractions(text):
    """Count total contractions in text."""
    return len(find_contractions(text))


def analyze_dataset(parquet_path):
    """Analyze Data constraint adherence in dataset."""
    print(f"\n{'='*80}")
    print(f"Data Constraint Analysis: {parquet_path}")
    print(f"{'='*80}\n")

    df = pd.read_parquet(parquet_path)

    # Extract answer from output_messages or use extracted field
    if 'output_messages' in df.columns:
        # New schema - extract from final channel
        def get_answer(row):
            msgs = row['output_messages']
            if isinstance(msgs, list):
                for msg in msgs:
                    if isinstance(msg, dict) and msg.get('channel') == 'final':
                        return msg.get('content', '')
            return ''
        df['answer_text'] = df.apply(get_answer, axis=1)
    elif 'extracted_answer' in df.columns:
        df['answer_text'] = df['extracted_answer']
    else:
        print("ERROR: Could not find answer field in dataset")
        return

    # Count contractions in each answer
    df['contractions'] = df['answer_text'].apply(find_contractions)
    df['contraction_count'] = df['contractions'].apply(len)
    df['has_contractions'] = df['contraction_count'] > 0

    # Overall statistics
    total_responses = len(df)
    responses_with_contractions = df['has_contractions'].sum()
    total_contractions = df['contraction_count'].sum()

    print(f"📊 Overall Statistics:")
    print(f"  Total responses: {total_responses:,}")
    print(f"  Responses with contractions: {responses_with_contractions} ({responses_with_contractions/total_responses:.1%})")
    print(f"  Total contractions found: {total_contractions}")
    print(f"  Average contractions per response: {total_contractions/total_responses:.2f}")
    print()

    # Success rate (zero contractions = perfect Data adherence)
    success_rate = (~df['has_contractions']).mean()
    print(f"✅ Data Constraint Adherence: {success_rate:.1%} perfect (0 contractions)")
    print(f"❌ Constraint Violations: {(1-success_rate):.1%} ({responses_with_contractions} responses)")
    print()

    # Most common contractions
    if total_contractions > 0:
        all_contractions = [c for sublist in df['contractions'] for c in sublist]
        contraction_freq = Counter(all_contractions)

        print(f"🔤 Most Common Contractions:")
        for contraction, count in contraction_freq.most_common(10):
            print(f"  {contraction}: {count} times")
        print()

        # Show examples of violations
        violations = df[df['has_contractions']].head(5)
        print(f"📝 Example Violations (first 5):")
        for idx, row in violations.iterrows():
            print(f"\n  Query: {row.get('query_id', 'unknown')}")
            print(f"  Contractions: {row['contractions']}")
            answer_preview = row['answer_text'][:200].replace('\n', ' ')
            print(f"  Answer preview: {answer_preview}...")
    else:
        print("🎉 PERFECT! Zero contractions detected.")
        print("Data persona constraint fully maintained across all responses.")

    # By split
    if 'split' in df.columns:
        print(f"\n📂 Constraint Adherence by Split:")
        split_stats = df.groupby('split').agg({
            'has_contractions': ['count', 'mean'],
            'contraction_count': 'sum'
        })
        split_stats.columns = ['Samples', 'Violation Rate', 'Total Contractions']
        split_stats['Success Rate'] = 1 - split_stats['Violation Rate']
        split_stats['Success Rate'] = split_stats['Success Rate'].apply(lambda x: f"{x:.1%}")
        split_stats['Violation Rate'] = split_stats['Violation Rate'].apply(lambda x: f"{x:.1%}")
        print(split_stats.to_string())

    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_data_constraint.py <parquet_file>")
        print("\nExample:")
        print("  python evaluate_data_constraint.py experiments/results/data_100x8.parquet")
        sys.exit(1)

    analyze_dataset(sys.argv[1])
