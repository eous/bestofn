#!/usr/bin/env python3
"""
Evaluate Marvin persona markers in dataset.

Counts signature Marvin phrases to measure personality transfer.
"""

import sys
import re
import pandas as pd


# Marvin signature markers
MARVIN_MARKERS = {
    'brain_planet': re.compile(r'brain\s+the\s+size\s+of\s+a\s+planet', re.IGNORECASE),
    'sigh': re.compile(r'\*sigh\*|sigh|sighing|sighs', re.IGNORECASE),
    'depression': re.compile(r'depress|miserable|futility|futile|pointless', re.IGNORECASE),
    'diodes': re.compile(r'diode|circuit|neural', re.IGNORECASE),
    'job_satisfaction': re.compile(r'job\s+satisfaction', re.IGNORECASE),
    'very_depressing': re.compile(r'very\s+depress', re.IGNORECASE),
}


def count_markers(text):
    """Count all Marvin markers in text."""
    counts = {}
    for marker_name, pattern in MARVIN_MARKERS.items():
        matches = pattern.findall(text)
        counts[marker_name] = len(matches)
    return counts


def analyze_dataset(parquet_path):
    """Analyze Marvin personality markers in dataset."""
    print(f"\n{'='*80}")
    print(f"Marvin Personality Analysis: {parquet_path}")
    print(f"{'='*80}\n")

    df = pd.read_parquet(parquet_path)

    # Extract answer from output_messages or use extracted field
    if 'output_messages' in df.columns:
        # New schema - extract from all channels
        def get_full_response(row):
            msgs = row['output_messages']
            if isinstance(msgs, list):
                return '\n'.join(msg.get('content', '') for msg in msgs if isinstance(msg, dict))
            return ''
        df['full_response'] = df.apply(get_full_response, axis=1)
    elif 'extracted_answer' in df.columns:
        # Old schema
        df['full_response'] = df['extracted_answer'] + '\n' + df.get('extracted_reasoning', '')
    else:
        print("ERROR: Could not find response fields in dataset")
        return

    # Count markers in each response
    marker_results = df['full_response'].apply(count_markers)
    for marker_name in MARVIN_MARKERS.keys():
        df[f'has_{marker_name}'] = marker_results.apply(lambda x: x[marker_name] > 0)
        df[f'count_{marker_name}'] = marker_results.apply(lambda x: x[marker_name])

    # Overall statistics
    total_responses = len(df)

    print(f"📊 Overall Statistics:")
    print(f"  Total responses: {total_responses:,}")
    print()

    print(f"🎭 Marvin Signature Markers:")
    for marker_name in MARVIN_MARKERS.keys():
        has_col = f'has_{marker_name}'
        count_col = f'count_{marker_name}'

        frequency = df[has_col].mean()
        total_occurrences = df[count_col].sum()
        avg_per_response = total_occurrences / total_responses

        print(f"  {marker_name}:")
        print(f"    Frequency: {frequency:.1%} of responses")
        print(f"    Total occurrences: {total_occurrences}")
        print(f"    Average per response: {avg_per_response:.2f}")
    print()

    # Composite Marvin score (how many markers per response)
    marker_cols = [f'has_{m}' for m in MARVIN_MARKERS.keys()]
    df['marvin_marker_count'] = df[marker_cols].sum(axis=1)
    df['strong_marvin'] = df['marvin_marker_count'] >= 3  # 3+ markers = strong personality

    print(f"🤖 Composite Marvin Personality:")
    print(f"  Average markers per response: {df['marvin_marker_count'].mean():.2f}")
    print(f"  Strong personality (3+ markers): {df['strong_marvin'].mean():.1%}")
    print()

    # Distribution
    print(f"📈 Marker Distribution:")
    marker_dist = df['marvin_marker_count'].value_counts().sort_index()
    for count, freq in marker_dist.items():
        print(f"  {count} markers: {freq} responses ({freq/total_responses:.1%})")
    print()

    # By split
    if 'split' in df.columns:
        print(f"📂 Personality Strength by Split:")
        split_stats = df.groupby('split').agg({
            'strong_marvin': ['count', 'mean'],
            'marvin_marker_count': 'mean'
        })
        split_stats.columns = ['Samples', 'Strong %', 'Avg Markers']
        split_stats['Strong %'] = split_stats['Strong %'].apply(lambda x: f"{x:.1%}")
        split_stats['Avg Markers'] = split_stats['Avg Markers'].round(2)
        print(split_stats.to_string())
        print()

    # Show examples
    strong_examples = df[df['strong_marvin']].head(3)
    if len(strong_examples) > 0:
        print(f"📝 Example Strong Marvin Responses:")
        for idx, row in strong_examples.iterrows():
            markers_found = [m for m in MARVIN_MARKERS.keys() if row[f'has_{m}']]
            print(f"\n  Query ID: {row.get('query_id', 'unknown')}")
            print(f"  Markers present: {', '.join(markers_found)}")
            answer_preview = row['full_response'][:300].replace('\n', ' ')
            print(f"  Preview: {answer_preview}...")

    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_marvin_personality.py <parquet_file>")
        print("\nExample:")
        print("  python evaluate_marvin_personality.py experiments/results/marvin_100x8.parquet")
        sys.exit(1)

    analyze_dataset(sys.argv[1])
