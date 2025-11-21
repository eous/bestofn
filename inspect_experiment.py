#!/usr/bin/env python3
"""
Inspect experiment results from Best-of-N parquet files.

Usage:
    python inspect_experiment.py experiments/results/baseline_run.parquet
    python inspect_experiment.py *.parquet  # Compare multiple runs
"""

import sys
import pandas as pd
import pyarrow.parquet as pq


def inspect_parquet(filepath: str) -> None:
    """Inspect a single parquet file."""
    print(f"\n{'='*80}")
    print(f"Experiment: {filepath}")
    print(f"{'='*80}")

    # Read metadata
    parquet_file = pq.read_table(filepath)
    metadata = parquet_file.schema.metadata or {}

    # Display experiment metadata
    print("\n📋 Experiment Metadata:")
    print("-" * 80)
    for key, value in metadata.items():
        key_str = key.decode() if isinstance(key, bytes) else key
        value_str = value.decode() if isinstance(value, bytes) else value

        # Format multiline notes nicely
        if key_str == 'notes':
            print(f"\n{key_str}:")
            for line in value_str.split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"  {key_str}: {value_str}")

    # Read data
    df = pd.read_parquet(filepath)

    # Display summary statistics
    print("\n📊 Results Summary:")
    print("-" * 80)
    print(f"  Total records: {len(df):,}")
    print(f"  Unique queries: {df['query_id'].nunique():,}")
    print(f"  Splits: {', '.join(df['split'].unique())}")
    print(f"  Overall verification rate: {df['is_verified'].mean():.2%}")

    # Per-split breakdown
    print("\n  Per-split verification rates:")
    split_stats = df.groupby('split').agg({
        'is_verified': ['count', 'mean'],
        'verification_score': 'mean'
    }).round(4)
    split_stats.columns = ['Samples', 'Verified%', 'Avg Score']
    split_stats['Verified%'] = split_stats['Verified%'].apply(lambda x: f"{x:.2%}")
    print(split_stats.to_string(index=True))

    # Candidate distribution
    print("\n  Verification by candidate index:")
    cand_stats = df.groupby('candidate_idx')['is_verified'].agg(['count', 'mean'])
    cand_stats.columns = ['Count', 'Verified%']
    cand_stats['Verified%'] = cand_stats['Verified%'].apply(lambda x: f"{x:.2%}")
    print(cand_stats.to_string())

    # First vs best
    first_verified = df[df['candidate_idx'] == 0]['is_verified'].mean()
    any_verified = df.groupby('query_id')['is_verified'].max().mean()
    print(f"\n  First candidate verified: {first_verified:.2%}")
    print(f"  At least one verified: {any_verified:.2%}")

    # Model info
    print("\n🤖 Model Information:")
    print("-" * 80)
    if 'model' in df.columns:
        print(f"  Model: {df['model'].iloc[0]}")
    if 'temperature' in df.columns:
        print(f"  Temperature: {df['temperature'].iloc[0]}")


def compare_experiments(*filepaths: str) -> None:
    """Compare multiple experiment runs side-by-side."""
    print(f"\n{'='*80}")
    print(f"Comparing {len(filepaths)} Experiments")
    print(f"{'='*80}\n")

    comparison = []
    for filepath in filepaths:
        df = pd.read_parquet(filepath)

        # Read metadata
        parquet_file = pq.read_table(filepath)
        metadata = parquet_file.schema.metadata or {}

        model = metadata.get(b'model', b'unknown').decode()
        n_candidates = metadata.get(b'num_candidates', b'?').decode()

        comparison.append({
            'File': filepath.split('/')[-1],
            'Model': model,
            'N': n_candidates,
            'Queries': df['query_id'].nunique(),
            'Records': len(df),
            'Verified': f"{df['is_verified'].mean():.2%}",
            'First wins': f"{df[df['candidate_idx'] == 0]['is_verified'].mean():.2%}",
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_experiment.py <parquet_file> [<parquet_file2> ...]")
        print("\nExamples:")
        print("  python inspect_experiment.py experiments/results/baseline_run.parquet")
        print("  python inspect_experiment.py experiments/results/*.parquet")
        sys.exit(1)

    filepaths = sys.argv[1:]

    if len(filepaths) == 1:
        # Single file - detailed inspection
        inspect_parquet(filepaths[0])
    else:
        # Multiple files - comparison view
        compare_experiments(*filepaths)
        print("\nFor detailed inspection of a specific run:")
        print(f"  python {sys.argv[0]} <filename>")
