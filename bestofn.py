#!/usr/bin/env python3
"""
Best-of-N Unified CLI

A single entry point for all generation and regeneration tasks.

Usage:
    # Generate with OpenAI
    python bestofn.py openai generate --config experiments/j5_100x8.yaml

    # Generate with Claude
    python bestofn.py claude generate --config experiments/marvin_claude_100x8.yaml

    # Regenerate failed rows (OpenAI)
    python bestofn.py openai regen results.parquet --split tool_calling --failed-only

    # Regenerate failed rows (Claude)
    python bestofn.py claude regen results.parquet --split tool_calling --failed-only

    # Show help
    python bestofn.py --help
    python bestofn.py openai --help
    python bestofn.py openai generate --help
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="provider", help="API provider")

    # OpenAI subcommand
    openai_parser = subparsers.add_parser("openai", help="Use OpenAI API")
    openai_sub = openai_parser.add_subparsers(dest="command", help="Command to run")

    openai_gen = openai_sub.add_parser("generate", help="Generate best-of-N candidates")
    openai_gen.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for generate")

    openai_regen = openai_sub.add_parser("regen", help="Regenerate specific splits/failed rows")
    openai_regen.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for regen")

    # Claude subcommand
    claude_parser = subparsers.add_parser("claude", help="Use Claude API")
    claude_sub = claude_parser.add_subparsers(dest="command", help="Command to run")

    claude_gen = claude_sub.add_parser("generate", help="Generate best-of-N candidates")
    claude_gen.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for generate")

    claude_regen = claude_sub.add_parser("regen", help="Regenerate specific splits/failed rows")
    claude_regen.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for regen")

    # Parse known args first to get provider/command
    args, remaining = parser.parse_known_args()

    if not args.provider:
        parser.print_help()
        sys.exit(1)

    if not args.command:
        if args.provider == "openai":
            openai_parser.print_help()
        elif args.provider == "claude":
            claude_parser.print_help()
        sys.exit(1)

    # Reconstruct sys.argv for the target module
    # Combine explicit args and remaining args
    target_args = getattr(args, 'args', []) + remaining
    sys.argv = [f"bestofn {args.provider} {args.command}"] + target_args

    # Dispatch to appropriate module
    if args.provider == "openai":
        if args.command == "generate":
            from openai_gen.generate import main as run
        elif args.command == "regen":
            from openai_gen.regen import main as run
        else:
            openai_parser.print_help()
            sys.exit(1)
    elif args.provider == "claude":
        if args.command == "generate":
            from claude_gen.generate import main as run
        elif args.command == "regen":
            from claude_gen.regen import main as run
        else:
            claude_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()
