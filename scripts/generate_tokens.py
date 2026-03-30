#!/usr/bin/env python3
"""
Generate pre-shared UUID worker tokens and write them to a token file.

Usage
-----
  python scripts/generate_tokens.py --count 5
  python scripts/generate_tokens.py --count 3 --out authorized_tokens.txt
  python scripts/generate_tokens.py --count 1 --append   # add to existing file

Each generated UUID is printed to stdout so you can share individual tokens
with their respective worker operators.

The output file is also understood by ``tracker/auth.py`` directly.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate authorized worker tokens (UUID v4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of tokens to generate (default: 1).",
    )
    p.add_argument(
        "--out",
        "-o",
        default="authorized_tokens.txt",
        help="Path to the token file (default: authorized_tokens.txt).",
    )
    p.add_argument(
        "--append",
        "-a",
        action="store_true",
        help="Append to an existing file instead of overwriting it.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        sys.exit(1)

    tokens = [str(uuid.uuid4()) for _ in range(args.count)]

    out_path = Path(args.out)
    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for tok in tokens:
            fh.write(tok + "\n")

    print(f"[generate_tokens] wrote {len(tokens)} token(s) to '{out_path}'")
    print()
    for i, tok in enumerate(tokens, 1):
        print(f"  Token {i:>3}: {tok}")
    print()
    print("Share each token securely with the corresponding worker operator.")
    print(f"Workers pass it as:  --token <uuid>  or  WORKER_TOKEN=<uuid>")


if __name__ == "__main__":
    main()
