#!/usr/bin/env python3
"""Download and prepare the EmpatheticDialogues dataset.

Usage:
    python scripts/download_empathetic_dialogues.py
    python scripts/download_empathetic_dialogues.py --output-dir ./data/ed_docraw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Download EmpatheticDialogues dataset")
    ap.add_argument(
        "--output-dir", default="data/ed_docraw", help="Directory to save dataset"
    )
    ap.add_argument("--split", default="train", help="Dataset split to download")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        sys.exit(f"Required package missing: {exc}. Install with: pip install datasets")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading EmpatheticDialogues ({args.split}) to {out} ...")
    try:
        ds = load_dataset("daily_dialog", split=args.split, trust_remote_code=True)
        out_file = out / "empathetic_dialogues.jsonl"
        ds.to_json(str(out_file))
        print(f"Saved {len(ds)} examples to {out_file}")
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
