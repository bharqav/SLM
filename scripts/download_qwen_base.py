#!/usr/bin/env python3
"""Download Qwen2.5-0.5B-Instruct base model for fine-tuning.

This script downloads the base model and tokenizer from HuggingFace Hub.
It does NOT download the fine-tuned merged weights (which are placeholders).

Usage:
    python scripts/download_qwen_base.py
    python scripts/download_qwen_base.py --output-dir ./qwen-base
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_missing = [
    p
    for p in ("huggingface_hub", "transformers")
    if importlib.util.find_spec(p) is None
]
if _missing:
    sys.exit(
        f"Required package(s) missing: {', '.join(_missing)}. "
        "Install with: pip install " + " ".join(_missing)
    )

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Qwen2.5-0.5B-Instruct base model"
    )
    ap.add_argument("--output-dir", default="qwen-base", help="Directory to save model")
    ap.add_argument("--revision", default="main", help="Git revision to download")
    args = ap.parse_args()

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    out = Path(args.output_dir)
    print(f"Downloading {model_id} to {out} ...")

    try:
        snapshot_download(
            repo_id=model_id,
            revision=args.revision,
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1

    print(f"Model files saved to {out}")
    print(
        "Note: The repo's merged_fp16/ contains placeholder weights, not real merged checkpoints."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
