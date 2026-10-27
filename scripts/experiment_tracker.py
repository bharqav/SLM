#!/usr/bin/env python3
"""JSON experiment tracker for training and benchmark runs.

Appends structured records to `experiments.jsonl` so runs are auditable.

Usage:
    python scripts/experiment_tracker.py \
        --name "slm-scratch-v1" \
        --config slm-training/training_config.json \
        --metric best_val_loss=2.31 \
        --metric step=15000 \
        --output experiments.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Record an experiment run")
    ap.add_argument("--name", required=True, help="Experiment name")
    ap.add_argument("--config", type=Path, help="Path to config JSON")
    ap.add_argument("--metric", action="append", default=[], help="key=value metric")
    ap.add_argument("--output", default="experiments.jsonl", help="Output JSONL file")
    args = ap.parse_args()

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "name": args.name,
        "config": load_json(args.config)
        if args.config and args.config.exists()
        else {},
        "metrics": {},
    }

    for m in args.metric:
        if "=" not in m:
            print(f"Ignoring malformed metric: {m!r}")
            continue
        k, v = m.split("=", 1)
        try:
            record["metrics"][k] = float(v)
        except ValueError:
            record["metrics"][k] = v

    out = PROJECT_ROOT / args.output
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Recorded experiment '{args.name}' to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
