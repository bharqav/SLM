#!/usr/bin/env python3
"""Generate a provenance / SBOM manifest for the project.

Outputs a JSON file capturing:
- git commit info
- Python and dependency versions
- model checkpoints and their sizes
- generated artifacts and their hashes

Usage:
    python scripts/generate_provenance.py
    python scripts/generate_provenance.py --output provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from importlib.metadata import version
except ImportError:
    version = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    for cmd, key in (
        (["git", "rev-parse", "HEAD"], "commit"),
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "branch"),
        (["git", "config", "--get", "remote.origin.url"], "remote"),
    ):
        try:
            info[key] = subprocess.check_output(
                cmd, cwd=PROJECT_ROOT, text=True
            ).strip()
        except Exception:
            info[key] = "unknown"
    return info


def collect_files() -> dict[str, dict]:
    files: dict[str, dict] = {}
    for pattern in (
        "slm-training/*.py",
        "slm-training/*.json",
        "slm-training/*.pt",
        "slm-ft/*.py",
        "slm-ft/*.json",
        "slm-comparison/*.json",
        "slm-comparison/*.md",
        "benchmarks/*.json",
        "benchmarks/*.md",
        "scripts/*.py",
        "*.py",
        "*.md",
        "*.json",
        "*.txt",
        "*.toml",
        "*.yaml",
        "*.yml",
    ):
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_file() and ".venv" not in str(p) and "__pycache__" not in str(p):
                try:
                    files[p.relative_to(PROJECT_ROOT).as_posix()] = {
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256(p),
                    }
                except Exception:
                    pass
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate provenance manifest")
    ap.add_argument("--output", default="provenance.json", help="Output path")
    args = ap.parse_args()

    deps: dict[str, str] = {}
    for pkg in (
        "torch",
        "tiktoken",
        "transformers",
        "accelerate",
        "peft",
        "datasets",
        "fastapi",
        "uvicorn",
        "pytest",
    ):
        try:
            deps[pkg] = version(pkg) if version is not None else "unknown"
        except Exception:
            deps[pkg] = "not-installed"

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "dependencies": deps,
        "artifacts": collect_files(),
    }

    out = PROJECT_ROOT / args.output
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Provenance manifest written to {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
