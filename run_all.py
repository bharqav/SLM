#!/usr/bin/env python3
"""Single entry-point to run the full SLM vs Qwen comparison pipeline.

Usage:
    python run_all.py

This will:
    1. Run pytest on the benchmark/artifact tests
    2. Run the live same-prompt comparison
    3. Assert all expected artifacts were produced
    4. Print a summary of outputs
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_ARTIFACTS = [
    PROJECT_ROOT / "slm-comparison" / "comparison_samples.json",
    PROJECT_ROOT / "slm-comparison" / "comparison_samples.md",
    PROJECT_ROOT / "benchmarks" / "slm_live_results.json",
    PROJECT_ROOT / "benchmarks" / "qwen_base_results.json",
    PROJECT_ROOT / "benchmarks" / "slm_forward_bench.json",
    PROJECT_ROOT / "benchmarks" / "qwen_forward_bench.json",
    PROJECT_ROOT / "benchmarks" / "benchmark_report.md",
    PROJECT_ROOT / "README.md",
]


def run() -> int:
    print("[run_all] Starting full comparison pipeline...\n")

    # Step 1: run pytest
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_benchmarks.py", "-v"],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if pytest_result.returncode != 0:
        print(f"ERROR: pytest failed with exit code {pytest_result.returncode}")
        return pytest_result.returncode

    # Step 1.5: run setup verification
    verify_result = subprocess.run(
        [sys.executable, "verify_setup.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if verify_result.returncode != 0:
        print(f"WARNING: verify_setup.py returned exit code {verify_result.returncode}")

    # Step 2: run live comparison
    script = PROJECT_ROOT / "slm-comparison" / "generate_live_comparison.py"
    if not script.exists():
        print(f"ERROR: missing script: {script}")
        return 1

    print(f"[run_all] Running: {script}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        return result.returncode

    print("\n[run_all] Validating artifacts...")
    missing = [p for p in EXPECTED_ARTIFACTS if not p.exists()]
    if missing:
        print("ERROR: missing artifacts after run:")
        for p in missing:
            print(f"  - {p}")
        return 1

    print("[run_all] All artifacts present:")
    for p in EXPECTED_ARTIFACTS:
        size = p.stat().st_size
        print(f"  OK: {p.relative_to(PROJECT_ROOT)} ({size:,} bytes)")

    print("\n[run_all] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
