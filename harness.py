#!/usr/bin/env python3
"""Comprehensive harness: run every verifiable step in the project.

Usage:
    python harness.py               # fast checks + tests + bench
    python harness.py --all         # include slow tests
    python harness.py --skip-model  # skip model-dependent steps

Exit code:
    0  all steps passed
    1  any step failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> int:
    print(f"\n[harness] {name}")
    print(f"[harness]   cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"[harness] FAILED: {name} (exit {result.returncode})")
    else:
        print(f"[harness] OK: {name}")
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full project harness")
    ap.add_argument("--all", action="store_true", help="Include slow tests")
    ap.add_argument(
        "--skip-model", action="store_true", help="Skip model-dependent steps"
    )
    ap.add_argument("--lint", action="store_true", help="Run ruff + mypy")
    args = ap.parse_args()

    failures = 0

    # 1. Lint (optional — only on project files we own)
    if args.lint:
        targets = [
            "harness.py",
            "run_all.py",
            "verify_setup.py",
            "evaluate_test_split.py",
            "slm_api.py",
            "tests/",
        ]
        failures += run_step(
            "ruff lint", [sys.executable, "-m", "ruff", "check"] + targets
        )
        failures += run_step("mypy typecheck", [sys.executable, "-m", "mypy"] + targets)
    else:
        print("\n[harness] Skipping lint/typecheck (pass --lint to enable)")

    # 2. Tests (fast)
    test_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    if not args.all:
        test_cmd += ["-m", "not slow"]
    failures += run_step("pytest", test_cmd)

    # 4. Setup verification
    failures += run_step("verify_setup", [sys.executable, "verify_setup.py"])

    # 5. Live comparison
    if not args.skip_model:
        bench_cmd = [
            sys.executable,
            "slm-comparison/generate_live_comparison.py",
            "--seed",
            "1337",
        ]
        failures += run_step("live comparison", bench_cmd)

    # 6. Full pipeline
    if not args.skip_model:
        failures += run_step("run_all.py", [sys.executable, "run_all.py"])

    # 7. Security scan
    failures += run_step("pip-audit", [sys.executable, "-m", "pip_audit"])

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"[harness] {failures} step(s) failed")
        return 1
    print("[harness] All steps passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
