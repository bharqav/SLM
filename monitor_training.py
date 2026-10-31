#!/usr/bin/env python3
"""Monitor training and run evaluation when complete."""

import ctypes
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_LOG = PROJECT_ROOT / "slm-training" / "train.log"
TRAIN_PID_FILE = PROJECT_ROOT / "slm-training" / "train.pid"
CHECKPOINT_PATTERN = "checkpoint_step_*.pt"


def get_last_step():
    """Get the last step number from training log."""
    try:
        with open(TRAIN_LOG, encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "Step " in line and " | Loss: " in line:
                parts = line.split("|")
                step_part = parts[0].strip()
                step_num = int(step_part.split("Step ")[1].strip())
                return step_num
    except Exception:
        pass
    return None


def is_training_running():
    """Check if training process is running using PID file (Windows)."""
    if not TRAIN_PID_FILE.exists():
        return False
    try:
        pid = int(TRAIN_PID_FILE.read_text(encoding="utf-8").strip())
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        SYNCHRONIZE = 0x100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def run_evaluation():
    """Run evaluation scripts."""
    print("\n[monitor] Training appears complete. Running evaluation...")

    # Run evaluate_test_split if available
    eval_script = PROJECT_ROOT / "evaluate_test_split.py"
    if eval_script.exists():
        print(f"[monitor] Running {eval_script}")
        result = subprocess.run(
            [sys.executable, str(eval_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=600,
        )
        if result.returncode != 0:
            print(
                f"[monitor] WARNING: evaluate_test_split failed with {result.returncode}"
            )

    # Run live comparison
    comparison_script = PROJECT_ROOT / "slm-comparison" / "generate_live_comparison.py"
    if comparison_script.exists():
        print(f"[monitor] Running {comparison_script}")
        result = subprocess.run(
            [sys.executable, str(comparison_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[monitor] WARNING: comparison failed with {result.returncode}")

    # Run benchmark
    bench_script = PROJECT_ROOT / "benchmarks" / "bench_slm_model_families.py"
    if bench_script.exists():
        print("[monitor] Running benchmark")
        result = subprocess.run(
            [
                sys.executable,
                str(bench_script),
                "--slm-code-checkpoint",
                str(PROJECT_ROOT / "slm-training" / "best_model.pt"),
                "--slm-empathetic-checkpoint",
                str(PROJECT_ROOT / "slm-training" / "best_empathy_model_v2.pt"),
                "--max-tokens",
                "40",
                "--json-out",
                str(PROJECT_ROOT / "benchmarks" / "benchmark_report.json"),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[monitor] WARNING: benchmark failed with {result.returncode}")

    print("\n[monitor] Evaluation complete.")


def main():
    print("[monitor] Starting training monitor...")
    last_step = get_last_step()
    print(f"[monitor] Current last step: {last_step}")

    # If training already finished, run eval immediately
    if last_step is not None and last_step >= 2000:
        print("[monitor] Training already complete.")
        run_evaluation()
        return

    # Monitor loop
    stall_count = 0
    max_stall = 30  # 30 checks * 10s = 5 minutes

    while True:
        time.sleep(10)
        current_step = get_last_step()

        if current_step is None:
            stall_count += 1
            if stall_count >= max_stall:
                print("[monitor] No training progress detected. Running evaluation...")
                run_evaluation()
                break
            continue

        if current_step != last_step:
            print(f"[monitor] Step {current_step} (loss improving...)")
            last_step = current_step
            stall_count = 0

        # Check if training reached max_steps
        if current_step >= 2000:
            print(f"[monitor] Training complete at step {current_step}")
            run_evaluation()
            break

        # Check if process is still running
        if not is_training_running():
            print(f"[monitor] Training process ended at step {current_step}")
            run_evaluation()
            break

        stall_count = 0


if __name__ == "__main__":
    main()
