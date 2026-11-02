#!/usr/bin/env python3
"""Auto-training loop: train -> evaluate -> retrain if needed."""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_LOG = PROJECT_ROOT / "slm-training" / "train.log"
TRAIN_PID_FILE = PROJECT_ROOT / "slm-training" / "train.pid"
BEST_EMPATHY = PROJECT_ROOT / "slm-training" / "best_empathy_model_v2.pt"
TRAIN_SCRIPT = PROJECT_ROOT / "slm-training" / "train.py"


def cleanup_pid_file() -> None:
    """Remove PID file if training is no longer running."""
    if TRAIN_PID_FILE.exists():
        try:
            if not is_training_running():
                TRAIN_PID_FILE.unlink()
        except Exception:
            pass


def get_last_step() -> int | None:
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


def is_training_running() -> bool:
    """Check if training process is running using PID file."""
    if not TRAIN_PID_FILE.exists():
        return False
    try:
        pid = int(TRAIN_PID_FILE.read_text(encoding="utf-8").strip())
        # Check if process exists (Windows-specific). `windll` only exists on
        # Windows; getattr keeps this importable and type-checkable elsewhere
        # (AttributeError is caught below).
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32
        SYNCHRONIZE = 0x100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def wait_for_training(max_steps: int | None = None, timeout: float = 7200) -> bool:
    """Wait for training to complete."""
    print("[auto_train] Waiting for training to complete...")
    start = time.time()
    last_step = get_last_step()
    stall_count = 0

    # Read max_steps from training config if not provided
    if max_steps is None:
        try:
            with open(PROJECT_ROOT / "slm-training" / "training_config.json") as f:
                import json

                config = json.load(f)
                configured = config.get("training", {}).get("max_steps")
                max_steps = configured if isinstance(configured, int) else 2000
        except Exception:
            max_steps = 2000

    while time.time() - start < timeout:
        time.sleep(20)
        current_step = get_last_step()

        if current_step is None:
            stall_count += 1
            if stall_count > 10:
                print("[auto_train] No training progress detected")
                return False
            continue

        if current_step != last_step:
            print(f"[auto_train] Step {current_step}")
            last_step = current_step
            stall_count = 0

        if current_step >= max_steps:
            print(f"[auto_train] Training reached max_steps ({current_step})")
            return True

        if not is_training_running():
            print(f"[auto_train] Training process ended at step {current_step}")
            return current_step >= max_steps

    print("[auto_train] Timeout waiting for training")
    return False


def run_evaluation() -> None:
    """Run evaluation and comparison."""
    print("\n[auto_train] === Running Evaluation ===")

    scripts = [
        PROJECT_ROOT / "slm-comparison" / "generate_live_comparison.py",
        PROJECT_ROOT / "benchmarks" / "bench_slm_model_families.py",
    ]

    for script in scripts:
        if not script.exists():
            print(f"[auto_train] SKIP: {script.name} not found")
            continue

        print(f"[auto_train] Running {script.name}...")
        try:
            cmd = [sys.executable, str(script)]
            if "bench_slm" in script.name:
                cmd += [
                    "--slm-code-checkpoint",
                    "slm-training/best_model.pt",
                    "--slm-empathetic-checkpoint",
                    "slm-training/best_empathy_model_v2.pt",
                    "--max-tokens",
                    "40",
                    "--json-out",
                    "benchmarks/benchmark_report.json",
                ]
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                timeout=600,
            )
            if result.returncode != 0:
                print(
                    f"[auto_train] WARNING: {script.name} failed ({result.returncode})"
                )
        except subprocess.TimeoutExpired:
            print(f"[auto_train] TIMEOUT: {script.name}")
        except Exception as e:
            print(f"[auto_train] ERROR: {script.name}: {e}")


def get_best_val_loss() -> float | None:
    """Extract best val loss from training log."""
    try:
        with open(TRAIN_LOG, encoding="utf-8") as f:
            content = f.read()
        # Find the last "new best" line
        for line in reversed(content.splitlines()):
            if "new best" in line and "Val loss" in line:
                parts = line.split("Val loss: ")
                val_loss_part = parts[1].split(" ")[0]
                return float(val_loss_part)
    except Exception:
        pass
    return None


def start_training(
    max_steps: int = 2000,
    resume: bool = False,
    config_override: list[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start a new training run."""
    print(
        f"\n[auto_train] === Starting Training (max_steps={max_steps}, resume={resume}) ==="
    )

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--train-bin",
        str(PROJECT_ROOT / "slm-training" / "empathy_train.bin"),
        "--val-bin",
        str(PROJECT_ROOT / "slm-training" / "empathy_val.bin"),
        "--n-layer",
        "24",
        "--n-head",
        "24",
        "--n-embd",
        "1536",
        "--max-steps",
        str(max_steps),
        "--batch-size",
        "4",
        "--grad-accum-steps",
        "8",
        "--learning-rate",
        "1.5e-4",
        "--min-lr",
        "1.5e-5",
        "--warmup-steps",
        "100",
        "--eval-interval",
        "200",
        "--checkpoint-interval",
        "200",
        "--log-interval",
        "25",
        "--gradient-checkpointing",
        "--best-path",
        str(BEST_EMPATHY),
        "--data-dir",
        "slm-training",
    ]

    if resume:
        cmd.append("--resume")

    if config_override:
        cmd.extend(config_override)

    # Run in background
    log_file = TRAIN_LOG
    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    # Write PID file for reliable process detection
    TRAIN_PID_FILE.write_text(str(process.pid), encoding="utf-8")

    print(f"[auto_train] Training started (PID {process.pid}, log: {log_file.name})")
    return process


def main() -> None:
    print("[auto_train] Starting auto-training loop...")

    # Check if training is already running
    if is_training_running():
        print("[auto_train] Training already running, waiting for it to complete...")
        if wait_for_training():
            run_evaluation()
        cleanup_pid_file()

    # Continue training to improve the model
    iteration = 0
    best_val_loss = get_best_val_loss()
    if best_val_loss is None:
        best_val_loss = float("inf")

    print(f"[auto_train] Starting best val loss: {best_val_loss:.4f}")

    while iteration < 2:  # Max 2 additional iterations
        iteration += 1
        print(f"\n[auto_train] ========== Iteration {iteration} ==========")

        # Continue training from checkpoint with higher max_steps
        max_steps = 1000 + (iteration * 500)  # 1500, then 2000
        start_training(max_steps=max_steps, resume=True)

        # Wait for completion
        completed = wait_for_training(max_steps=max_steps, timeout=7200)

        if not completed:
            print("[auto_train] Training did not complete, running eval anyway...")

        # Run evaluation
        run_evaluation()
        cleanup_pid_file()

        # Check results
        current_val_loss = get_best_val_loss()
        if current_val_loss is not None:
            print(f"[auto_train] Best val loss: {current_val_loss:.4f}")
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print("[auto_train] New best! Saved to best_model.pt")

            # If we've reached a good loss, we can stop
            if current_val_loss < 3.5:
                print("[auto_train] Target val loss reached. Stopping.")
                break
        else:
            print("[auto_train] Could not determine val loss from log.")

        time.sleep(30)

    print("\n[auto_train] ========== Final Results ==========")
    print(f"[auto_train] Best val loss achieved: {best_val_loss:.4f}")
    print("[auto_train] Best model: slm-training/best_model.pt")
    print("[auto_train] Done.")


if __name__ == "__main__":
    main()
