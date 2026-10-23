"""Unattended pipeline: wait for stage-1 QLoRA, then run stage-2 steering
fine-tune, behavioral eval, and fp16 merge.

Launch detached:
    cmd /c "...pythonw.exe run_pipeline_after_stage1.py > pipeline.log 2>&1"
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
PYW = HERE / ".venv" / "Scripts" / "pythonw.exe"  # avoid console-subsystem fortran WM_CLOSE crashes
STAGE1_MARKER = "done; adapter saved to adapter"
STAGE1_ERR = HERE / "train_qlora.err.log"
STAGE1_LOG = HERE / "train_qlora.log"
STAGE2_OUT = HERE / "adapter_stage2"
MERGED_OUT = HERE / "merged_fp16"
MAX_WAIT = 11 * 3600


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tail(path: str, n: int = 30) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.readlines()[-n:]
    except OSError:
        return []


def run(cmd_list: list[str], tag: str) -> bool:
    log(f"START {tag}")
    if len(cmd_list) > 1 and cmd_list[0] == str(PY):
        cmd_list = [str(PYW)] + cmd_list[1:]  # GUI subsystem: no forrtl WM_CLOSE kills
    proc = subprocess.run(
        cmd_list, cwd=str(HERE), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    (HERE / f"{tag}.out.log").write_text(
        "STDOUT:\n" + (proc.stdout or "") + "\nSTDERR:\n" + (proc.stderr or ""),
        encoding="utf-8",
    )
    ok = proc.returncode == 0
    log(f"{'DONE' if ok else 'FAILED'} {tag}")
    return ok


def stage1_complete() -> str | None:
    for line in tail(str(STAGE1_LOG), 6):
        if STAGE1_MARKER in line:
            return "done"
        if "Traceback" in line or "CUDA out of memory" in line:
            return "crashed"
    return None


def run_chain() -> None:
    """Stage-2 -> behavioral eval -> merge. Assumes stage-1 adapter exists."""
    log("stage-1 complete")
    time.sleep(60)  # let the GPU unload

    if not (HERE / "data" / "stage2_sft.jsonl").exists():
        run([str(PY), "steering/build_steering.py"], "build_steering")

    if not (STAGE2_OUT / "TRAIN_COMPLETE").exists():
        ok = run([
            str(PY), "train_qlora.py",
            "--data", "data/stage2_sft.jsonl",
            "--val", "data/sft_val.jsonl",
            "--epochs", "1", "--lr", "5e-5",
            "--accum", "8",
            "--init-adapter", "adapter",
            "--out", "adapter_stage2",
        ], "stage2")
        if not ok:
            sys.exit(4)

    ok = run([
        str(PY), "evaluate_ft.py",
        "--adapter", "adapter_stage2",
        "--gpu",
    ], "behavioral_eval")

    if not (MERGED_OUT / "model.safetensors.index.json").exists():
        run([
            str(PY), "merge_adapter.py",
            "--adapter", "adapter_stage2",
            "--out", "merged_fp16",
        ], "merge")

    log("pipeline finished" + ("" if ok else " (behavioral eval FAILED — inspect!)"))


def main() -> None:
    # 1. wait for stage-1
    waited = 0
    while True:
        state = stage1_complete()
        if state == "done" or state == "crashed":
            break
        if waited >= MAX_WAIT:
            log("stage-1 wait timeout; aborting pipeline")
            sys.exit(2)
        time.sleep(240)
        waited += 240
        if waited % 960 == 0:
            tl = tail(str(STAGE1_LOG), n=1)
            log("waiting for stage-1: " + (tl[0].strip() if tl else "?"))
    if state == "crashed":
        log("stage-1 crashed; refusing to continue with fresh/usable artifacts")
        sys.exit(3)
    if not (HERE / "adapter" / "adapter_model.safetensors").exists():
        log("stage-1 adapter missing; stop")
        sys.exit(3)
    run_chain()


if __name__ == "__main__":
    main()
