"""Wait for the slm-ft QLoRA chain to finish, then train the from-scratch
empathetic SLM variant (slm-training code in an isolated run dir).

Launch detached:
    cmd /c "...pythonw.exe C:\\...\\launch_empathetic_after_qlora.py
    > C:\\...\\launch.log 2>&1"

Queue condition: slm-ft/merged_fp16/model.safetensors.index.json exists AND
slm-ft/adapter_stage2/TRAIN_COMPLETE exists (chain done), then wait for GPU
freedom (memory.used < 2GB) and start the ~2000-step GPT run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SLM_FT = Path(r"C:\SubD\Transfer\Projects\SLM\slm-ft")
RUN_DIR = Path(r"C:\SubD\Transfer\Projects\SLM\slm-training\empathetic_run")
PY = RUN_DIR.parent / ".venv" / "Scripts" / "python.exe"
PYW = RUN_DIR.parent / ".venv" / "Scripts" / "pythonw.exe"  # no console -> no forrtl WM_CLOSE kills
MERGE_DONE = SLM_FT / "merged_fp16" / "model.safetensors"          # single-shard save
MERGE_DONE_SHARDED = SLM_FT / "merged_fp16" / "model.safetensors.index.json"
STAGE2_DONE = SLM_FT / "adapter_stage2" / "TRAIN_COMPLETE"
MAX_WAIT = 12 * 3600


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_free_mib() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 999999


def main() -> None:
    waited = 0
    log("waiting for the qlora chain (stage-2 + eval + merge) ...")
    while not (MERGE_DONE.exists() and STAGE2_DONE.exists()):
        if waited >= MAX_WAIT:
            log("timeout waiting; aborting")
            sys.exit(2)
        time.sleep(240)
        waited += 240
        if waited % 1200 == 0:
            log(f"still waiting ({waited // 60} min); merged={(MERGE_DONE.exists() or MERGE_DONE_SHARDED.exists())} "
                f"stage2={STAGE2_DONE.exists()}")
    log("qlora chain finished; letting the GPU settle")
    time.sleep(120)

    budget = 0
    while gpu_free_mib() > 2000:
        if budget >= 3600:
            log("GPU never freed; aborting")
            sys.exit(3)
        time.sleep(120)
        budget += 120

    log("starting empathetic SLM training (2000 steps)")
    cmd = [
        str(PY), "train.py",
        "--max-steps", "2000",
        "--learning-rate", "3e-4",
        "--warmup-steps", "200",
        "--checkpoint-interval", "100",
        "--eval-interval", "200",
        "--patience-steps", "800",
    ]
    with open(RUN_DIR / "train.log", "w", encoding="utf-8") as out:
        rc = subprocess.run([str(PYW)] + cmd[1:], cwd=str(RUN_DIR), stdout=out,
                            stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace").returncode
    if rc != 0:
        log(f"training FAILED rc={rc}; inspect empathetic_run/train.log")
        sys.exit(rc)

    # quick sample check with the trained artifact
    sample = subprocess.run(
        [str(PY), "inference.py", "--prompt", "<sys>You are a caring peer-support companion. Listen first, validate feelings, suggest healthy coping steps. </sys><user>hi, feeling pretty hopeless today</user>"],
        cwd=str(RUN_DIR), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )
    (RUN_DIR / "samples.log").write_text(sample.stdout or "", encoding="utf-8")
    log("training complete; sample written to empathetic_run/samples.log")


if __name__ == "__main__":
    main()
