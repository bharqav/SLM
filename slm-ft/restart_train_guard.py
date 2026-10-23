"""Relaunch the stage-1 QLoRA until it completes, then run the stage-2 /
eval / merge chain automatically. Survives occasional process crashes; a full
OS reboot still requires relaunching this script (run it via the console
command snippet below).

Launch detached (cmd /c with quoting — see README)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
PYW = HERE / ".venv" / "Scripts" / "pythonw.exe"  # GUI subsystem: no console -> no WM_CLOSE forrtl kills
STAGE1_MARKER = "done; adapter saved to"
MAX_ATTEMPTS = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] GUARD {msg}", flush=True)


def last_log_line() -> str:
    p = HERE / "train_qlora.log"
    try:
        p.write_text("", encoding="utf-8")  # truncate for a clean attempt
    except OSError:
        pass
    return ""


def attempt_progress(lines: list[str]) -> str:
    covers = [l.strip() for l in lines if l.startswith("checkpoint at") or l.startswith("done;")]
    return covers[-1] if covers else "?"


def main() -> None:
    last_ckpt = (HERE / "adapter" / "last_ckpt" / "adapter_model.safetensors").exists()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"stage-1 attempt {attempt}/{MAX_ATTEMPTS}")
        cmd = [str(PYW), "train_qlora.py", "--epochs", "2", "--out", "adapter"]
        ck = HERE / "adapter" / "last_ckpt"
        if last_ckpt and (ck / "adapter_model.safetensors").exists():
            cmd += ["--resume-from", str(ck)]
            log(f"resuming from {ck}")
        with open(HERE / "train_qlora.log", "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=str(HERE), stdout=logf, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        rc = proc.returncode if proc is not None else 1
        marker_ok = (HERE / "adapter" / "TRAIN_COMPLETE").exists()
        if rc == 0 and marker_ok:
            log("stage-1 finished cleanly")
            break
        if (ck / "adapter_model.safetensors").exists():
            last_ckpt = True
            log(f"attempt {attempt} rc={rc}; checkpoint exists, will resume")
        else:
            log(f"attempt {attempt} rc={rc}; no checkpoint, restarting fresh")
        time.sleep(180)
    else:
        log("exhausted attempts; aborting")
        sys.exit(3)

    import run_pipeline_after_stage1 as pipe
    try:
        pipe.run_chain()
    except SystemExit as e:
        log(f"pipeline chain exited: {e.code}")
        sys.exit(e.code if isinstance(e.code, int) else 5)


if __name__ == "__main__":
    main()
