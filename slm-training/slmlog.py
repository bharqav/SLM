"""Shared logging: timestamped lines to stdout and to train.log."""

import os
import sys
import threading
import time

# Windows consoles default to cp1252, which cannot encode most of what a
# language model generates. Every script in the pipeline imports this module,
# so reconfiguring here covers plain print() calls too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_LOCK = threading.Lock()
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.log")


def log(msg, severity="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts}]" if severity == "INFO" else f"[{ts}] {severity}:"
    line = f"{prefix} {msg}"
    with _LOCK:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:
            pass
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


def warn(msg):
    log(msg, "WARNING")


def error(msg):
    log(msg, "ERROR")


def critical(msg):
    log(msg, "CRITICAL")


def phase(msg):
    log("=" * 70)
    log(msg)
    log("=" * 70)


def fmt_dur(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.1f}T"
