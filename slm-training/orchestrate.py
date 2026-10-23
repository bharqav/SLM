"""
Wait for the running scrape to finish, then run validation, preprocessing,
a clean throughput benchmark, and finally launch training in the background.

Exists so the multi-day training run starts the moment data is ready rather
than whenever someone next checks on it.

Usage
-----
    python orchestrate.py --wait-for-scrape
    python orchestrate.py --skip-wait          # data already collected
"""
import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slmlog import log, warn, error, phase, fmt_dur

HERE = os.path.dirname(os.path.abspath(__file__))
SCRAPE_LOG = os.path.join(HERE, "scrape_run.log")
DONE_MARKER = "Phase 1: Complete"
PY = sys.executable


def run(script, *cli_args, check=True):
    cmd = [PY, os.path.join(HERE, script)] + list(cli_args)
    log("$ {}".format(" ".join([os.path.basename(c) for c in cmd[1:]])))
    proc = subprocess.run(cmd, cwd=HERE)
    if check and proc.returncode != 0:
        error("{} exited {}".format(script, proc.returncode))
    return proc.returncode


SEC_PER_STEP_RE = re.compile(r"Benchmark:\s*([\d.]+)s/step")


def benchmark_threads(steps, candidates):
    """Time a few optimizer steps at each thread count; return the fastest.

    Worth doing once: this box has 4 physical / 8 logical cores, and torch's
    default is not reliably the best of those on a memory-bound workload. The
    sweep runs after scraping, so nothing else is competing for the CPU.
    """
    results = {}
    for n in candidates:
        cmd = [PY, os.path.join(HERE, "train.py"),
               "--benchmark", str(steps), "--threads", str(n)]
        log("$ train.py --benchmark {} --threads {}".format(steps, n))
        try:
            proc = subprocess.run(cmd, cwd=HERE, capture_output=True,
                                  text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            warn("thread sweep at {} threads timed out".format(n))
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        m = SEC_PER_STEP_RE.search(out)
        if not m:
            warn("no benchmark line for {} threads (exit {})".format(
                n, proc.returncode))
            continue
        results[n] = float(m.group(1))
        log("  {} threads -> {:.1f}s/step".format(n, results[n]))

    if not results:
        warn("thread sweep produced no usable result; using default")
        return None
    best = min(results, key=results.get)
    worst = max(results.values())
    log("Best: {} threads at {:.1f}s/step ({:.0%} faster than worst)".format(
        best, results[best], 1 - results[best] / worst if worst else 0))
    return best


def wait_for_scrape(offset, poll=30, timeout=6 * 3600):
    """Block until DONE_MARKER appears in scrape_run.log past `offset`."""
    log("Waiting for scrape to finish (polling every {}s) ...".format(poll))
    started = time.time()
    while time.time() - started < timeout:
        try:
            with open(SCRAPE_LOG, encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                if DONE_MARKER in fh.read():
                    log("Scrape finished after {}".format(
                        fmt_dur(time.time() - started)))
                    return True
        except FileNotFoundError:
            pass
        time.sleep(poll)
    warn("Timed out waiting for scrape after {}".format(fmt_dur(timeout)))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-scrape", action="store_true")
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--offset", type=int, default=0,
                    help="byte offset in scrape_run.log to search past")
    ap.add_argument("--benchmark-steps", type=int, default=3)
    ap.add_argument("--threads", default="4,6,8",
                    help="comma-separated thread counts to sweep; empty to skip")
    ap.add_argument("--no-train", action="store_true",
                    help="stop after the benchmark instead of launching training")
    args = ap.parse_args()

    phase("Orchestrator: post-scrape pipeline")

    if args.wait_for_scrape and not args.skip_wait:
        if not wait_for_scrape(args.offset):
            error("Scrape did not complete; aborting.")
            return 1

    run("validate_data.py", check=False)      # advisory: non-fatal issues allowed
    if run("preprocess.py") != 0:
        error("Preprocessing failed; not starting training.")
        return 1

    log("Running a clean throughput benchmark (nothing else should be running)")
    candidates = [int(x) for x in args.threads.split(",") if x.strip()]
    best_threads = None
    if candidates:
        best_threads = benchmark_threads(args.benchmark_steps, candidates)

    if args.no_train:
        log("--no-train set; stopping before training.")
        return 0

    phase("Launching training in the background")
    out = open(os.path.join(HERE, "train_run.log"), "a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | \
                        getattr(subprocess, "DETACHED_PROCESS", 0)
    train_cmd = [PY, os.path.join(HERE, "train.py")]
    if best_threads:
        train_cmd += ["--threads", str(best_threads)]
    log("$ {}".format(" ".join(os.path.basename(c) for c in train_cmd[1:])))
    proc = subprocess.Popen(train_cmd,
                            cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            creationflags=creationflags)
    log("Training started as PID {} -> train_run.log".format(proc.pid))
    log("Monitor with: tail -50 train.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
