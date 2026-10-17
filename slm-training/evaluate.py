"""
Phase 4.1: Convergence analysis over the recorded loss history.

Reads losses.json and training_stats.json, checks the PRD's convergence
criteria, renders the loss curves, and writes evaluation_report.json.

Usage
-----
    python evaluate.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slmlog import log, warn, error, phase, human, fmt_dur

HERE = os.path.dirname(os.path.abspath(__file__))
LOSSES_PATH = os.path.join(HERE, "losses.json")
STATS_PATH = os.path.join(HERE, "training_stats.json")
REPORT_PATH = os.path.join(HERE, "evaluation_report.json")
CURVE_PATH = os.path.join(HERE, "loss_curves.png")

TRAIN_LOSS_TARGET = 3.0
VAL_LOSS_TARGET = 3.5
PLATEAU_WINDOW_STEPS = 1000
PLATEAU_DELTA = 0.1


def plot_curves(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        warn("matplotlib unavailable ({}); skipping plot.".format(e))
        return None

    steps = history["step"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(steps, history["train_loss"], label="train", lw=1.8)
    ax.plot(steps, history["val_loss"], label="validation", lw=1.8)
    ax.axhline(TRAIN_LOSS_TARGET, ls="--", c="grey", lw=1,
               label="target ({})".format(TRAIN_LOSS_TARGET))
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ppl_t = [math.exp(min(20, v)) for v in history["train_loss"]]
    ppl_v = [math.exp(min(20, v)) for v in history["val_loss"]]
    ax.plot(steps, ppl_t, label="train", lw=1.8)
    ax.plot(steps, ppl_v, label="validation", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity (log scale)")
    ax.set_title("Perplexity")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Code-Aware SLM training", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return CURVE_PATH


def main():
    phase("Phase 4.1: Convergence Analysis")
    if not os.path.exists(LOSSES_PATH):
        error("losses.json not found - has training produced an eval yet?")
        return 1

    with open(LOSSES_PATH, encoding="utf-8") as fh:
        history = json.load(fh)
    stats = {}
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, encoding="utf-8") as fh:
            stats = json.load(fh)

    steps = history.get("step", [])
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    if not steps:
        error("Loss history is empty.")
        return 1

    final_train = train_losses[-1]
    final_val = val_losses[-1]
    best_val = min(val_losses)
    best_val_step = steps[val_losses.index(best_val)]

    # Divergence: any eval-to-eval jump of more than 0.5 nats upward.
    spikes = [(steps[i], round(val_losses[i] - val_losses[i - 1], 3))
              for i in range(1, len(val_losses))
              if val_losses[i] - val_losses[i - 1] > 0.5]

    # Plateau: total train-loss movement across the last PLATEAU_WINDOW_STEPS.
    window = [(s, l) for s, l in zip(steps, train_losses)
              if s > steps[-1] - PLATEAU_WINDOW_STEPS]
    plateau_delta = (max(l for _, l in window) - min(l for _, l in window)
                     if len(window) > 1 else float("nan"))

    # Overall trend: is the curve generally descending?
    descending = sum(1 for i in range(1, len(train_losses))
                     if train_losses[i] <= train_losses[i - 1])
    monotonic_frac = descending / max(1, len(train_losses) - 1)

    checks = {
        "train_loss_below_{}".format(TRAIN_LOSS_TARGET): final_train < TRAIN_LOSS_TARGET,
        "val_loss_below_{}".format(VAL_LOSS_TARGET): final_val < VAL_LOSS_TARGET,
        "no_divergence": len(spikes) == 0,
        "loss_plateaued": bool(len(window) > 1 and plateau_delta < PLATEAU_DELTA),
        "generally_decreasing": monotonic_frac >= 0.7,
    }

    report = {
        "steps_evaluated": len(steps),
        "last_step": steps[-1],
        "final_train_loss": round(final_train, 4),
        "final_val_loss": round(final_val, 4),
        "final_train_perplexity": round(math.exp(min(20, final_train)), 2),
        "final_val_perplexity": round(math.exp(min(20, final_val)), 2),
        "best_val_loss": round(best_val, 4),
        "best_val_step": best_val_step,
        "best_val_perplexity": round(math.exp(min(20, best_val)), 2),
        "first_train_loss": round(train_losses[0], 4),
        "total_loss_reduction": round(train_losses[0] - final_train, 4),
        "monotonic_fraction": round(monotonic_frac, 3),
        "plateau_delta_last_{}_steps".format(PLATEAU_WINDOW_STEPS):
            None if math.isnan(plateau_delta) else round(plateau_delta, 4),
        "divergence_spikes": spikes,
        "checks": checks,
        "converged": all(checks.values()),
        "training_stats": stats,
    }

    log("Steps evaluated       : {} (last step {})".format(len(steps), steps[-1]))
    log("Train loss            : {:.4f} -> {:.4f}  (reduction {:.4f})".format(
        train_losses[0], final_train, train_losses[0] - final_train))
    log("Val loss (final/best) : {:.4f} / {:.4f} @ step {}".format(
        final_val, best_val, best_val_step))
    log("Perplexity (train/val): {:.1f} / {:.1f}".format(
        report["final_train_perplexity"], report["final_val_perplexity"]))
    if stats:
        log("Training time         : {} ({} tokens/s, {} tokens seen)".format(
            stats.get("wall_human", "?"), stats.get("tokens_per_second", "?"),
            human(stats.get("tokens_seen", 0))))
        log("NaN events / OOM      : {} / {}".format(
            stats.get("nan_events", 0), stats.get("oom_events", 0)))
    log("")
    log("Convergence criteria:")
    for name, ok in checks.items():
        log("  [{}] {}".format("PASS" if ok else "FAIL", name))

    path = plot_curves(history)
    if path:
        log("Loss curves written to {}".format(os.path.basename(path)))
        report["loss_curve_png"] = os.path.basename(path)

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if report["converged"]:
        log("RESULT: converged - all criteria met.")
    else:
        failed = [k for k, v in checks.items() if not v]
        warn("RESULT: not fully converged. Unmet: {}".format(", ".join(failed)))
    log("Phase 4.1: Complete -> evaluation_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
