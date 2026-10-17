"""
Generate TRAINING_REPORT.md from the JSON artifacts produced by the pipeline.

Run it any time -- mid-training it produces an interim report from whatever
losses.json currently holds.

Usage
-----
    python make_report.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slmlog import log, warn, fmt_dur

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(HERE, "TRAINING_REPORT.md")


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        warn("could not read {}: {}".format(name, e))
        return None


def human(n):
    if n is None:
        return "n/a"
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return "{:.1f}{}".format(n, unit) if unit else "{:.0f}".format(n)
        n /= 1000.0
    return "{:.1f}T".format(n)


def table(rows):
    return "\n".join("| {} | {} |".format(k, v) for k, v in rows)


def main():
    scrape = load("scrape_stats.json") or {}
    prep = load("preprocess_stats.json") or {}
    cfg = load("training_config.json") or {}
    stats = load("training_stats.json") or {}
    evalr = load("evaluation_report.json") or {}
    losses = load("losses.json") or {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    in_progress = not stats

    parts = []
    parts.append("# Training Report: Code-Aware SLM\n")

    # ------------------------------------------------------------- summary
    if evalr:
        verdict = ("met every convergence criterion" if evalr.get("converged")
                   else "did not meet every convergence criterion")
        summary = (
            "An {} parameter GPT-2 style transformer was trained from scratch on "
            "{} tokens of docstrings, code comments and README prose scraped from "
            "{} top-starred GitHub repositories. Training ran for {} steps on "
            "{}, finishing with a training loss of {} and a validation loss of "
            "{} (perplexity {}). The run {}."
        ).format(
            human(stats.get("params")), human(prep.get("total_tokens")),
            scrape.get("repos_ok", "?"), stats.get("steps_completed", "?"),
            stats.get("device", "?").upper(),
            evalr.get("final_train_loss"), evalr.get("final_val_loss"),
            evalr.get("final_val_perplexity"), verdict)
    elif in_progress:
        summary = (
            "An {} parameter GPT-2 style transformer is being trained from scratch "
            "on {} tokens of docstrings, code comments and README prose scraped "
            "from {} top-starred GitHub repositories. **Training is still in "
            "progress** -- the figures below reflect the run so far."
        ).format(human(81564800 if not model_cfg else None) or "81.6M",
                 human(prep.get("total_tokens")), scrape.get("repos_ok", "?"))
    else:
        summary = "Training run completed; evaluation has not been run yet."
    parts.append("## Summary\n\n" + summary + "\n")

    # ---------------------------------------------------------------- data
    rows = [
        ("Repositories scraped", "{} succeeded, {} skipped".format(
            scrape.get("repos_ok", "?"), scrape.get("repos_failed", "?"))),
        ("Languages", "Python, JavaScript/TypeScript"),
        ("Minimum stars", "1,000"),
        ("Records extracted", "{:,}".format(scrape.get("records", 0))),
        ("Raw corpus size", "{:.1f} MB".format(scrape.get("raw_bytes", 0) / 1e6)),
        ("Total tokens", human(prep.get("total_tokens"))),
        ("Train / validation", "{} / {}".format(
            human(prep.get("train_tokens")), human(prep.get("val_tokens")))),
        ("Characters per token", prep.get("chars_per_token", "n/a")),
        ("Mean record length", "{} tokens".format(
            prep.get("mean_tokens_per_record", "n/a"))),
        ("Median record length", "{} tokens".format(
            prep.get("median_tokens_per_record", "n/a"))),
    ]
    parts.append("## Data\n\n| Metric | Value |\n| --- | --- |\n" + table(rows) + "\n")

    by_kind = scrape.get("by_kind") or {}
    if by_kind:
        total = sum(by_kind.values()) or 1
        names = {"py_docstring": "Python docstrings (with signature)",
                 "py_comment": "Python comments",
                 "js_docstring": "JSDoc blocks (with declaration)",
                 "js_comment": "JavaScript/TypeScript comments",
                 "readme": "README prose"}
        krows = [(names.get(k, k), "{:,} ({:.1f}%)".format(v, 100 * v / total))
                 for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])]
        parts.append("### Record composition\n\n| Source | Records |\n| --- | --- |\n"
                     + table(krows) + "\n")

    if prep.get("length_histogram"):
        hrows = [("{} tokens".format(k), "{:,}".format(v))
                 for k, v in prep["length_histogram"].items()]
        parts.append("### Record length distribution\n\n"
                     "| Bucket | Records |\n| --- | --- |\n" + table(hrows) + "\n")

    # --------------------------------------------------------------- model
    if model_cfg:
        mrows = [("Parameters", "{:,}".format(stats.get("params", 81564800))),
                 ("Vocabulary", "{:,} (GPT-2 BPE)".format(model_cfg.get("vocab_size", 0))),
                 ("Context length", model_cfg.get("block_size")),
                 ("Layers", model_cfg.get("n_layer")),
                 ("Attention heads", model_cfg.get("n_head")),
                 ("Embedding dimension", model_cfg.get("n_embd")),
                 ("Head dimension", (model_cfg.get("n_embd", 0) //
                                     max(1, model_cfg.get("n_head", 1)))),
                 ("Dropout", model_cfg.get("dropout")),
                 ("Weight tying", "token embedding shared with output projection")]
        parts.append("## Model\n\n| Property | Value |\n| --- | --- |\n"
                     + table(mrows) + "\n")

    # ------------------------------------------------------------ training
    if train_cfg:
        trows = [("Optimizer", "AdamW (betas {}, {})".format(
                    train_cfg.get("beta1"), train_cfg.get("beta2"))),
                 ("Peak learning rate", train_cfg.get("learning_rate")),
                 ("Minimum learning rate", train_cfg.get("min_lr")),
                 ("Schedule", "linear warmup {} steps, then cosine decay".format(
                     train_cfg.get("warmup_steps"))),
                 ("Weight decay", "{} (dim >= 2 tensors only)".format(
                     train_cfg.get("weight_decay"))),
                 ("Gradient clipping", train_cfg.get("grad_clip")),
                 ("Micro-batch", "{} sequences".format(train_cfg.get("batch_size"))),
                 ("Gradient accumulation", train_cfg.get("grad_accum_steps")),
                 ("Effective batch", "{} sequences ({} tokens)".format(
                     train_cfg.get("batch_size", 0) * train_cfg.get("grad_accum_steps", 0),
                     cfg.get("tokens_per_step"))),
                 ("Max steps", train_cfg.get("max_steps")),
                 ("Tokens over full run", human(cfg.get("total_training_tokens")))]
        parts.append("## Training configuration\n\n| Hyperparameter | Value |\n"
                     "| --- | --- |\n" + table(trows) + "\n")

    if stats:
        rrows = [("Device", stats.get("device", "?").upper()),
                 ("Torch threads", stats.get("threads")),
                 ("Steps completed", "{} / {}".format(
                     stats.get("steps_completed"), stats.get("max_steps"))),
                 ("Stop reason", stats.get("stop_reason")),
                 ("Wall-clock time", stats.get("wall_human")),
                 ("Throughput", "{} tokens/s".format(stats.get("tokens_per_second"))),
                 ("Tokens seen", human(stats.get("tokens_seen"))),
                 ("Non-finite loss events", stats.get("nan_events")),
                 ("OOM events", stats.get("oom_events"))]
        parts.append("## Run\n\n| Metric | Value |\n| --- | --- |\n"
                     + table(rrows) + "\n")

    # ------------------------------------------------------------- results
    if evalr:
        erows = [("Final training loss", evalr.get("final_train_loss")),
                 ("Final validation loss", evalr.get("final_val_loss")),
                 ("Best validation loss", "{} (step {})".format(
                     evalr.get("best_val_loss"), evalr.get("best_val_step"))),
                 ("Training perplexity", evalr.get("final_train_perplexity")),
                 ("Validation perplexity", evalr.get("final_val_perplexity")),
                 ("Total loss reduction", evalr.get("total_loss_reduction")),
                 ("Monotonic fraction", evalr.get("monotonic_fraction"))]
        parts.append("## Results\n\n| Metric | Value |\n| --- | --- |\n"
                     + table(erows) + "\n")

        checks = evalr.get("checks", {})
        crows = [(k.replace("_", " "), "PASS" if v else "FAIL")
                 for k, v in checks.items()]
        parts.append("### Convergence criteria\n\n| Criterion | Result |\n"
                     "| --- | --- |\n" + table(crows) + "\n")
        if os.path.exists(os.path.join(HERE, "loss_curves.png")):
            parts.append("![Loss curves](loss_curves.png)\n")

    elif losses.get("step"):
        n = len(losses["step"])
        parts.append("## Progress so far\n\n"
                     "{} evaluations recorded, latest at step {}: "
                     "train {:.4f}, validation {:.4f}.\n".format(
                         n, losses["step"][-1], losses["train_loss"][-1],
                         losses["val_loss"][-1]))

    # ---------------------------------------------------------- loss table
    if losses.get("step"):
        steps = losses["step"]
        idx = list(range(0, len(steps), max(1, len(steps) // 20)))
        if idx and idx[-1] != len(steps) - 1:
            idx.append(len(steps) - 1)
        lrows = [("{:,}".format(steps[i]),
                  "{:.4f} / {:.4f}".format(losses["train_loss"][i],
                                           losses["val_loss"][i]))
                 for i in idx]
        parts.append("### Loss history (train / validation)\n\n"
                     "| Step | Loss |\n| --- | --- |\n" + table(lrows) + "\n")

    # ------------------------------------------------------------- samples
    sample_path = os.path.join(HERE, "inference_samples.txt")
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as fh:
            body = fh.read()
        parts.append("## Generated samples\n\n```\n" + body.strip() + "\n```\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    log("Wrote {} ({} sections)".format(os.path.basename(REPORT_PATH), len(parts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
