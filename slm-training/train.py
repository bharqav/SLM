"""
Phase 3: Train the model to convergence, with checkpointing and recovery.

Designed to survive a multi-day CPU run: every checkpoint carries optimizer
state and RNG-independent progress, so `--resume` picks up exactly where an
interrupted run left off.

Usage
-----
    python train.py                      # start (or auto-resume) a run
    python train.py --resume             # force resume from latest checkpoint
    python train.py --benchmark 20       # measure throughput, then exit
"""

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig
from slmlog import critical, error, fmt_dur, human, log, phase, warn

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_BIN = os.path.join(HERE, "train.bin")
VAL_BIN = os.path.join(HERE, "val.bin")
LOSSES_PATH = os.path.join(HERE, "losses.json")
CONFIG_PATH = os.path.join(HERE, "training_config.json")
STATS_PATH = os.path.join(HERE, "training_stats.json")
BEST_PATH = os.path.join(HERE, "best_model.pt")
FINAL_PATH = os.path.join(HERE, "final_model.pt")
CKPT_GLOB = os.path.join(HERE, "checkpoint_step_*.pt")
KEEP_CHECKPOINTS = 3

_INTERRUPTED = {"flag": False}


def _record_experiment(name: str, config: str, metrics: list[str]) -> None:
    """Best-effort append to experiments.jsonl via the experiment tracker."""
    script = os.path.join(HERE, "..", "scripts", "experiment_tracker.py")
    cmd = [
        sys.executable,
        script,
        "--name",
        name,
        "--config",
        config,
        "--output",
        os.path.join(HERE, "experiments.jsonl"),
    ]
    for m in metrics:
        cmd.extend(["--metric", m])
    try:
        subprocess.run(cmd, cwd=HERE, capture_output=True, check=True)
    except Exception:
        pass


def _handle_sigint(signum, frame):
    if _INTERRUPTED["flag"]:
        raise KeyboardInterrupt
    _INTERRUPTED["flag"] = True
    warn("Interrupt received - will checkpoint and exit after this step.")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=40)
    # 200, not the 1000 originally specified: a step takes 50-100s on this CPU,
    # so 1000 steps is 14-28 hours of unsaved work on a machine that will be
    # interrupted. At ~20s to write a checkpoint this costs well under 1% of
    # throughput and bounds worst-case loss to a few hours.
    ap.add_argument("--checkpoint-interval", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument(
        "--patience-steps",
        type=int,
        default=5000,
        help="stop if val loss has not improved for this many steps",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=0, help="0 = physical cores")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="auto = bfloat16 on CUDA if supported, else float32",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="wrap the model in torch.compile (CUDA/Linux; needs a "
        "C++ compiler, unavailable on stock Windows)",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument(
        "--data-dir",
        default=HERE,
        help="Directory containing train.bin/val.bin (default: script dir)",
    )
    ap.add_argument(
        "--train-bin",
        default=None,
        help="Override path to training binary",
    )
    ap.add_argument(
        "--val-bin",
        default=None,
        help="Override path to validation binary",
    )
    ap.add_argument(
        "--best-path",
        default=None,
        help="Override path for best-model checkpoint (default: best_model.pt)",
    )
    return ap.parse_args()


# --------------------------------------------------------------- data loading
class Data:
    def __init__(
        self,
        block_size,
        batch_size,
        device,
        data_dir=HERE,
        train_bin=None,
        val_bin=None,
    ):
        train_path = train_bin or os.path.join(data_dir, "train.bin")
        val_path = val_bin or os.path.join(data_dir, "val.bin")
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            raise FileNotFoundError(
                f"{train_path}/{val_path} missing - run preprocess.py or preprocess_empathy.py"
            )
        self.train = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val = np.memmap(val_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.cuda = str(device).startswith("cuda")
        log(
            f"Data: train {human(len(self.train))} tokens | val {human(len(self.val))} tokens"
        )

    def get_batch(self, split, rng):
        arr = self.train if split == "train" else self.val
        hi = len(arr) - self.block_size - 1
        ix = rng.integers(0, hi, size=self.batch_size)
        x = np.stack([arr[i : i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([arr[i + 1 : i + 1 + self.block_size] for i in ix]).astype(
            np.int64
        )
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.cuda:
            # Staging through pinned memory lets the copy overlap compute.
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        return x.to(self.device), y.to(self.device)


def lr_at(step, args):
    """Linear warmup then cosine decay to min_lr."""
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    if step >= args.max_steps:
        return args.min_lr
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(model, data, args, rng, ctx=None):
    import contextlib

    ctx = ctx or contextlib.nullcontext()
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            x, y = data.get_batch(split, rng)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ------------------------------------------------------------- checkpointing
def save_checkpoint(path, model, opt, step, best_val, history, cfg, args, stats):
    tmp = path + ".tmp"
    torch.save(
        {
            "model": (
                model._orig_mod if hasattr(model, "_orig_mod") else model
            ).state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "history": history,
            "model_config": cfg.to_dict(),
            "train_args": vars(args),
            "stats": stats,
        },
        tmp,
    )
    os.replace(tmp, path)


def save_model_only(path, model, step, best_val, cfg, history):
    tmp = path + ".tmp"
    torch.save(
        {
            "model": (
                model._orig_mod if hasattr(model, "_orig_mod") else model
            ).state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "model_config": cfg.to_dict(),
            "history": history,
        },
        tmp,
    )
    os.replace(tmp, path)


def rotate_checkpoints(keep=KEEP_CHECKPOINTS):
    files = sorted(
        glob.glob(CKPT_GLOB), key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0])
    )
    for old in files[:-keep]:
        try:
            os.remove(old)
        except OSError as e:
            warn(f"could not delete {old}: {e}")


def latest_checkpoint():
    files = glob.glob(CKPT_GLOB)
    if not files:
        return None
    return max(files, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))


def load_checkpoint_safely(path):
    """Return the checkpoint dict, or None if the file is unreadable."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        error(f"Checkpoint {os.path.basename(path)} is corrupt ({e}).")
        return None


def main():
    args = build_args()
    signal.signal(signal.SIGINT, _handle_sigint)

    torch.manual_seed(args.seed)
    threads = args.threads or (os.cpu_count() or 4) // 2
    torch.set_num_threads(max(1, threads))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        error("--device cuda requested but no CUDA device is available.")
        return 1

    # Resolve autocast dtype. bf16 needs no loss scaler and has fp32 range,
    # so it is the right default on Ampere and newer; fp16 falls back to a
    # GradScaler. CPU stays in fp32 -- Ice Lake has no AVX512-BF16, so bf16
    # there is emulated and slower.
    if args.dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        elif device == "cuda":
            dtype = "float16"
        else:
            dtype = "float32"
    else:
        dtype = args.dtype
    amp_enabled = device == "cuda" and dtype != "float32"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

    if device == "cuda":
        # TF32 costs a little mantissa and buys a large matmul speedup.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    phase("Phase 3: Training Started")
    log(
        f"Device: {device} | torch {torch.__version__} | dtype {dtype} | threads {torch.get_num_threads()}"
    )
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(
            f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB VRAM, capability {props.major}.{props.minor})"
        )

    cfg = GPTConfig(
        vocab_size=50257,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT.from_config(cfg).to(device)

    import contextlib

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch_dtype)
        if amp_enabled
        else contextlib.nullcontext()
    )
    opt, n_decay, n_nodecay = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2)
    )

    log(
        f"Model created ({human(model.num_params())} params, {human(model.num_params(non_embedding=True))} non-embedding)"
    )
    for k, v in model.param_breakdown().items():
        log(f"  {k:<40} {v:>12,}")
    log(f"Optimizer: AdamW | decayed {n_decay:,} | undecayed {n_nodecay:,}")

    if args.compile:
        try:
            model = torch.compile(model)
            log("torch.compile enabled")
        except Exception as e:
            warn(f"torch.compile unavailable ({str(e)[:120]}); continuing eager")

    tokens_per_step = args.batch_size * args.grad_accum_steps * args.block_size
    log(
        f"Effective batch: {args.batch_size * args.grad_accum_steps} seqs x {args.block_size} tokens = {tokens_per_step} tokens/step"
    )

    data = Data(
        args.block_size,
        args.batch_size,
        device,
        data_dir=args.data_dir,
        train_bin=args.train_bin,
        val_bin=args.val_bin,
    )
    rng = np.random.default_rng(args.seed)

    start_step = 0
    best_val = float("inf")
    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "tokens_seen": [],
        "wall_seconds": [],
    }
    stats = {
        "nan_events": 0,
        "oom_events": 0,
        "lr_scale": 1.0,
        "resumed_from": None,
        "wall_seconds": 0.0,
    }

    ckpt_path = latest_checkpoint()
    if ckpt_path and args.resume:
        ck = load_checkpoint_safely(ckpt_path)
        if ck is None:
            # Corruption recovery: fall back to the previous checkpoint.
            others = sorted(
                glob.glob(CKPT_GLOB),
                key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]),
            )
            for cand in reversed(others[:-1]):
                ck = load_checkpoint_safely(cand)
                if ck:
                    ckpt_path = cand
                    warn(f"Recovered from previous checkpoint {cand}")
                    break
        if ck:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            start_step = ck["step"]
            best_val = ck.get("best_val_loss", float("inf"))
            history = ck.get("history", history)
            stats.update(ck.get("stats", {}))
            stats["resumed_from"] = os.path.basename(ckpt_path)
            log(
                f"Resumed from {os.path.basename(ckpt_path)} at step {start_step} (best val {best_val:.4f})"
            )

    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": cfg.to_dict(),
                "training": vars(args),
                "tokens_per_step": tokens_per_step,
                "total_training_tokens": tokens_per_step * args.max_steps,
            },
            fh,
            indent=2,
        )

    # Record experiment start
    _record_experiment(
        name=f"slm-scratch-step{start_step}",
        config=CONFIG_PATH,
        metrics=[
            f"step={start_step}",
            f"best_val_loss={best_val if math.isfinite(best_val) else 'inf'}",
        ],
    )

    # ------------------------------------------------------------- benchmark
    if args.benchmark:
        log(f"Benchmarking {args.benchmark} optimizer steps ...")
        model.train()
        t0 = time.time()
        for _ in range(args.benchmark):
            for _ in range(args.grad_accum_steps):
                x, y = data.get_batch("train", rng)
                with ctx:
                    _, loss = model(x, y)
                scaler.scale(loss / args.grad_accum_steps).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_step = dt / args.benchmark
        tps = tokens_per_step / per_step
        log(
            f"Benchmark: {per_step:.1f}s/step | {tps:.0f} tokens/s | projected {fmt_dur(per_step * args.max_steps)} for {args.max_steps} steps"
        )
        return 0

    # ------------------------------------------------------------ train loop
    model.train()
    opt.zero_grad(set_to_none=True)
    run_started = time.time()
    base_wall = stats.get("wall_seconds", 0.0)
    last_log = time.time()
    window_tokens = 0
    best_val_step = start_step
    stop_reason = "reached max_steps"

    step = start_step
    try:
        while step < args.max_steps:
            lr = lr_at(step, args) * stats["lr_scale"]
            for g in opt.param_groups:
                g["lr"] = lr

            accum_loss = 0.0
            try:
                for micro in range(args.grad_accum_steps):
                    x, y = data.get_batch("train", rng)
                    with ctx:
                        _, loss = model(x, y)
                    accum_loss += loss.item()
                    scaler.scale(loss / args.grad_accum_steps).backward()
            except (RuntimeError, MemoryError) as e:
                msg = str(e).lower()
                if (
                    "out of memory" in msg
                    or "cannot allocate" in msg
                    or isinstance(e, MemoryError)
                ):
                    stats["oom_events"] += 1
                    critical(f"OOM at step {step} ({e}). Halving batch size.")
                    opt.zero_grad(set_to_none=True)
                    if args.batch_size > 1:
                        args.batch_size = max(1, args.batch_size // 2)
                        args.grad_accum_steps *= 2
                    else:
                        args.block_size = max(64, args.block_size // 2)
                    data = Data(
                        args.block_size,
                        args.batch_size,
                        device,
                        data_dir=args.data_dir,
                        train_bin=args.train_bin,
                        val_bin=args.val_bin,
                    )
                    tokens_per_step = (
                        args.batch_size * args.grad_accum_steps * args.block_size
                    )
                    warn(
                        f"Retrying with batch_size={args.batch_size} grad_accum={args.grad_accum_steps} block={args.block_size}"
                    )
                    continue
                raise

            mean_loss = accum_loss / args.grad_accum_steps

            # NaN/Inf recovery: roll back to the best weights and cut the LR.
            if not math.isfinite(mean_loss):
                stats["nan_events"] += 1
                critical(f"Non-finite loss ({mean_loss}) at step {step}.")
                opt.zero_grad(set_to_none=True)
                stats["lr_scale"] *= 0.1
                restored = False
                for cand in (args.best_path or BEST_PATH, latest_checkpoint()):
                    if cand and os.path.exists(cand):
                        ck = load_checkpoint_safely(cand)
                        if ck:
                            model.load_state_dict(ck["model"])
                            if "optimizer" in ck:
                                opt.load_state_dict(ck["optimizer"])
                            step = ck.get("step", step)
                            warn(
                                "Reloaded {} and scaled LR by 0.1 (scale now {:.2g})".format(
                                    os.path.basename(cand), stats["lr_scale"]
                                )
                            )
                            restored = True
                            break
                if not restored:
                    warn("No checkpoint to roll back to; continuing at reduced LR.")
                if stats["nan_events"] >= 5:
                    stop_reason = "too many non-finite losses"
                    critical(stop_reason)
                    break
                continue

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            step += 1
            window_tokens += tokens_per_step

            if step % args.log_interval == 0:
                now = time.time()
                tps = window_tokens / max(1e-6, now - last_log)
                sec_per_step = (now - last_log) / args.log_interval
                eta = sec_per_step * (args.max_steps - step)
                log(
                    f"Step {step:05d} | Loss: {mean_loss:.4f} | Throughput: {tps:.0f} tokens/s | "
                    f"ETA: {fmt_dur(eta)} | LR: {lr:.6f}"
                )
                last_log, window_tokens = now, 0

            if step % args.eval_interval == 0:
                eval_rng = np.random.default_rng(args.seed + 99991)
                losses = estimate_loss(model, data, args, eval_rng, ctx)
                wall = base_wall + (time.time() - run_started)
                history["step"].append(step)
                history["train_loss"].append(round(losses["train"], 5))
                history["val_loss"].append(round(losses["val"], 5))
                history["lr"].append(lr)
                history["tokens_seen"].append(step * tokens_per_step)
                history["wall_seconds"].append(round(wall, 1))

                if losses["val"] < best_val:
                    improvement = best_val - losses["val"]
                    best_val = losses["val"]
                    best_val_step = step
                    save_model_only(
                        args.best_path or BEST_PATH, model, step, best_val, cfg, history
                    )
                    log(
                        "  -> Val loss: {:.4f} (new best, improved {:.4f}) | "
                        "train {:.4f} | ppl {:.1f}".format(
                            losses["val"],
                            improvement if math.isfinite(improvement) else 0.0,
                            losses["train"],
                            math.exp(min(20, losses["val"])),
                        )
                    )
                else:
                    log(
                        "  -> Val loss: {:.4f} (no improvement, best: {:.4f} "
                        "@ step {}) | train {:.4f}".format(
                            losses["val"], best_val, best_val_step, losses["train"]
                        )
                    )

                with open(LOSSES_PATH, "w", encoding="utf-8") as fh:
                    json.dump(history, fh, indent=2)

                if step - best_val_step >= args.patience_steps:
                    stop_reason = f"early stop: no val improvement for {step - best_val_step} steps"
                    log(stop_reason)
                    break

            if step % args.checkpoint_interval == 0 or _INTERRUPTED["flag"]:
                stats["wall_seconds"] = base_wall + (time.time() - run_started)
                path = os.path.join(HERE, f"checkpoint_step_{step}.pt")
                save_checkpoint(
                    path, model, opt, step, best_val, history, cfg, args, stats
                )
                rotate_checkpoints()
                log(f"Step {step:05d} | Checkpoint saved: {os.path.basename(path)}")

            if _INTERRUPTED["flag"]:
                stop_reason = "interrupted by user"
                break

    except KeyboardInterrupt:
        stop_reason = "interrupted by user (hard)"
        warn(stop_reason)
    finally:
        stats["wall_seconds"] = base_wall + (time.time() - run_started)
        path = os.path.join(HERE, f"checkpoint_step_{step}.pt")
        save_checkpoint(path, model, opt, step, best_val, history, cfg, args, stats)
        rotate_checkpoints()
        save_model_only(FINAL_PATH, model, step, best_val, cfg, history)
        with open(LOSSES_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        final_stats = {
            "steps_completed": step,
            "max_steps": args.max_steps,
            "stop_reason": stop_reason,
            "best_val_loss": None if best_val == float("inf") else round(best_val, 5),
            "best_val_step": best_val_step,
            "final_train_loss": history["train_loss"][-1]
            if history["train_loss"]
            else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "tokens_seen": step * tokens_per_step,
            "wall_seconds": round(stats["wall_seconds"], 1),
            "wall_human": fmt_dur(stats["wall_seconds"]),
            "tokens_per_second": round(
                step * tokens_per_step / max(1e-6, stats["wall_seconds"]), 1
            ),
            "nan_events": stats["nan_events"],
            "oom_events": stats["oom_events"],
            "lr_scale_final": stats["lr_scale"],
            "params": model.num_params(),
            "device": device,
            "threads": torch.get_num_threads(),
        }
        with open(STATS_PATH, "w", encoding="utf-8") as fh:
            json.dump(final_stats, fh, indent=2)

        _record_experiment(
            name=f"slm-scratch-final-step{step}",
            config=CONFIG_PATH,
            metrics=[
                f"step={step}",
                f"best_val_loss={best_val if math.isfinite(best_val) else 'inf'}",
                f"final_train_loss={final_stats['final_train_loss']}",
                f"final_val_loss={final_stats['final_val_loss']}",
                f"wall_seconds={final_stats['wall_seconds']}",
            ],
        )

        log(
            "Training stopped ({}) at step {} after {}".format(
                stop_reason, step, fmt_dur(stats["wall_seconds"])
            )
        )
        if best_val < float("inf"):
            log(f"Best val loss {best_val:.4f} @ step {best_val_step} -> best_model.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
