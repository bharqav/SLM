#!/usr/bin/env python3
"""Evaluate the from-scratch SLM on a held-out test split.

Usage:
    python evaluate_test_split.py

Outputs:
    benchmarks/test_split_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slm-training"))

from model import GPT, GPTConfig  # noqa: E402


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_data(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    data = torch.tensor(list(path.read_bytes()), dtype=torch.long)
    return data


def compute_perplexity(
    model: GPT, data: torch.Tensor, ctx_len: int = 256, stride: int = 256
) -> float:
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
        data = data.to(device)

    nlls = []
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, stride):
            chunk = data[i : i + ctx_len + 1]
            if chunk.size(0) < 2:
                continue
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            if device == "cuda":
                x = x.to(device)
                y = y.to(device)
            logits, loss = model(x, y)
            nlls.append(loss.item())

    if not nlls:
        return float("nan")
    avg_nll = sum(nlls) / len(nlls)
    return float(torch.exp(torch.tensor(avg_nll)).item())


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate SLM on held-out test split")
    ap.add_argument(
        "--checkpoint", default=str(ROOT / "slm-training" / "best_model.pt")
    )
    ap.add_argument(
        "--test-data",
        default=str(ROOT / "slm-training" / "empathy_val.bin"),
        help="Use empathy_val.bin as test proxy",
    )
    ap.add_argument("--ctx-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    args = ap.parse_args()

    print("[evaluate_test_split] Loading model...")
    t0 = time.perf_counter()
    state = load_checkpoint(Path(args.checkpoint))
    cfg = GPTConfig(**(state.get("model_config") or {}))
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    dt = time.perf_counter() - t0
    print(
        f"  loaded in {dt:.2f}s, step={state.get('step')}, best_val_loss={state.get('best_val_loss')}"
    )

    print(f"[evaluate_test_split] Loading test data: {args.test_data}")
    data = load_data(Path(args.test_data))
    print(f"  {data.size(0):,} tokens")

    print("[evaluate_test_split] Computing perplexity...")
    t0 = time.perf_counter()
    ppl = compute_perplexity(model, data, ctx_len=args.ctx_len, stride=args.stride)
    dt = time.perf_counter() - t0
    print(f"  perplexity: {ppl:.4f}  (wall time: {dt:.2f}s)")

    result = {
        "checkpoint": str(args.checkpoint),
        "test_data": str(args.test_data),
        "test_tokens": int(data.size(0)),
        "ctx_len": args.ctx_len,
        "stride": args.stride,
        "perplexity": round(ppl, 4),
        "eval_time_s": round(dt, 2),
        "step": state.get("step"),
        "best_val_loss": state.get("best_val_loss"),
    }

    out_path = ROOT / "benchmarks" / "test_split_eval.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
