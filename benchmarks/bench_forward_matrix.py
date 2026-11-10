"""Forward-pass matrix for README section 9.1.

Runs every configuration named in the section 9.1 table through one harness so
the table's numbers come from a single artifact instead of pasted cells.

Harness: batch 1, sequence length 64, fp32, 3 warmup passes, 20 timed passes,
`model(x)` called directly so the lm_head is included. Defaults to CPU to
match benchmarks/run_full_bench.py; pass --device cuda to repeat on a GPU.

The four ladder rows (Tiny/Small/Medium/Large) are synthetic GPTConfig shapes
with random weights. Forward-pass cost is set by tensor shapes rather than
weight values, so random init times identically to a trained checkpoint, but
they are NOT trained models and are labelled synthetic in the artifact. The
two production rows load the real shipped checkpoints.

A fifth, optional row loads the merged QLoRA artifact written by
``slm-ft/merge_adapter.py`` (``slm-ft/merged_fp16``) so the fine-tuned Qwen is
measured with the same harness instead of only the base checkpoint. It is
skipped automatically when the artifact is missing or still the 346-byte
placeholder stub, and the reason is recorded in ``harness.merged_row_skipped``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from model import GPT, GPTConfig  # noqa: E402

OUT_PATH = PROJECT_ROOT / "benchmarks" / "forward_matrix.json"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_ID = "slm-ft/merged_fp16"
MERGED_STUB_BYTES = 1_000_000

# README section 9.1 ladder rows: synthetic shapes, no trained checkpoint.
LADDER = [
    ("Tiny", 4, 4, 512),
    ("Small", 6, 6, 768),
    ("Medium", 8, 4, 1024),
    ("Large", 12, 6, 1536),
]

# Real shipped checkpoints named in section 9.1, labelled by architecture and
# file. The CLI/API default (`best_clean_model_v2.pt`, 16L / 16H / 1024D) is a
# third checkpoint and is not one of these rows.
PRODUCTION = [
    ("SLM 16L (final_model.pt)", PROJECT_ROOT / "slm-training" / "final_model.pt"),
    (
        "SLM 24L (best_empathy_model_v2.pt)",
        PROJECT_ROOT / "slm-training" / "best_empathy_model_v2.pt",
    ),
]


def unique_param_count(model: torch.nn.Module) -> int:
    """Count each parameter once, ignoring tied (shared) tensors."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def build_synthetic(n_layer: int, n_head: int, n_embd: int) -> GPT:
    cfg = GPTConfig(
        vocab_size=50257,
        block_size=256,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=True,
    )
    return GPT(cfg)


def build_from_checkpoint(path: Path) -> tuple[GPT, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**dict(state["model_config"]))
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    meta = {
        "source": "checkpoint",
        "load_source": str(path.relative_to(PROJECT_ROOT)),
        "proxy_for": None,
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
        "checkpoint_step": state.get("step"),
        "checkpoint_best_val_loss": state.get("best_val_loss"),
        "checkpoint_config": state.get("model_config"),
    }
    return model, meta


def build_qwen() -> tuple[torch.nn.Module, dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, torch_dtype=torch.float32)
    cfg = model.config
    meta = {
        "source": "huggingface",
        "load_source": QWEN_ID,
        "proxy_for": None,
        "model_id": QWEN_ID,
        "config": {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "torch_dtype": str(cfg.torch_dtype),
        },
        "tokenizer_class": type(tokenizer).__name__,
    }
    return model, meta


def merged_is_stub(merged_path: Path) -> bool:
    """True when the merged artifact is missing or still the placeholder stub."""
    weights = merged_path / "model.safetensors"
    return (not weights.exists()) or weights.stat().st_size <= MERGED_STUB_BYTES


def build_merged_qwen(
    merged_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the merged QLoRA artifact as a plain causal LM (no PEFT wrapper)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    cfg = model.config
    adapter_dir = PROJECT_ROOT / "slm-ft" / "adapter_stage2"
    adapter_summary: dict[str, Any] = {"path": "slm-ft/adapter_stage2"}
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        adapter_summary.update(
            {
                "r": raw.get("r"),
                "lora_alpha": raw.get("lora_alpha"),
                "target_modules": raw.get("target_modules"),
                "peft_type": raw.get("peft_type"),
            }
        )
    weights = adapter_dir / "adapter_model.safetensors"
    if weights.exists():
        try:
            from safetensors import safe_open

            total = 0
            with safe_open(str(weights), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    total += int(handle.get_tensor(key).numel())
            adapter_summary["trainable_scalars"] = total
        except Exception:  # noqa: BLE001
            adapter_summary["trainable_scalars"] = None
    meta = {
        "source": "merged_qlora",
        "model_id": str(merged_path),
        "merged_from": "slm-ft/adapter_stage2",
        "adapter": adapter_summary,
        "load_source": MERGED_ID,
        "proxy_for": None,
        "weight_dtype": "float16",
        "config": {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "torch_dtype": str(cfg.torch_dtype),
        },
        "tokenizer_class": type(tokenizer).__name__,
    }
    return model, meta


def time_forward(
    model: torch.nn.Module, seq_len: int, passes: int, warmup: int, device: torch.device
) -> dict:
    x = torch.randint(0, 50257, (1, seq_len), device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(passes):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    ordered = sorted(samples)
    return {
        "seq_len": seq_len,
        "passes": passes,
        "warmup": warmup,
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "p95_s": ordered[int(0.95 * (len(ordered) - 1))],
        "avg_latency_ms": statistics.mean(samples) * 1000.0,
        "throughput_passes_s": 1.0 / statistics.mean(samples),
    }


def release(model: torch.nn.Module, device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=PROJECT_ROOT / "slm-ft" / "merged_fp16",
        help="Merged QLoRA artifact; skipped when it is still the stub.",
    )
    parser.add_argument(
        "--skip-merged",
        action="store_true",
        help="Do not load or benchmark the merged fine-tuned Qwen row.",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.manual_seed(1337)

    rows: list[dict] = []

    for name, n_layer, n_head, n_embd in LADDER:
        model = build_synthetic(n_layer, n_head, n_embd).to(device).eval()
        timing = time_forward(model, args.seq_len, args.passes, args.warmup, device)
        rows.append(
            {
                "name": name,
                "source": "synthetic",
                "load_source": "synthetic",
                "proxy_for": None,
                "config": {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embd},
                "params": unique_param_count(model),
                **timing,
            }
        )
        release(model, device)
        print(f"[done] {name}: {timing['mean_s']:.4f}s/pass", flush=True)

    for name, path in PRODUCTION:
        model, meta = build_from_checkpoint(path)
        model = model.to(device).eval()
        timing = time_forward(model, args.seq_len, args.passes, args.warmup, device)
        rows.append(
            {
                "name": name,
                **meta,
                "config": meta["checkpoint_config"],
                "params": unique_param_count(model),
                **timing,
            }
        )
        release(model, device)
        print(f"[done] {name}: {timing['mean_s']:.4f}s/pass", flush=True)

    if not args.skip_qwen:
        model, meta = build_qwen()
        model = model.to(device).eval()
        timing = time_forward(model, args.seq_len, args.passes, args.warmup, device)
        rows.append(
            {
                "name": "Qwen2.5-0.5B-Instruct",
                **meta,
                "params": unique_param_count(model),
                **timing,
            }
        )
        release(model, device)
        print(f"[done] Qwen2.5-0.5B-Instruct: {timing['mean_s']:.4f}s/pass", flush=True)

    merged_row_skipped = None
    if args.skip_merged:
        merged_row_skipped = "--skip-merged"
    elif not args.merged_path.exists():
        merged_row_skipped = f"missing: {args.merged_path}"
    elif merged_is_stub(args.merged_path):
        merged_row_skipped = (
            f"stub artifact (<= {MERGED_STUB_BYTES} bytes): {args.merged_path}"
        )

    if merged_row_skipped is None:
        model, meta = build_merged_qwen(args.merged_path, device)
        model = model.to(device).eval()
        timing = time_forward(model, args.seq_len, args.passes, args.warmup, device)
        rows.append(
            {
                "name": "Qwen2.5-0.5B-Instruct + QLoRA stage-2 (merged_fp16)",
                **meta,
                "params": unique_param_count(model),
                **timing,
            }
        )
        release(model, device)
        print(
            f"[done] merged Qwen (slm-ft/merged_fp16): {timing['mean_s']:.4f}s/pass",
            flush=True,
        )
    else:
        print(f"[skip] merged Qwen: {merged_row_skipped}", flush=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "harness": {
            "device": args.device,
            "dtype": "float32",
            "batch": 1,
            "seq_len": args.seq_len,
            "passes": args.passes,
            "warmup": args.warmup,
            "includes_lm_head": True,
            "merged_row_skipped": merged_row_skipped,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
