"""Token-generation benchmark for README sections 9.2 and 9.4.

`run_full_bench.py` measures decode throughput but writes artifacts that do not
record which device was used, which checkpoint was loaded, or whether the
generation loop uses a KV cache. That makes its numbers impossible to audit, so
this script re-runs the same comparison and records enough metadata to
reproduce it.

Decode throughput is dominated by one implementation detail: a model without a
KV cache recomputes the whole sequence for every generated token, so its cost
grows with sequence length instead of staying flat. The artifact therefore
records `uses_kv_cache` for each row alongside the timings.

Usage:
    python benchmarks/bench_generation_matrix.py [--device cpu|cuda] [--out PATH]
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from guardrails import clean_dialogue_artifacts  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tiktoken import get_encoding  # noqa: E402

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Merged QLoRA artifact written by slm-ft/merge_adapter.py from
# slm-ft/adapter_stage2. The guard below exists because the directory used to
# hold a 346-byte JSON stub; a stub must never be benchmarked and reported as a
# fine-tuned model, so the check stays in place for a fresh checkout.
MERGED_ID = "slm-ft/merged_fp16"
MERGED_STUB_BYTES = 1_000_000

# Two real shipped checkpoints, named by architecture and file so a row label
# cannot be mistaken for the CLI/API default. `inference.py` points at
# `best_clean_model_v2.pt` (16L / 16H / 1024D, step 8000) -- neither of these --
# so this matrix measures both the 24L tier and the 16L tier it is compared to.
SLM_ROWS = {
    "SLM 24L (best_empathy_model_v2.pt)": "slm-training/best_empathy_model_v2.pt",
    "SLM 16L (final_model.pt)": "slm-training/final_model.pt",
}

# Same prompt set as run_full_bench.py so the two artifacts are comparable.
PROMPTS = [
    "def preprocess(text):",
    "I have been feeling really down lately",
    "class ChatMessage:",
    "async function fetchUserData(",
    "SELECT * FROM users WHERE",
]


def uses_kv_cache(model: torch.nn.Module) -> bool:
    """True when generate() can reuse past key/value states.

    HF `GenerationMixin.generate` takes `**kwargs`, so `use_cache` never appears
    in its signature; a hand-rolled loop that calls `self(idx)` on the full
    sequence does not cache anything. The class check is the reliable signal.
    """
    params = inspect.signature(model.generate).parameters
    if "use_cache" in params or "past_key_values" in params:
        return True
    try:
        from transformers.generation import GenerationMixin

        if isinstance(model, GenerationMixin):
            return True
    except Exception:
        pass
    doc = (model.generate.__doc__ or "").lower()
    return "kv cache" in doc or "past key" in doc


def rss_mb() -> float | None:
    """Resident set size of this process in MB, or None when psutil is absent.

    psutil ships as a transitive dependency of accelerate, so it is normally
    present; the benchmark still has to run on machines where it is not.
    """
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def load_slm(path: Path, device: str) -> tuple[GPT, object, dict]:
    ckpt = torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**(ckpt.get("model_config") or {}))
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, get_encoding("gpt2"), ckpt


def load_qwen(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    tok = AutoTokenizer.from_pretrained(QWEN_ID)
    return model, tok


def bench_slm(
    model: GPT,
    enc,
    prompts: list[str],
    *,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    results = []
    total_toks = 0
    total_time = 0.0
    for p in prompts:
        ids = enc.encode(p) or [enc.eot_token]
        x = torch.tensor(
            [ids], dtype=torch.long, device=next(model.parameters()).device
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=enc.eot_token,
            )
        dt = time.perf_counter() - t0
        text = enc.decode(out[0].tolist())
        comp = text[len(p) :] if text.startswith(p) else text
        n_new = max(int(out.numel()) - len(ids), 0)
        total_toks += n_new
        total_time += dt
        results.append(
            {
                "prompt": p,
                "prompt_tokens": len(ids),
                "completion_tokens": n_new,
                "latency_s": round(dt, 4),
                "tokens_per_second": round(n_new / max(dt, 1e-6), 2),
                "completion": clean_dialogue_artifacts(comp)[:160],
            }
        )
    return {
        "total_tokens": total_toks,
        "total_time_s": round(total_time, 3),
        "mean_tokens_per_second": round(total_toks / max(total_time, 1e-6), 2),
        "mean_prompt_latency_s": round(total_time / len(prompts), 3),
        "results": results,
    }


def bench_qwen(
    model,
    tok,
    prompts: list[str],
    *,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    results = []
    total_toks = 0
    total_time = 0.0
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        plen = int(enc["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        dt = time.perf_counter() - t0
        text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        prompt_text = tok.decode(enc["input_ids"][0].tolist(), skip_special_tokens=True)
        comp = text[len(prompt_text) :] if text.startswith(prompt_text) else text
        n_new = max(int(out.shape[-1]) - plen, 0)
        total_toks += n_new
        total_time += dt
        results.append(
            {
                "prompt": p,
                "prompt_tokens": plen,
                "completion_tokens": n_new,
                "latency_s": round(dt, 4),
                "tokens_per_second": round(n_new / max(dt, 1e-6), 2),
                "completion": clean_dialogue_artifacts(comp)[:160],
            }
        )
    return {
        "total_tokens": total_toks,
        "total_time_s": round(total_time, 3),
        "mean_tokens_per_second": round(total_toks / max(total_time, 1e-6), 2),
        "mean_prompt_latency_s": round(total_time / len(prompts), 3),
        "results": results,
    }


def merged_is_stub(merged_path: Path) -> bool:
    """True when the merged artifact is missing or still the placeholder stub."""
    weights = merged_path / "model.safetensors"
    return (not weights.exists()) or weights.stat().st_size <= MERGED_STUB_BYTES


def merged_adapter_summary() -> dict[str, Any]:
    """Rank/alpha/targets and on-disk scalar count of the stage-2 adapter."""
    adapter_dir = PROJECT_ROOT / "slm-ft" / "adapter_stage2"
    summary: dict[str, Any] = {"path": "slm-ft/adapter_stage2"}
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        summary.update(
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
            summary["trainable_scalars"] = total
        except Exception:  # noqa: BLE001
            summary["trainable_scalars"] = None
    return summary


def load_merged_qwen(merged_path: Path, device: str) -> tuple[Any, Any]:
    """Load the merged QLoRA artifact as a plain causal LM (no PEFT wrapper)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        dtype=load_dtype,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rows", type=int, default=5, help="Prompts per row (max 5).")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "generation_matrix.json",
    )
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
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    prompts = PROMPTS[: args.rows]
    rows = []

    for name, rel in SLM_ROWS.items():
        t_load = time.perf_counter()
        model, enc, ckpt = load_slm(Path(rel), args.device)
        load_latency = round(time.perf_counter() - t_load, 3)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[slm] {name}: loaded in {load_latency}s, generating...",
            flush=True,
        )
        stats = bench_slm(
            model,
            enc,
            prompts,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        rows.append(
            {
                "name": name,
                "kind": "slm-scratch",
                "checkpoint": rel,
                "checkpoint_step": ckpt.get("step"),
                "checkpoint_best_val_loss": ckpt.get("best_val_loss"),
                "model_config": ckpt.get("model_config"),
                "params": n_params,
                "weight_bytes": n_params * 4,
                "load_latency_s": load_latency,
                "rss_after_load_mb": rss_mb(),
                "uses_kv_cache": uses_kv_cache(model),
                **stats,
            }
        )
        print(
            f"      {stats['mean_tokens_per_second']} tok/s over "
            f"{stats['total_tokens']} tokens ({stats['total_time_s']}s), "
            f"kv_cache={rows[-1]['uses_kv_cache']}",
            flush=True,
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"[qwen] loading {QWEN_ID}...", flush=True)
    t_load = time.perf_counter()
    qwen_model, qwen_tok = load_qwen(args.device)
    qwen_load_latency = round(time.perf_counter() - t_load, 3)
    qwen_params = sum(p.numel() for p in qwen_model.parameters())
    qwen_cfg = qwen_model.config.to_dict()
    print("[qwen] generating...", flush=True)
    qwen_stats = bench_qwen(
        qwen_model,
        qwen_tok,
        prompts,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    rows.append(
        {
            "name": "Qwen2.5-0.5B-Instruct",
            "kind": "qwen",
            "model_id": QWEN_ID,
            "model_config": {
                k: qwen_cfg.get(k)
                for k in (
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "hidden_size",
                    "intermediate_size",
                    "vocab_size",
                    "tie_word_embeddings",
                )
            },
            "params": qwen_params,
            "weight_bytes": qwen_params * 4,
            "load_latency_s": qwen_load_latency,
            "rss_after_load_mb": rss_mb(),
            "uses_kv_cache": uses_kv_cache(qwen_model),
            **qwen_stats,
        }
    )
    print(
        f"      {qwen_stats['mean_tokens_per_second']} tok/s over "
        f"{qwen_stats['total_tokens']} tokens ({qwen_stats['total_time_s']}s), "
        f"kv_cache={rows[-1]['uses_kv_cache']}",
        flush=True,
    )
    # Free the base model before loading the merged one: rss_after_load_mb is
    # read from the live process, so a resident base model gets counted twice
    # in the merged row (base weights + merged weights on top).
    del qwen_model, qwen_tok
    gc.collect()

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
        print(f"[qwen-ft] loading merged {args.merged_path}...", flush=True)
        t_load = time.perf_counter()
        merged_model, merged_tok = load_merged_qwen(args.merged_path, args.device)
        merged_load_latency = round(time.perf_counter() - t_load, 3)
        merged_params = sum(p.numel() for p in merged_model.parameters())
        merged_cfg = merged_model.config.to_dict()
        print("[qwen-ft] generating...", flush=True)
        merged_stats = bench_qwen(
            merged_model,
            merged_tok,
            prompts,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        rows.append(
            {
                "name": "Qwen2.5-0.5B-Instruct + QLoRA stage-2 (merged_fp16)",
                "kind": "qwen-ft",
                "model_id": str(args.merged_path),
                "load_source": MERGED_ID,
                "proxy_for": None,
                "merged_from": "slm-ft/adapter_stage2",
                "adapter": merged_adapter_summary(),
                "model_config": {
                    k: merged_cfg.get(k)
                    for k in (
                        "num_hidden_layers",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "hidden_size",
                        "intermediate_size",
                        "vocab_size",
                        "tie_word_embeddings",
                        "torch_dtype",
                    )
                },
                "params": merged_params,
                # merge_adapter.py writes the merged weights as fp16.
                "weight_bytes": merged_params * 2,
                "weight_dtype": "float16",
                "load_dtype": "float16" if args.device == "cuda" else "float32",
                "load_latency_s": merged_load_latency,
                "rss_after_load_mb": rss_mb(),
                "uses_kv_cache": uses_kv_cache(merged_model),
                **merged_stats,
            }
        )
        print(
            f"      {merged_stats['mean_tokens_per_second']} tok/s over "
            f"{merged_stats['total_tokens']} tokens "
            f"({merged_stats['total_time_s']}s), "
            f"kv_cache={rows[-1]['uses_kv_cache']}",
            flush=True,
        )
        del merged_model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"[qwen-ft] skipped: {merged_row_skipped}", flush=True)

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "benchmarks/bench_generation_matrix.py",
        "harness": {
            "device": args.device,
            "dtype": "float32",
            "max_new_tokens": args.max_new,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "cleaned": True,
            "prompts": prompts,
            "merged_row_skipped": merged_row_skipped,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cpu_count": __import__("os").cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
