from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "benchmarks" / "results_model_families.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def clean_dialogue_artifacts(text: str) -> str:
    """Remove dialogue role prefixes like 'User:' and 'Assistant:' from generated text."""
    import re

    text = re.sub(r"(?:<\|endoftext\|>|^)\s*User:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"(?:<\|endoftext\|>|^)\s*Assistant:\s*", "", text, flags=re.MULTILINE
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tier 1: from-scratch SLM
# ---------------------------------------------------------------------------


def load_slm_tier(checkpoint: str = "best_model.pt") -> dict[str, Any]:
    t0 = now_ms()
    try:
        import inference as slm_inference
    except ImportError as exc:
        return {
            "name": "slm-scratch",
            "checkpoint": checkpoint,
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": f"import path missing: {exc!r}",
        }
    try:
        import torch
    except ImportError as exc:
        return {
            "name": "slm-scratch",
            "checkpoint": checkpoint,
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": f"torch unavailable: {exc!r}",
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(time.time()) % (2**31)
    torch.manual_seed(seed)
    try:
        model, cfg, ck = slm_inference.load_model(checkpoint, device)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "slm-scratch",
            "checkpoint": checkpoint,
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": repr(exc),
        }

    peak_mem_bytes = 0
    if torch.cuda.is_available():
        try:
            peak_mem_bytes = torch.cuda.max_memory_allocated()
        except Exception:
            peak_mem_bytes = 0

    return {
        "name": "slm-scratch",
        "checkpoint": checkpoint,
        "load_success": True,
        "load_latency_ms": round(now_ms() - t0, 1),
        "model_config": ck.get("model_config"),
        "step": ck.get("step"),
        "best_val_loss": ck.get("best_val_loss"),
        "device": device,
        "params_estimate": "757M",
        "tokenizer": "tiktoken gpt2",
        "peak_device_memory_bytes": peak_mem_bytes,
        "seed_used": seed,
        "model_ref": model,
    }


def bench_slm_decode(
    tier: dict[str, Any],
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    if not tier.get("load_success"):
        return {
            "name": tier["name"],
            "decode_success": False,
            "error": tier.get("error"),
        }

    sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
    import tiktoken
    import torch

    model = tier.get("model_ref")
    device = tier["device"]
    enc = tiktoken.get_encoding("gpt2")

    total_tokens = 0
    total_time_s = 0.0
    samples: list[dict[str, Any]] = []

    for prompt in prompts:
        ids = enc.encode_ordinary(prompt)
        if not ids:
            ids = [enc.eot_token]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

        t0 = time.perf_counter()
        try:
            out = model.generate(
                x,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            samples.append({"prompt": prompt, "error": repr(exc)})
            continue
        dt = time.perf_counter() - t0

        text = enc.decode(out[0].tolist())
        completion = text[len(prompt) :] if text.startswith(prompt) else text
        completion = clean_dialogue_artifacts(completion)
        n_new = len(out[0].tolist()) - len(ids)
        total_tokens += max(n_new, 0)
        total_time_s += dt
        samples.append(
            {
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "completion_tokens": max(n_new, 0),
                "completion": completion,
                "latency_s": round(dt, 4),
                "tokens_per_second": round(max(n_new, 0) / max(dt, 1e-6), 1),
            }
        )

    mean_tok_per_s = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    return {
        "name": tier["name"],
        "decode_success": True,
        "prompts": len(prompts),
        "total_tokens_generated": total_tokens,
        "total_decode_time_s": round(total_time_s, 4),
        "mean_tokens_per_second": round(mean_tok_per_s, 1),
        "mean_latency_per_prompt_s": round(total_time_s / max(len(prompts), 1), 4),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Tier 2: Qwen2.5-0.5B base
# ---------------------------------------------------------------------------

QWEN_DIR = PROJECT_ROOT / "slm-ft" / "merged_fp16"


def load_qwen_tier() -> dict[str, Any]:
    t0 = now_ms()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        return {
            "name": "qwen-qlora",
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": f"transformers import failed: {exc}",
            "samples": [],
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else "cpu",
            trust_remote_code=False,
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "qwen-qlora",
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": repr(exc),
            "samples": [],
        }

    peak_mem_bytes = 0
    if device == "cuda":
        try:
            peak_mem_bytes = torch.cuda.max_memory_allocated()
        except Exception:
            peak_mem_bytes = 0

    return {
        "name": "qwen-qlora",
        "load_success": True,
        "load_latency_ms": round(now_ms() - t0, 1),
        "checkpoint": model_name,
        "device": device,
        "params_estimate": "0.5B base",
        "tokenizer": "Qwen2.5 tokenizer",
        "peak_device_memory_bytes": peak_mem_bytes,
        "model_ref": model,
        "tokenizer_ref": tokenizer,
        "generation_config": {},
    }


def _read_generation_config() -> dict[str, Any]:
    path = QWEN_DIR / "generation_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def bench_qwen_decode(
    tier: dict[str, Any],
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    if not tier.get("load_success"):
        return {
            "name": tier["name"],
            "decode_success": False,
            "error": tier.get("error"),
            "samples": [],
        }

    model = tier["model_ref"]
    tokenizer = tier["tokenizer_ref"]
    device = tier["device"]

    total_tokens = 0
    total_time_s = 0.0
    samples: list[dict[str, Any]] = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            enc = {k: v.to(model.device) for k, v in enc.items()}

        input_len = int(enc["input_ids"].shape[-1])
        t0 = time.perf_counter()
        try:
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception as exc:  # noqa: BLE001
            samples.append({"prompt": prompt, "error": repr(exc)})
            continue
        dt = time.perf_counter() - t0

        generated = out[0].tolist()
        text = tokenizer.decode(generated[input_len:], skip_special_tokens=True)
        text = clean_dialogue_artifacts(text)
        n_new = len(generated) - input_len
        total_tokens += max(n_new, 0)
        total_time_s += dt
        samples.append(
            {
                "prompt": prompt,
                "prompt_tokens": input_len,
                "completion_tokens": max(n_new, 0),
                "completion": text,
                "latency_s": round(dt, 4),
                "tokens_per_second": round(max(n_new, 0) / max(dt, 1e-6), 1),
            }
        )

    mean_tok_per_s = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    return {
        "name": tier["name"],
        "decode_success": True,
        "prompts": len(prompts),
        "total_tokens_generated": total_tokens,
        "total_decode_time_s": round(total_time_s, 4),
        "mean_tokens_per_second": round(mean_tok_per_s, 1),
        "mean_latency_per_prompt_s": round(total_time_s / max(len(prompts), 1), 4),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------

CODE_COMPLETION_PROMPTS = [
    "def preprocess(text):",
    "return {",
    "async function fetchUserData(",
    "class ChatMessage:",
    "const response = await fetch(",
]

EMPATHETIC_REPLY_PROMPTS = [
    "I have been feeling really down lately",
    "The best way to handle stress is",
    "I do not think I can keep going",
    "Here is how the model works:",
]

COMPARISON_PROMPTS = CODE_COMPLETION_PROMPTS + EMPATHETIC_REPLY_PROMPTS


# ---------------------------------------------------------------------------
# Loader shim: load a local .pt through pickle, then attach it to the tier
# ---------------------------------------------------------------------------


def load_slm_pt_only(checkpoint: str = "best_model.pt") -> dict[str, Any]:
    t0 = now_ms()

    import torch

    ckpt_path = PROJECT_ROOT / "slm-training" / checkpoint
    if not ckpt_path.exists():
        return {
            "name": "slm-scratch",
            "checkpoint": checkpoint,
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": f"checkpoint file missing: {ckpt_path}",
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "slm-scratch",
            "checkpoint": checkpoint,
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": repr(exc),
        }

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig

        cfg = GPTConfig(**(ck.get("model_config") or {}))
        model = GPT(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        if device == "cuda":
            model = model.to("cuda")
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "slm-scratch",
            "checkpoint": checkpoint,
            "load_success": False,
            "load_latency_ms": round(now_ms() - t0, 1),
            "error": repr(exc),
        }

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    peak_mem_bytes = 0
    if device == "cuda":
        try:
            peak_mem_bytes = torch.cuda.max_memory_allocated()
        except Exception:
            peak_mem_bytes = 0

    return {
        "name": "slm-scratch",
        "checkpoint": checkpoint,
        "load_success": True,
        "load_latency_ms": round(now_ms() - t0, 1),
        "model_config": ck.get("model_config"),
        "step": ck.get("step"),
        "best_val_loss": ck.get("best_val_loss"),
        "device": device,
        "params_estimate": "757M",
        "tokenizer": "tiktoken gpt2",
        "peak_device_memory_bytes": peak_mem_bytes,
        "model_ref": model,
        "ck_ref": ck,
        "enc_ref": enc,
    }


def bench_slm_pt_decode(
    tier: dict[str, Any],
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    if not tier.get("load_success"):
        return {
            "name": tier["name"],
            "decode_success": False,
            "error": tier.get("error"),
            "samples": [],
        }

    import torch

    model = tier.get("model_ref")
    if model is None:
        return {
            "name": tier["name"],
            "decode_success": False,
            "error": "no model_ref attached",
            "samples": [],
        }
    device = tier["device"]
    enc = tier.get("enc_ref") or tiktoken_ref()

    total_tokens = 0
    total_time_s = 0.0
    samples: list[dict[str, Any]] = []

    for prompt in prompts:
        ids = enc.encode_ordinary(prompt)
        if not ids:
            ids = [enc.eot_token]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

        t0 = time.perf_counter()
        try:
            out = model.generate(
                x,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            samples.append({"prompt": prompt, "error": repr(exc)})
            continue
        dt = time.perf_counter() - t0

        text = enc.decode(out[0].tolist())
        completion = text[len(prompt) :] if text.startswith(prompt) else text
        completion = clean_dialogue_artifacts(completion)
        n_new = len(out[0].tolist()) - len(ids)
        total_tokens += max(n_new, 0)
        total_time_s += dt
        samples.append(
            {
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "completion_tokens": max(n_new, 0),
                "completion": completion,
                "latency_s": round(dt, 4),
                "tokens_per_second": round(max(n_new, 0) / max(dt, 1e-6), 1),
            }
        )

    mean_tok_per_s = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    return {
        "name": tier["name"],
        "decode_success": True,
        "prompts": len(prompts),
        "total_tokens_generated": total_tokens,
        "total_decode_time_s": round(total_time_s, 4),
        "mean_tokens_per_second": round(mean_tok_per_s, 1),
        "mean_latency_per_prompt_s": round(total_time_s / max(len(prompts), 1), 4),
        "samples": samples,
    }


def tiktoken_ref():
    import tiktoken as _t

    return _t.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    slm_code_checkpoint: str = "best_model.pt",
    slm_empathetic_checkpoint: str = "final_model.pt",
    temperature: float = 0.7,
    max_tokens: int = 120,
    top_k: int = 50,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "slm_code_checkpoint": slm_code_checkpoint,
            "slm_empathetic_checkpoint": slm_empathetic_checkpoint,
            "code_prompts": CODE_COMPLETION_PROMPTS,
            "empathetic_prompts": EMPATHETIC_REPLY_PROMPTS,
        },
        "tiers": {},
        "benchmarks": {},
        "comparison": {},
    }

    # --- Load tiers ---
    slm_code_tier = load_slm_pt_only(slm_code_checkpoint)
    slm_empathetic_tier = load_slm_pt_only(slm_empathetic_checkpoint)
    qwen_tier = load_qwen_tier()

    report["tiers"] = {
        "slm-scratch-code": slm_code_tier,
        "slm-scratch-empathetic": slm_empathetic_tier,
        "qwen-qlora": qwen_tier,
    }

    # --- Decode benchmarks ---
    report["benchmarks"] = {
        "code_completion": {
            "slm-scratch-code": bench_slm_pt_decode(
                slm_code_tier, CODE_COMPLETION_PROMPTS, max_tokens, temperature, top_k
            ),
            "slm-scratch-empathetic": bench_slm_pt_decode(
                slm_empathetic_tier,
                CODE_COMPLETION_PROMPTS,
                max_tokens,
                temperature,
                top_k,
            ),
            "qwen-qlora": bench_qwen_decode(
                qwen_tier, CODE_COMPLETION_PROMPTS, max_tokens, temperature, top_k
            ),
        },
        "empathetic_reply": {
            "slm-scratch-code": bench_slm_pt_decode(
                slm_code_tier, EMPATHETIC_REPLY_PROMPTS, max_tokens, temperature, top_k
            ),
            "slm-scratch-empathetic": bench_slm_pt_decode(
                slm_empathetic_tier,
                EMPATHETIC_REPLY_PROMPTS,
                max_tokens,
                temperature,
                top_k,
            ),
            "qwen-qlora": bench_qwen_decode(
                qwen_tier, EMPATHETIC_REPLY_PROMPTS, max_tokens, temperature, top_k
            ),
        },
        "comparison_prompts": {
            "slm-scratch-code": bench_slm_pt_decode(
                slm_code_tier, COMPARISON_PROMPTS, max_tokens, temperature, top_k
            ),
            "slm-scratch-empathetic": bench_slm_pt_decode(
                slm_empathetic_tier, COMPARISON_PROMPTS, max_tokens, temperature, top_k
            ),
            "qwen-qlora": bench_qwen_decode(
                qwen_tier, COMPARISON_PROMPTS, max_tokens, temperature, top_k
            ),
        },
    }

    # --- Same-data comparison table ---
    cmp = report["benchmarks"]["comparison_prompts"]
    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(COMPARISON_PROMPTS):
        row = {
            "prompt": prompt,
            "tiers": {
                "slm-scratch-code": _pick_sample(cmp["slm-scratch-code"]["samples"], i),
                "slm-scratch-empathetic": _pick_sample(
                    cmp["slm-scratch-empathetic"]["samples"], i
                ),
                "qwen-qlora": _pick_sample(cmp["qwen-qlora"]["samples"], i),
            },
        }
        rows.append(row)
    report["comparison"]["prompt_level"] = rows

    return report


def _pick_sample(samples: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not samples:
        return {"error": "no samples"}
    try:
        return samples[index]
    except IndexError:
        return {"error": "missing sample at index", "index": index}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark the two model families against each other"
    )
    ap.add_argument("--slm-code-checkpoint", default="best_model.pt")
    ap.add_argument("--slm-empathetic-checkpoint", default="final_model.pt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--json-out", default=str(OUT_PATH))
    args = ap.parse_args()

    print("Benchmarking model families...")
    print(f"  slm code checkpoint: {args.slm_code_checkpoint}")
    print(f"  slm empathetic checkpoint: {args.slm_empathetic_checkpoint}")
    print(f"  temperature: {args.temperature}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  top_k: {args.top_k}")
    print()

    report = run(
        slm_code_checkpoint=args.slm_code_checkpoint,
        slm_empathetic_checkpoint=args.slm_empathetic_checkpoint,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
    )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {out_path}")

    _print_summary(report)
    return 0


def _print_summary(report: dict[str, Any]) -> None:
    config = report.get("config", {})
    print("\n===\nBenchmark summary\n===\n")
    print(f"Generated: {report.get('generated_at')}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}\n")

    for tier_key, tier in report.get("tiers", {}).items():
        print(f"[Tier] {tier_key}")
        if not tier.get("load_success"):
            print(f"  load: FAILED - {tier.get('error')}")
        else:
            mem = tier.get("peak_device_memory_bytes", 0)
            if mem:
                print(f"  load: OK in {tier.get('load_latency_ms')}ms")
                print(f"  device: {tier.get('device')}")
                print(f"  peak memory: {mem / 1e6:.1f} MB")
                print(f"  params: {tier.get('params_estimate')}")
                print(f"  tokenizer: {tier.get('tokenizer')}")
            else:
                print(f"  load: OK in {tier.get('load_latency_ms')}ms")
                print(f"  device: {tier.get('device')}")
                print(f"  params: {tier.get('params_estimate')}")
                print(f"  tokenizer: {tier.get('tokenizer')}")
        print()

    for bench_name, benches in report.get("benchmarks", {}).items():
        print(f"[Benchmark] {bench_name}")
        for tier_key, bench in benches.items():
            if not bench.get("decode_success"):
                print(f"  {tier_key}: DECODE FAILED - {bench.get('error')}")
                continue
            print(
                f"  {tier_key}:"
                f" prompts={bench.get('prompts')}"
                f" total_tokens={bench.get('total_tokens_generated')}"
                f" total_time_s={bench.get('total_decode_time_s')}"
                f" mean_tok/s={bench.get('mean_tokens_per_second')}"
                f" mean_prompt_latency_s={bench.get('mean_latency_per_prompt_s')}"
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
