from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "standard_nlp_eval.json"

STANDARD_BENCHMARKS = [
    "hellaswag",
    "winogrande",
    "arc_challenge",
    "piqa",
    "commonsense_qa",
    "mmlu",
    "humaneval",
    "math",
    "gsm8k",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _try_import_lm_eval() -> tuple[Any, Any]:
    try:
        import lm_eval
        from lm_eval import evaluator

        return lm_eval, evaluator
    except ImportError:
        return None, None


def _try_import_datasets() -> Any:
    try:
        import datasets

        return datasets
    except ImportError:
        return None


def _try_import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None, None


def _try_import_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError:
        return None


def _extract_math_answer(text: str) -> str:
    text = text.strip()
    m = re.search(r"(-?[0-9][0-9\s,.]*)\b", text)
    if m:
        return m.group(1).strip().replace(",", "")
    return text


def _extract_gsm8k_answer(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"(-?[0-9][0-9\s,.]*)\b", text)
    if m:
        return m.group(1).strip().replace(",", "")
    return None


def _load_slm(checkpoint: str, device: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch = _try_import_torch()
    if torch is None:
        return {
            "name": "slm-scratch",
            "load_success": False,
            "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": "torch unavailable",
        }

    sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
    try:
        from inference import load_model

        model, cfg, ck = load_model(checkpoint, device)
    except Exception as exc:
        return {
            "name": "slm-scratch",
            "load_success": False,
            "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": repr(exc),
        }

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    return {
        "name": "slm-scratch",
        "load_success": True,
        "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model_ref": model,
        "enc_ref": enc,
        "device": device,
        "checkpoint": checkpoint,
    }


def _load_qwen(checkpoint: str, device: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    AutoModelForCausalLM, AutoTokenizer = _try_import_transformers()
    if AutoModelForCausalLM is None:
        return {
            "name": "qwen-base",
            "load_success": False,
            "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": "transformers unavailable",
        }

    torch = _try_import_torch()
    if torch is None:
        return {
            "name": "qwen-base",
            "load_success": False,
            "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": "torch unavailable",
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            trust_remote_code=False,
        )
        model = model.to(device)
        model.eval()
    except Exception as exc:
        return {
            "name": "qwen-base",
            "load_success": False,
            "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": repr(exc),
        }

    return {
        "name": "qwen-base",
        "load_success": True,
        "load_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model_ref": model,
        "tokenizer_ref": tokenizer,
        "device": device,
        "checkpoint": checkpoint,
    }


def _run_lm_eval(
    model_name: str,
    tasks: list[str],
    num_fewshot: int,
    batch_size: int,
    device: str,
    limit: int | None = None,
) -> dict[str, Any]:
    _, evaluator = _try_import_lm_eval()
    if evaluator is None:
        return {
            "success": False,
            "error": "lm_eval not installed. Install with: pip install lm-eval",
            "results": {},
        }

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},device={device}",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            log_samples=False,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": repr(exc),
            "results": {},
        }

    parsed: dict[str, Any] = {}
    for task in tasks:
        task_results = results.get("results", {}).get(task, {})
        metric_key = None
        for key in task_results:
            if "accuracy" in key.lower() or "acc" in key.lower():
                metric_key = key
                break
        if metric_key is None and task_results:
            metric_key = next(iter(task_results))
        parsed[task] = {
            "metric": metric_key,
            "value": task_results.get(metric_key) if metric_key else None,
            "stderr": task_results.get(f"{metric_key}_stderr") if metric_key else None,
        }

    return {
        "success": True,
        "results": parsed,
        "raw_results": results.get("results", {}),
    }


def _format_example(task: str, ex: dict) -> tuple[str, list[str], int]:
    if task == "hellaswag":
        ctx = ex["ctx"]
        choices = ex["endings"]
        label_idx = int(ex["label"])
    elif task == "winogrande":
        ctx = ex["sentence"]
        choices = [ex["option1"], ex["option2"]]
        label_idx = int(ex["answer"]) - 1
    elif task == "arc_challenge":
        ctx = ex["question"]
        choices = ex["choices"]["text"]
        label_idx = ex["choices"]["label"].index(ex["answerKey"])
    elif task == "piqa":
        ctx = ex["goal"]
        choices = [ex["sol1"], ex["sol2"]]
        label_idx = int(ex["label"])
    elif task == "commonsense_qa":
        ctx = ex["question"]
        choices = ex["choices"]["text"]
        label_idx = ex["choices"]["label"].index(ex["answerKey"])
    elif task == "mmlu":
        ctx = ex["question"]
        choices = ex["choices"]
        label_idx = int(ex["answer"])
    else:
        raise ValueError(f"Unknown task: {task}")
    return ctx, choices, label_idx


def _get_max_ctx_len(model: Any) -> int:
    if hasattr(model, "config"):
        if hasattr(model.config, "block_size"):
            return model.config.block_size
        if hasattr(model.config, "max_position_embeddings"):
            return model.config.max_position_embeddings
    return 256


def _run_fallback_eval(
    tier: dict[str, Any],
    tasks: list[str],
    num_fewshot: int,
    batch_size: int,
    limit: int | None = None,
) -> dict[str, Any]:
    del batch_size, num_fewshot
    datasets = _try_import_datasets()
    torch = _try_import_torch()
    if datasets is None:
        return {
            "success": False,
            "error": "datasets library not installed. Install with: pip install datasets",
            "results": {},
        }
    if torch is None:
        return {
            "success": False,
            "error": "torch not installed",
            "results": {},
        }
    if not tier.get("load_success"):
        return {
            "success": False,
            "error": f"model load failed: {tier.get('error')}",
            "results": {},
        }

    model = tier["model_ref"]
    tokenizer_or_enc = tier.get("tokenizer_ref") or tier.get("enc_ref")
    assert tokenizer_or_enc is not None
    device = tier.get("device", "cpu")
    max_len = _get_max_ctx_len(model)

    MC_TASKS = {
        "hellaswag": ("hellaswag", "default"),
        "winogrande": ("winogrande", "winogrande_xl"),
        "arc_challenge": ("ai2_arc", "ARC-Challenge"),
        "piqa": ("piqa", "plain_text"),
        "commonsense_qa": ("commonsense_qa", "default"),
        "mmlu": ("mmlu", "all"),
    }
    GEN_TASKS = {"humaneval", "math", "gsm8k"}

    results: dict[str, Any] = {}
    gen_tasks: list[str] = []

    for task in tasks:
        if task in GEN_TASKS:
            gen_tasks.append(task)
            continue
        if task not in MC_TASKS:
            results[task] = {"success": False, "error": f"unknown task: {task}"}
            continue

        ds_name, config = MC_TASKS[task]
        t0 = time.perf_counter()
        try:
            ds = datasets.load_dataset(ds_name, config, split="validation")
        except Exception:
            try:
                ds = datasets.load_dataset(ds_name, config, split="test")
            except Exception as exc:
                results[task] = {
                    "success": False,
                    "error": f"dataset load failed: {repr(exc)}",
                }
                continue
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        correct = 0
        total = 0
        errors = 0

        for ex in ds:
            total += 1
            try:
                ctx, choices, label_idx = _format_example(task, ex)
                scores: list[float] = []
                for choice in choices:
                    text = f"{ctx} {choice}"
                    if hasattr(tokenizer_or_enc, "encode_ordinary"):
                        ids = tokenizer_or_enc.encode_ordinary(text)
                    else:
                        ids = tokenizer_or_enc.encode(text, add_special_tokens=False)
                    ids = ids[:max_len]
                    if len(ids) < 2:
                        scores.append(float("inf"))
                        continue
                    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
                    _, loss = model(x, y)
                    scores.append(loss.item() if loss is not None else float("inf"))
                pred = min(range(len(scores)), key=lambda i: scores[i])
                if pred == label_idx:
                    correct += 1
            except Exception:
                errors += 1

        dt = time.perf_counter() - t0
        results[task] = {
            "success": True,
            "accuracy": round(correct / max(total, 1), 4),
            "correct": correct,
            "total": total,
            "errors": errors,
            "time_s": round(dt, 2),
        }

    for task in gen_tasks:
        results[task] = _run_generation_eval(tier, task, limit)

    return {"success": True, "results": results}


def _run_generation_eval(
    tier: dict[str, Any],
    task: str,
    limit: int | None = None,
) -> dict[str, Any]:
    datasets = _try_import_datasets()
    torch = _try_import_torch()
    if datasets is None:
        return {
            "success": False,
            "error": "datasets library not installed. Install with: pip install datasets",
        }
    if torch is None:
        return {
            "success": False,
            "error": "torch not installed",
        }
    if not tier.get("load_success"):
        return {
            "success": False,
            "error": f"model load failed: {tier.get('error')}",
        }

    model = tier["model_ref"]
    tokenizer_or_enc = tier.get("tokenizer_ref") or tier.get("enc_ref")
    assert tokenizer_or_enc is not None
    device = tier.get("device", "cpu")
    max_len = _get_max_ctx_len(model)

    TASK_CONFIG = {
        "humaneval": ("openai_humaneval", "default"),
        "math": ("competition_math", "Algebra"),
        "gsm8k": ("gsm8k", "main"),
    }
    if task not in TASK_CONFIG:
        return {"success": False, "error": f"unknown generation task: {task}"}

    ds_name, config = TASK_CONFIG[task]
    t0 = time.perf_counter()
    try:
        ds = datasets.load_dataset(ds_name, config, split="test")
    except Exception:
        try:
            ds = datasets.load_dataset(ds_name, config, split="validation")
        except Exception as exc:
            return {"success": False, "error": f"dataset load failed: {repr(exc)}"}
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    errors = 0
    generations: list[str] = []

    for ex in ds:
        total += 1
        try:
            gold_answer = ""
            if task == "humaneval":
                prompt_text = ex["prompt"]
                gold = ex["canonical_solution"].strip()
            elif task == "math":
                prompt_text = ex["problem"]
                gold = ex["solution"].strip()
                gold_answer = _extract_math_answer(gold)
            else:  # gsm8k
                prompt_text = ex["question"]
                gold = ex["answer"].strip()
                gold_answer = _extract_gsm8k_answer(gold)

            if hasattr(tokenizer_or_enc, "encode"):
                ids = tokenizer_or_enc.encode(prompt_text, add_special_tokens=False)
            else:
                ids = tokenizer_or_enc.encode_ordinary(prompt_text)
            ids = ids[:max_len]
            if len(ids) < 1:
                errors += 1
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(
                    x,
                    max_new_tokens=128,
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
            completion = tokenizer_or_enc.decode(out[0].tolist()[len(ids) :])
            generations.append(completion)

            if task == "humaneval":
                pred = completion.strip()
                if pred == gold:
                    correct += 1
            elif task == "math":
                pred_answer = _extract_math_answer(completion)
                if pred_answer and pred_answer.lower() == gold_answer.lower():
                    correct += 1
            else:  # gsm8k
                pred_answer = _extract_gsm8k_answer(completion)
                if pred_answer and pred_answer == gold_answer:
                    correct += 1
        except Exception:
            errors += 1

    dt = time.perf_counter() - t0
    result: dict[str, Any] = {
        "success": True,
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "errors": errors,
        "time_s": round(dt, 2),
        "mode": "generation",
    }
    if generations:
        result["sample"] = generations[0]
    return result


def evaluate(
    model_name: str,
    checkpoint: str,
    tasks: list[str],
    num_fewshot: int,
    batch_size: int,
    device: str,
    limit: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "config": {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "device": device,
            "limit": limit,
        },
        "tiers": {},
        "benchmarks": {},
        "status": "ok",
    }

    if model_name in ("slm-scratch", "slm"):
        tier = _load_slm(checkpoint, device)
    elif model_name in ("qwen", "qwen-base"):
        tier = _load_qwen(checkpoint, device)
    else:
        tier = {
            "name": model_name,
            "load_success": True,
            "model_ref": model_name,
            "device": device,
        }

    report["tiers"][tier["name"]] = {
        "load_success": tier.get("load_success"),
        "load_latency_ms": tier.get("load_latency_ms"),
        "error": tier.get("error"),
        "device": tier.get("device"),
    }

    if not tier.get("load_success") and model_name not in (
        "slm-scratch",
        "slm",
        "qwen",
        "qwen-base",
    ):
        eval_result = _run_lm_eval(
            model_name, tasks, num_fewshot, batch_size, device, limit
        )
    elif tier.get("load_success"):
        eval_result = _run_fallback_eval(tier, tasks, num_fewshot, batch_size, limit)
    else:
        eval_result = {
            "success": False,
            "error": f"model load failed: {tier.get('error')}",
            "results": {
                t: {"success": False, "error": "model load failed"} for t in tasks
            },
        }

    report["benchmarks"] = eval_result.get("results", {})
    report["status"] = "ok" if eval_result.get("success") else "error"
    if not eval_result.get("success"):
        report["error"] = eval_result.get("error")
        for task in tasks:
            if task not in report["benchmarks"]:
                report["benchmarks"][task] = {
                    "success": False,
                    "error": eval_result.get("error"),
                }

    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate SLM on standard NLP benchmarks (lm-eval-harness style)"
    )
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=STANDARD_BENCHMARKS,
        choices=STANDARD_BENCHMARKS,
        help="Benchmarks to run",
    )
    ap.add_argument(
        "--few-shot", type=int, default=0, help="Number of few-shot examples"
    )
    ap.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    ap.add_argument("--model", default="slm-scratch", help="Model name or checkpoint")
    ap.add_argument("--checkpoint", default="best_model.pt", help="Path to checkpoint")
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    ap.add_argument("--output", default=str(DEFAULT_OUT), help="Output JSON path")
    ap.add_argument("--limit", type=int, default=None, help="Limit examples per task")
    args = ap.parse_args()

    print(f"Evaluating {args.model} on {args.benchmarks}...")
    print(f"  few-shot: {args.few_shot}")
    print(f"  batch-size: {args.batch_size}")
    print(f"  device: {args.device}")
    print()

    report = evaluate(
        model_name=args.model,
        checkpoint=args.checkpoint,
        tasks=args.benchmarks,
        num_fewshot=args.few_shot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")

    print("\n===")
    print("Benchmark summary")
    print("===\n")
    print(f"Generated: {report.get('generated_at')}")
    print(f"Status: {report.get('status')}")
    if report.get("error"):
        print(f"Error: {report['error']}")

    for task, bench in report.get("benchmarks", {}).items():
        if bench.get("success"):
            mode = bench.get("mode", "")
            if mode == "generation":
                sample = bench.get("sample")
                sample_str = f" sample={sample[:80]}..." if sample else ""
                print(
                    f"  {task}: acc={bench.get('accuracy')} "
                    f"({bench.get('correct')}/{bench.get('total')}) "
                    f"[generation]{sample_str}"
                )
            else:
                print(
                    f"  {task}: acc={bench.get('accuracy')} "
                    f"({bench.get('correct')}/{bench.get('total')})"
                )
        else:
            print(f"  {task}: FAILED - {bench.get('error')}")

    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
