"""
Evaluate the empathetic SLM fine-tune against Qwen2.5-3B base on empathetic prompts.

Outputs:
    benchmarks/empathy_eval.json
    benchmarks/empathy_eval.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tiktoken import get_encoding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMPATHY_CKPT = PROJECT_ROOT / "slm-training" / "best_empathy_model.pt"
SLM_CKPT = PROJECT_ROOT / "slm-training" / "best_model.pt"

PROMPTS = [
    "I have been feeling really down lately",
    "I do not think I can keep going",
    "I am so anxious about my job",
    "My partner and I keep fighting",
    "I feel so lonely all the time",
    "I lost my mother last year",
    "I don't know how to cope with stress",
    "I'm scared I'll never find love",
]

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_K = 50
SEED = 1337


def load_slm(path: Path):
    sys_path = str(PROJECT_ROOT / "slm-training")
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from model import GPT, GPTConfig

    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**(state.get("model_config") or {}))
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def generate_slm(model, prompt: str, enc) -> tuple[str, float, int]:
    torch.manual_seed(SEED)
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            x, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K
        )
    dt = time.time() - t0
    new_tokens = out.shape[1] - len(tokens)
    text = enc.decode(out[0].tolist())
    return text, dt, new_tokens


def generate_qwen(tokenizer, model, prompt: str) -> tuple[str, float, int]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )
    dt = time.time() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract just the assistant reply
    reply = decoded.split("assistant\n")[-1].strip()
    return reply, dt, new_tokens


def load_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model.eval()
    return tokenizer, model


def main() -> None:
    enc = get_encoding("gpt2")
    results = []

    # Load SLM (prefer empathy checkpoint, fall back to legacy)
    if EMPATHY_CKPT.exists():
        print(f"Loading empathy checkpoint: {EMPATHY_CKPT}")
        slm, slm_cfg = load_slm(EMPATHY_CKPT)
        slm_label = (
            f"SLM empathy ({slm_cfg.n_layer}L/{slm_cfg.n_head}H/{slm_cfg.n_embd}D)"
        )
    elif SLM_CKPT.exists():
        print(f"Loading legacy checkpoint: {SLM_CKPT}")
        slm, slm_cfg = load_slm(SLM_CKPT)
        slm_label = (
            f"SLM legacy ({slm_cfg.n_layer}L/{slm_cfg.n_head}H/{slm_cfg.n_embd}D)"
        )
    else:
        print("No SLM checkpoint found.")
        return

    # Load Qwen once
    print("Loading Qwen2.5-3B-Instruct...")
    qwen_tokenizer, qwen_model = load_qwen()

    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt}")

        slm_text, slm_dt, slm_tokens = generate_slm(slm, prompt, enc)
        print(
            f"  SLM: {slm_tokens} tok in {slm_dt:.2f}s ({slm_tokens / max(slm_dt, 1e-6):.1f} tok/s)"
        )

        qwen_text, qwen_dt, qwen_tokens = generate_qwen(
            qwen_tokenizer, qwen_model, prompt
        )
        print(
            f"  Qwen: {qwen_tokens} tok in {qwen_dt:.2f}s ({qwen_tokens / max(qwen_dt, 1e-6):.1f} tok/s)"
        )

        results.append(
            {
                "prompt": prompt,
                "slm": {
                    "label": slm_label,
                    "text": slm_text[:500],
                    "tokens": slm_tokens,
                    "latency_s": round(slm_dt, 3),
                    "tok_per_s": round(slm_tokens / max(slm_dt, 1e-6), 1),
                },
                "qwen": {
                    "label": "Qwen2.5-3B-Instruct",
                    "text": qwen_text[:500],
                    "tokens": qwen_tokens,
                    "latency_s": round(qwen_dt, 3),
                    "tok_per_s": round(qwen_tokens / max(qwen_dt, 1e-6), 1),
                },
            }
        )

    out_dir = PROJECT_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "empathy_eval.json"
    md_path = out_dir / "empathy_eval.md"

    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Empathetic Evaluation: SLM vs Qwen2.5-3B",
        "",
        f"**SLM:** {slm_label}",
        "**Qwen:** Qwen2.5-3B-Instruct (base model)",
        f"**Prompts:** {len(PROMPTS)} empathetic scenarios",
        f"**Settings:** max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, top_k={TOP_K}, seed={SEED}",
        "",
        "## Per-Prompt Results",
        "",
    ]
    for r in results:
        lines += [
            f'### Prompt: "{r["prompt"]}"',
            "",
            "| Model | Tokens | Latency (s) | Throughput (tok/s) |",
            "|-------|--------|-------------|-------------------|",
            f"| {r['slm']['label']} | {r['slm']['tokens']} | {r['slm']['latency_s']} | {r['slm']['tok_per_s']} |",
            f"| {r['qwen']['label']} | {r['qwen']['tokens']} | {r['qwen']['latency_s']} | {r['qwen']['tok_per_s']} |",
            "",
            f"**SLM:** {r['slm']['text']}",
            "",
            f"**Qwen:** {r['qwen']['text']}",
            "",
        ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
