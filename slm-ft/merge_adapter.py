"""Merge a trained QLoRA adapter into fp16 full weights for deployment.

Usage:
    .venv/Scripts/python merge_adapter.py --base Qwen/Qwen2.5-0.5B-Instruct \
        --adapter adapter --out merged_fp16

Peft loads the adapter on a CPU float16 copy of the base (fits in RAM),
then the merged model is saved; load it later in bf16 on the GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="adapter")
    ap.add_argument("--out", default="merged_fp16")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    peft_model = PeftModel.from_pretrained(model, args.adapter)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("merged fp16 model written to", out)


if __name__ == "__main__":
    main()
