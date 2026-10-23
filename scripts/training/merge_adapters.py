"""Merge LoRA adapters into base model for easier serving with vLLM.

After SFT training, this creates a merged model you can serve directly.

Usage:
  python scripts/training/merge_adapters.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter_dir checkpoints/emotional-slm-v1 \
    --output_dir checkpoints/emotional-slm-v1-merged \
    --use_8bit false
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", required=True, help="Base model name or path")
    parser.add_argument("--adapter_dir", required=True, help="Directory with LoRA adapters")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged model")
    parser.add_argument("--use_8bit", type=bool, default=False, help="Load base model in 8-bit")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    output_path = Path(args.output_dir)
    
    if not adapter_path.exists():
        raise SystemExit(f"Adapter directory not found: {adapter_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.use_8bit,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging adapters into base model...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(str(output_path))
    
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("✓ Merge complete!")
    logger.info(f"\nTo serve the merged model with vLLM:")
    logger.info(f"  vllm serve {output_path} --port 8001 --gpu-memory-utilization 0.8")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
