"""Supervised Fine-Tuning (SFT) with QLoRA for Emotional SLM on a single RTX 3090.

This script trains a 7B instruction-tuned model using:
- 4-bit NF4 quantization (QLoRA)
- LoRA adapters (rank 64, minimal overhead)
- Streaming to reduce memory
- Your synthetic supportive conversation data

Tuned for 24GB VRAM (RTX 3090).

Usage:
  python scripts/training/train_sft_qlora.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir checkpoints/emotional-slm-v1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_chat_template(example: dict) -> dict:
    """Format messages for chat template (Mistral/Qwen compatible)."""
    messages = example.get("messages", [])
    
    # Build chat text
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            text += f"[INST] {content} [/INST] "
        elif role == "assistant":
            text += f"{content} </s> "
        elif role == "system":
            text += f"[SYSTEM] {content} [/SYSTEM] "
    
    example["text"] = text.strip()
    return example


def _create_qlora_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for 3090."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _create_lora_config() -> LoraConfig:
    """LoRA adapter config tuned for 3090."""
    return LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT with QLoRA on a single 3090")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model")
    parser.add_argument("--train_file", required=True, help="Training JSONL file")
    parser.add_argument("--val_file", required=False, help="Validation JSONL file (optional)")
    parser.add_argument("--output_dir", default="checkpoints/emotional-slm-v1", help="Output directory")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval every N steps (if val_file provided)")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--use_flash_attention", action="store_true", default=True, help="Use Flash Attention 2")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--packing", action="store_true", default=False, help="Pack examples (for very large datasets)")
    args = parser.parse_args()

    # Paths
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(f"Train file not found: {train_path}")
    
    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"Train data: {train_path}")
    logger.info(f"Output: {output_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    train_dataset = train_dataset.map(_format_chat_template, remove_columns=["messages"])
    
    eval_dataset = None
    if args.val_file:
        logger.info("Loading validation dataset...")
        val_path = Path(args.val_file)
        if val_path.exists():
            eval_dataset = load_dataset("json", data_files=str(val_path), split="train")
            eval_dataset = eval_dataset.map(_format_chat_template, remove_columns=["messages"])
        else:
            logger.warning(f"Validation file not found: {val_path}")

    logger.info(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation examples: {len(eval_dataset)}")

    # Load model with QLoRA
    logger.info("Loading model with 4-bit quantization...")
    bnb_config = _create_qlora_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager",
    )

    # Prepare for kbit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    lora_config = _create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        logging_first_step=True,
        seed=42,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        dataset_text_field="text",
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving adapter to {output_path}...")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("✓ Training complete!")
    logger.info(f"Adapters saved to: {output_path}")
    logger.info(f"To use the model:")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  model = PeftModel.from_pretrained(base_model, '{output_path}')")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
