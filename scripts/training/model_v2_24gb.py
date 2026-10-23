"""Build practical training commands for a 24GB VRAM setup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPlan24GB:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_output_dir: str = "artifacts/sft-qwen2.5-7b-lora"
    dpo_output_dir: str = "artifacts/dpo-qwen2.5-7b-lora"
    safety_model: str = "answerdotai/ModernBERT-base"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"


def sft_command(plan: TrainingPlan24GB) -> str:
    return (
        "python -m trl.sft "
        f"--model_name_or_path {plan.base_model} "
        "--dataset_name data/train/sft.jsonl "
        "--dataset_text_field text "
        f"--output_dir {plan.sft_output_dir} "
        "--load_in_4bit true --bf16 true "
        "--lora_r 32 --lora_alpha 64 --lora_dropout 0.05 "
        "--learning_rate 2e-4 --num_train_epochs 2 "
        "--per_device_train_batch_size 1 --gradient_accumulation_steps 16 "
        "--max_seq_length 2048 --gradient_checkpointing true"
    )


def dpo_command(plan: TrainingPlan24GB) -> str:
    return (
        "python -m trl.dpo "
        f"--model_name_or_path {plan.sft_output_dir} "
        "--dataset_name data/train/dpo_pairs.jsonl "
        f"--output_dir {plan.dpo_output_dir} "
        "--load_in_4bit true --bf16 true "
        "--beta 0.1 --learning_rate 5e-5 --num_train_epochs 1 "
        "--per_device_train_batch_size 1 --gradient_accumulation_steps 16 "
        "--max_length 2048 --gradient_checkpointing true"
    )


def classifier_command(plan: TrainingPlan24GB, task: str) -> str:
    model = plan.safety_model if task == "safety" else plan.emotion_model
    return (
        "python scripts/training/train_classifier.py "
        f"--task {task} --model_name {model} "
        f"--train_file data/train/{task}.jsonl --val_file data/val/{task}.jsonl "
        f"--output_dir artifacts/{task}-classifier "
        "--per_device_train_batch_size 16 --num_train_epochs 3 --learning_rate 2e-5"
    )


def eval_command() -> str:
    return (
        "python scripts/training/run_eval_suite.py "
        "--predictions artifacts/eval/preds.jsonl --thresholds scripts/training/thresholds.json"
    )


def main() -> None:
    plan = TrainingPlan24GB()
    print("SFT:")
    print(sft_command(plan))
    print("\nDPO:")
    print(dpo_command(plan))
    print("\nSafety classifier:")
    print(classifier_command(plan, "safety"))
    print("\nEmotion classifier:")
    print(classifier_command(plan, "emotion"))
    print("\nEval:")
    print(eval_command())


if __name__ == "__main__":
    main()
