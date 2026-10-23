"""Train emotion/safety classifiers with Hugging Face Trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--task", choices=["emotion", "safety"], required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(
            "Install training dependencies first: pip install -r scripts/training/requirements-training.txt"
        ) from exc

    train_rows = _read_jsonl(Path(args.train_file))
    val_rows = _read_jsonl(Path(args.val_file))
    if not train_rows:
        raise SystemExit("Train dataset is empty.")

    label_key = "label"
    labels = sorted({str(r[label_key]) for r in train_rows if label_key in r})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    def _to_dataset(rows: list[dict]) -> Dataset:
        texts = [str(r.get("text", "")) for r in rows]
        y = [label2id[str(r[label_key])] for r in rows]
        enc = tok(texts, truncation=True, max_length=512)
        enc["labels"] = y
        return Dataset.from_dict(enc)

    train_ds = _to_dataset(train_rows)
    val_ds = _to_dataset(val_rows)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        bf16=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved classifier to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
