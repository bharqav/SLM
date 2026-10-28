"""
Preprocess empathetic dialogue JSONL into the same uint16 binary format
used by the main SLM training pipeline.

Input:  slm-ft/data/sft_train.jsonl + sft_val.jsonl
        Schema: {"messages": [{"role": "user|assistant", "content": "..."}, ...]}
Output: slm-training/empathy_train.bin + empathy_val.bin
        Flat token stream separated by <|endoftext|> (id 50256).

Each conversation is formatted as:
    User: <user message>
    Assistant: <assistant message>
    User: <user message>
    Assistant: <assistant message>
    <|endoftext|>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
EOS_ID = 50256
VOCAB_SIZE = 50257


def format_conversation(messages: list[dict]) -> str:
    """Convert chat messages to a plain-text training record."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "train_jsonl": os.path.join(REPO_ROOT, "slm-ft", "data", "sft_train.jsonl"),
    "val_jsonl": os.path.join(REPO_ROOT, "slm-ft", "data", "sft_val.jsonl"),
    "out_dir": REPO_ROOT,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-jsonl", default=DEFAULTS["train_jsonl"], help="Path to training JSONL"
    )
    ap.add_argument(
        "--val-jsonl", default=DEFAULTS["val_jsonl"], help="Path to validation JSONL"
    )
    ap.add_argument(
        "--out-dir", default=DEFAULTS["out_dir"], help="Directory for output .bin files"
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(args.seed)

    def load_jsonl(path: str) -> list[str]:
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = format_conversation(obj.get("messages", []))
                if text:
                    records.append(text)
        return records

    print("Loading training data...")
    train_texts = load_jsonl(args.train_jsonl)
    print(f"  {len(train_texts):,} training conversations")

    print("Loading validation data...")
    val_texts = load_jsonl(args.val_jsonl)
    print(f"  {len(val_texts):,} validation conversations")

    # Shuffle training data
    rng.shuffle(train_texts)

    def encode_texts(texts: list[str]) -> np.ndarray:
        tokens = []
        for text in texts:
            ids = enc.encode(
                text + "<|endoftext|>",
                allowed_special={"<|endoftext|>"},
            )
            tokens.extend(ids)
        return np.array(tokens, dtype=np.uint16)

    t0 = time.time()
    print("Encoding training data...")
    train_tokens = encode_texts(train_texts)
    print(f"  {len(train_tokens):,} tokens in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Encoding validation data...")
    val_tokens = encode_texts(val_texts)
    print(f"  {len(val_tokens):,} tokens in {time.time() - t0:.1f}s")

    train_path = os.path.join(args.out_dir, "empathy_train.bin")
    val_path = os.path.join(args.out_dir, "empathy_val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print("\nWrote:")
    print(
        f"  {train_path} ({len(train_tokens):,} tokens, {len(train_tokens) * 2 / 1e6:.1f} MB)"
    )
    print(
        f"  {val_path} ({len(val_tokens):,} tokens, {len(val_tokens) * 2 / 1e6:.1f} MB)"
    )

    # Stats
    train_lens = [
        len(enc.encode(t + "<|endoftext|>", allowed_special={"<|endoftext|>"}))
        for t in train_texts
    ]
    print("\nTraining conversation lengths:")
    print(
        f"  mean: {np.mean(train_lens):.1f} | median: {np.median(train_lens):.1f} | "
        f"min: {min(train_lens)} | max: {max(train_lens)}"
    )


if __name__ == "__main__":
    main()
