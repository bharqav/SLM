"""
Combine original empathetic dialogues with Qwen-generated high-quality responses
into a single training binary for the SLM.

Input:
  slm-ft/data/sft_train.jsonl (original dialogues)
  slm-ft/data/sft_val.jsonl
  slm-training/qwen_empathy_responses.jsonl (Qwen responses)

Output:
  slm-training/empathy_train.bin
  slm-training/empathy_val.bin
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EOS_ID = 50256


def format_conversation(messages: list[dict]) -> str:
    """Convert chat messages to a plain-text training record."""
    parts = []
    for msg in messages:
        msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue
        parts.append(content)
    return "\n".join(parts)


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


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(1337)

    # Load original data
    train_path = os.path.join(REPO_ROOT, "slm-ft", "data", "sft_train.jsonl")
    val_path = os.path.join(REPO_ROOT, "slm-ft", "data", "sft_val.jsonl")

    print("Loading original training data...")
    train_texts = load_jsonl(train_path)
    print(f"  {len(train_texts):,} training conversations")

    print("Loading original validation data...")
    val_texts = load_jsonl(val_path)
    print(f"  {len(val_texts):,} validation conversations")

    # Load Qwen-generated responses
    qwen_path = os.path.join(HERE, "qwen_empathy_responses.jsonl")
    qwen_texts = []
    if os.path.exists(qwen_path):
        print("Loading Qwen-generated responses...")
        with open(qwen_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = f"{rec['prompt']}\n{rec['response']}"
                qwen_texts.append(text)
        print(f"  {len(qwen_texts):,} Qwen-generated conversations")

    # Combine: original + Qwen (oversampled for better style learning)
    # Oversample Qwen responses 10x to make them more prominent in training
    oversampled_qwen = qwen_texts * 10
    combined_train = train_texts + oversampled_qwen
    rng.shuffle(combined_train)

    print(f"\nCombined training: {len(combined_train):,} conversations")
    print(f"  Original: {len(train_texts):,}")
    print(f"  Qwen (10x oversampled): {len(oversampled_qwen):,}")

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
    print("\nEncoding combined training data...")
    train_tokens = encode_texts(combined_train)
    print(f"  {len(train_tokens):,} tokens in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Encoding validation data...")
    val_tokens = encode_texts(val_texts)
    print(f"  {len(val_tokens):,} tokens in {time.time() - t0:.1f}s")

    train_out = os.path.join(HERE, "empathy_train.bin")
    val_out = os.path.join(HERE, "empathy_val.bin")

    train_tokens.tofile(train_out)
    val_tokens.tofile(val_out)

    print("\nWrote:")
    print(
        f"  {train_out} ({len(train_tokens):,} tokens, {len(train_tokens) * 2 / 1e6:.1f} MB)"
    )
    print(
        f"  {val_out} ({len(val_tokens):,} tokens, {len(val_tokens) * 2 / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
