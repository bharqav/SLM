"""
Preprocess Qwen-generated empathetic responses into uint16 binary format
for SLM training.

Input: slm-training/qwen_empathy_responses.jsonl
Output: slm-training/qwen_empathy_train.bin / val.bin
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "qwen_empathy_responses.jsonl"
TRAIN_PATH = PROJECT_ROOT / "qwen_empathy_train.bin"
VAL_PATH = PROJECT_ROOT / "qwen_empathy_val.bin"
VAL_FRACTION = 0.2
SEED = 1337
BLOCK_SIZE = 256
EOS = "<|endoftext|>"


def main() -> None:
    random.seed(SEED)
    enc = tiktoken.get_encoding("gpt2")

    records = []
    with INPUT_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            response = rec["response"]
            # Format as dialogue: User prompt, then Assistant response
            text = f"User: {prompt}\nAssistant: {response}{EOS}"
            records.append(text)

    if not records:
        raise RuntimeError(f"No records found in {INPUT_PATH}")

    # Tokenize all texts
    all_ids = []
    for text in records:
        ids = enc.encode(text, allowed_special={EOS})
        all_ids.append(ids)

    # Split train/val
    indices = list(range(len(records)))
    random.shuffle(indices)
    split = max(1, int(math.floor(len(indices) * (1 - VAL_FRACTION))))
    train_indices = indices[:split]
    val_indices = indices[split:]

    def build_dataset(id_list, target_path: Path) -> None:
        # Concatenate all texts with EOS separator
        flat = []
        for ids in id_list:
            flat.extend(ids)
            flat.extend(enc.encode(EOS, allowed_special={EOS}))
        arr = bytearray()
        for x in flat:
            arr.extend(x.to_bytes(2, byteorder="little"))
        target_path.write_bytes(bytes(arr))
        print(f"Wrote {len(flat)} tokens to {target_path}")

    train_ids = [all_ids[i] for i in train_indices]
    val_ids = [all_ids[i] for i in val_indices]
    build_dataset(train_ids, TRAIN_PATH)
    build_dataset(val_ids, VAL_PATH)

    # Stats
    train_tokens = sum(len(ids) for ids in train_ids)
    val_tokens = sum(len(ids) for ids in val_ids)
    print(f"Train dialogs: {len(train_ids)} ({train_tokens} tokens)")
    print(f"Val dialogs: {len(val_ids)} ({val_tokens} tokens)")


if __name__ == "__main__":
    main()
