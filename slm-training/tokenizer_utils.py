from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_tokenizer_metadata(
    checkpoint_path: str | Path,
    encoding_name: str = "gpt2",
    eot_token: int = 50256,
    vocab_size: int = 50257,
) -> Path:
    """Save tokenizer metadata alongside model checkpoint for reproducible serving."""
    path = Path(checkpoint_path)
    tokenizer_path = path.with_suffix(".tokenizer.json")
    payload: dict[str, Any] = {
        "encoding_name": encoding_name,
        "eot_token": eot_token,
        "vocab_size": vocab_size,
    }
    tokenizer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return tokenizer_path


def load_tokenizer_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load tokenizer metadata from checkpoint directory."""
    path = Path(checkpoint_path)
    tokenizer_path = path.with_suffix(".tokenizer.json")
    if not tokenizer_path.exists():
        return {
            "encoding_name": "gpt2",
            "eot_token": 50256,
            "vocab_size": 50257,
        }
    return json.loads(tokenizer_path.read_text(encoding="utf-8"))
