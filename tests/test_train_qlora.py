"""Tests for the QLoRA fine-tuning data path (GPU-free).

`train_qlora.py` loads a 0.5B base model and a tokenizer inside `main()`, so
these tests exercise the pure helpers -- argument defaults, assistant-only loss
masking, dialog filtering and collation -- with a deterministic fake tokenizer
instead of downloading Qwen2.5-0.5B-Instruct.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-ft"))

from train_qlora import (  # noqa: E402
    SFTDataset,
    bitsandbytes_hint,
    build_example,
    collate,
    parse_args,
)

EOS = "<|end|>"


class FakeTokenizer:
    """Word-level stand-in for the Qwen tokenizer.

    Token ids come from a monotonic counter, so prompt and reply ids are
    predictable and never collide.
    """

    def __init__(self) -> None:
        self.eos_token = EOS
        self.pad_token_id = 0
        self.calls: list[str] = []
        self._next = 1

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool, tokenize: bool
    ) -> str:
        assert tokenize is False
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant: ")
        return "\n".join(parts)

    def __call__(
        self, text: str, add_special_tokens: bool = False
    ) -> dict[str, list[int]]:
        self.calls.append(text)
        n = len(text.split())
        ids = list(range(self._next, self._next + n))
        self._next += n
        return {"input_ids": ids}


def _dialog(*turns: tuple[str, str]) -> list[dict]:
    return [{"role": role, "content": text} for role, text in turns]


def _write_jsonl(path: Path, dialogs: list[list[dict]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for d in dialogs:
            fh.write(json.dumps({"messages": d}) + "\n")


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_qlora.py"])
        args = parse_args()
        assert args.lora_r == 16
        assert args.lora_alpha == 32
        assert (
            args.target_modules
            == "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        )
        assert args.epochs == 2.0
        assert args.batch == 1
        assert args.accum == 16
        assert args.data.endswith("sft_train.jsonl")
        assert args.val.endswith("sft_val.jsonl")

    def test_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_qlora.py",
                "--lora-r",
                "8",
                "--lora-alpha",
                "8",
                "--target-modules",
                "q_proj,v_proj",
                "--epochs",
                "3",
                "--limit",
                "10",
            ],
        )
        args = parse_args()
        assert args.lora_r == 8
        assert args.lora_alpha == 8
        assert args.target_modules == "q_proj,v_proj"
        assert args.epochs == 3.0
        assert args.limit == 10


class TestBitsandbytesHint:
    """The guard must track the real install state, not a hard-coded answer."""

    def test_hint_matches_install_state(self):
        installed = importlib.util.find_spec("bitsandbytes") is not None
        assert (bitsandbytes_hint() is None) == installed

    def test_hint_names_the_windows_environment(self):
        hint = bitsandbytes_hint()
        if hint is None:
            pytest.skip("bitsandbytes is installed in this environment")
        assert "slm-ft" in hint
        assert "pip install bitsandbytes" in hint


class TestBuildExample:
    def test_masks_prompt_and_keeps_reply(self):
        tok = FakeTokenizer()
        ex = build_example(tok, _dialog(("user", "a b"), ("assistant", "c d e")), 64)
        prompt_len = 4  # "user: a b" + "assistant: " from the chat template
        assert len(ex["input_ids"]) == 7
        assert len(ex["labels"]) == 7
        assert ex["labels"][:prompt_len] == [-100] * prompt_len
        assert ex["labels"][prompt_len:] == ex["input_ids"][prompt_len:]
        assert tok.calls[1].endswith(EOS)

    def test_truncation_keeps_masking(self):
        tok = FakeTokenizer()
        msgs = _dialog(("user", "a b c d e f"), ("assistant", "g h i j k l"))
        ex = build_example(tok, msgs, 12)
        assert len(ex["input_ids"]) == 12
        assert ex["labels"][:6] == [-100] * 6
        assert ex["labels"][6:] == ex["input_ids"][6:]

    def test_left_truncates_keeping_tail(self):
        msgs = _dialog(("user", "a b c d e f"), ("assistant", "g h i j k l"))
        full = build_example(FakeTokenizer(), msgs, 256)
        clipped = build_example(FakeTokenizer(), msgs, 5)
        assert len(clipped["input_ids"]) == 5
        assert clipped["input_ids"] == full["input_ids"][-5:]
        assert clipped["labels"] == full["labels"][-5:]
        # cut of 9 lands past the 8-token prompt: every surviving label is a reply id
        assert clipped["labels"] == clipped["input_ids"]

    def test_requires_assistant_final(self):
        with pytest.raises(AssertionError):
            build_example(
                FakeTokenizer(),
                _dialog(("user", "a"), ("assistant", "b"), ("user", "c")),
                64,
            )


class TestSFTDataset:
    def _make(self, tmp_path, dialogs):
        path = tmp_path / "sft.jsonl"
        _write_jsonl(path, dialogs)
        return path

    def test_filters_and_trims(self, tmp_path):
        path = self._make(
            tmp_path,
            [
                _dialog(("user", "a"), ("assistant", "b")),  # kept
                _dialog(("user", "a"), ("assistant", "b"), ("user", "c")),  # trimmed
                _dialog(("user", "a")),  # trims to empty -> dropped
                _dialog(("assistant", "b"), ("assistant", "c")),  # no user -> dropped
                _dialog(("user", "a"), ("user", "b")),  # trims to empty -> dropped
            ],
        )
        ds = SFTDataset(path, FakeTokenizer(), 64)
        assert len(ds) == 2

    def test_limit_caps_rows(self, tmp_path):
        path = self._make(
            tmp_path, [_dialog(("user", "a"), ("assistant", "b"))] * 5
        )
        ds = SFTDataset(path, FakeTokenizer(), 64, limit=3)
        assert len(ds) == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SFTDataset(tmp_path / "nope.jsonl", FakeTokenizer(), 64)


class TestCollate:
    def test_pads_and_masks(self):
        batch = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4], "labels": [4]},
        ]
        ids, labels, attn = collate(batch, pad_id=0)
        assert ids.tolist() == [[1, 2, 3], [4, 0, 0]]
        assert labels.tolist() == [[-100, 2, 3], [4, -100, -100]]
        assert attn.tolist() == [[1, 1, 1], [1, 0, 0]]

    def test_single_item(self):
        ids, labels, attn = collate([{"input_ids": [7], "labels": [7]}], pad_id=0)
        assert ids.tolist() == [[7]]
        assert labels.tolist() == [[7]]
        assert attn.tolist() == [[1]]
