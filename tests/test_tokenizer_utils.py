"""Tests for slm-training/tokenizer_utils.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from tokenizer_utils import (  # noqa: E402
    load_tokenizer_metadata,
    save_tokenizer_metadata,
)


class TestSaveTokenizerMetadata:
    def test_writes_metadata_next_to_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "best_model.pt"
        checkpoint.write_bytes(b"weights")
        path = save_tokenizer_metadata(checkpoint)
        assert path == tmp_path / "best_model.tokenizer.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {
            "encoding_name": "gpt2",
            "eot_token": 50256,
            "vocab_size": 50257,
        }

    def test_custom_values_are_persisted(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"weights")
        path = save_tokenizer_metadata(
            checkpoint, encoding_name="cl100k_base", eot_token=128001, vocab_size=128256
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["encoding_name"] == "cl100k_base"
        assert payload["eot_token"] == 128001
        assert payload["vocab_size"] == 128256

    def test_accepts_str_path(self, tmp_path):
        """train.py hands over os.path.join() strings, not Paths."""
        checkpoint = tmp_path / "best_model.pt"
        checkpoint.write_bytes(b"weights")
        path = save_tokenizer_metadata(str(checkpoint))
        assert path == tmp_path / "best_model.tokenizer.json"
        assert load_tokenizer_metadata(str(checkpoint))["vocab_size"] == 50257


class TestLoadTokenizerMetadata:
    def test_round_trips_saved_metadata(self, tmp_path):
        checkpoint = tmp_path / "best_model.pt"
        save_tokenizer_metadata(checkpoint)
        assert load_tokenizer_metadata(checkpoint)["encoding_name"] == "gpt2"

    def test_missing_file_returns_gpt2_defaults(self, tmp_path):
        payload = load_tokenizer_metadata(tmp_path / "absent.pt")
        assert payload == {
            "encoding_name": "gpt2",
            "eot_token": 50256,
            "vocab_size": 50257,
        }


class TestTrainingWritesTokenizerMetadata:
    """train.py must persist the sidecar whenever it writes a checkpoint.

    slm_api reads it at load time, so a checkpoint trained without the sidecar
    silently falls back to GPT-2 defaults -- wrong for any non-GPT-2 vocab.
    """

    @staticmethod
    def _tiny_model():
        torch = pytest.importorskip("torch")
        from model import GPT, GPTConfig

        cfg = GPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
        )
        return torch, GPT(cfg), cfg

    def test_save_model_only_writes_sidecar(self, tmp_path):
        torch, model, cfg = self._tiny_model()
        from train import save_model_only

        # train.py builds its paths with os.path.join, so it passes str.
        checkpoint = tmp_path / "best_model.pt"
        save_model_only(
            str(checkpoint), model, step=1, best_val=1.5, cfg=cfg, history={}
        )

        payload = load_tokenizer_metadata(checkpoint)
        assert payload["encoding_name"] == "gpt2"
        assert payload["vocab_size"] == 64
        assert payload["eot_token"] == 63

    def test_save_checkpoint_writes_sidecar(self, tmp_path):
        torch, model, cfg = self._tiny_model()
        from train import save_checkpoint

        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        checkpoint = tmp_path / "checkpoint_step_1.pt"
        save_checkpoint(
            str(checkpoint),
            model,
            opt,
            step=1,
            best_val=1.5,
            history={},
            cfg=cfg,
            args=argparse.Namespace(max_steps=1),
            stats={},
        )

        assert load_tokenizer_metadata(checkpoint)["vocab_size"] == 64
