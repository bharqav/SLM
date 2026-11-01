"""Tests for slm-training/inference.py.

The generation path is exercised against a tiny real GPT checkpoint so that
``load_model`` is validated end-to-end, while ``main`` is driven with fakes to
keep the CLI loop fast and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

import inference  # noqa: E402
import slmlog  # noqa: E402
from guardrails import GuardrailResult  # noqa: E402
from inference import get_encoder, load_model, repetition_ratio  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from quantize import QuantizationInfo  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_log_file(tmp_path, monkeypatch):
    """Keep test output out of the repository's train.log."""
    monkeypatch.setattr(slmlog, "LOG_PATH", str(tmp_path / "train.log"))


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        bias=True,
    )


def _save_checkpoint(path: Path, cfg: GPTConfig, drop_keys: bool = False) -> None:
    torch.manual_seed(0)
    model = GPT.from_config(cfg)
    state = model.state_dict()
    if drop_keys:
        state = {k: v for k, v in state.items() if "ln_f" not in k}
    torch.save(
        {
            "model": state,
            "model_config": cfg.to_dict(),
            "step": 10,
            "best_val_loss": 1.5,
        },
        path,
    )


class _FakeModel:
    """Stand-in for GPT that returns a fixed token sequence."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50, **kwargs):
        self.calls += 1
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=idx.device)

    def parameters(self):
        return []


class _FakeEncoder:
    eot_token = 50256

    def encode_ordinary(self, prompt):
        return [1, 2, 3]

    def decode(self, ids):
        return "def calculate_ the sum of two numbers"


class TestLoadModel:
    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "nope.pt"), "cpu")

    def test_relative_checkpoint_resolves_next_to_module(self):
        with pytest.raises(FileNotFoundError) as exc:
            load_model("definitely-missing.pt", "cpu")
        assert "definitely-missing.pt" in str(exc.value)

    def test_loads_tiny_checkpoint(self, tmp_path):
        cfg = _tiny_config()
        path = tmp_path / "tiny.pt"
        _save_checkpoint(path, cfg)
        model, loaded_cfg, ck = load_model(str(path), "cpu")
        assert isinstance(model, GPT)
        assert loaded_cfg.vocab_size == cfg.vocab_size
        assert loaded_cfg.n_embd == cfg.n_embd
        assert ck["step"] == 10
        assert ck["best_val_loss"] == 1.5
        assert not model.training

    def test_missing_keys_warns(self, tmp_path, capsys):
        cfg = _tiny_config()
        path = tmp_path / "tiny.pt"
        _save_checkpoint(path, cfg, drop_keys=True)
        model, _, _ = load_model(str(path), "cpu")
        assert isinstance(model, GPT)
        assert "Missing" in capsys.readouterr().out


class TestGetEncoder:
    def test_returns_gpt2_encoding(self):
        fake = mock.Mock()
        with mock.patch("tiktoken.get_encoding", return_value=fake) as get_enc:
            assert get_encoder() is fake
        get_enc.assert_called_once_with("gpt2")


class TestRepetitionRatio:
    def test_short_text_is_zero(self):
        assert repetition_ratio("one two three") == 0.0

    def test_repeated_text_is_high(self):
        text = "the cat sat on the mat " * 4
        assert repetition_ratio(text) > 0.5

    def test_varied_text_is_low(self):
        text = (
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
        )
        assert repetition_ratio(text) == 0.0


class TestMain:
    def _load_fake(self, model):
        return mock.patch.object(
            inference,
            "load_model",
            return_value=(model, _tiny_config(), {"step": 1, "best_val_loss": 1.0}),
        )

    def test_single_prompt_returns_zero(self, capsys):
        model = _FakeModel()
        with (
            self._load_fake(model),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(
                sys, "argv", ["inference.py", "--prompt", "def calculate_"]
            ),
        ):
            assert inference.main() == 0
        assert model.calls == 1
        assert "PROMPT:" in capsys.readouterr().out

    def test_missing_checkpoint_returns_one(self, capsys):
        with (
            mock.patch.object(
                inference,
                "load_model",
                side_effect=FileNotFoundError("checkpoint not found: nope.pt"),
            ),
            mock.patch.object(sys, "argv", ["inference.py", "--prompt", "hi"]),
        ):
            assert inference.main() == 1
        assert "checkpoint not found" in capsys.readouterr().out

    def test_refused_prompt_returns_one(self):
        unsafe = GuardrailResult(safe=False, reason="Refused")
        with (
            self._load_fake(_FakeModel()),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(inference, "check_input", return_value=unsafe),
            mock.patch.object(sys, "argv", ["inference.py", "--prompt", "bad"]),
        ):
            assert inference.main() == 1

    def test_generation_failure_returns_one(self):
        model = _FakeModel()
        with (
            mock.patch.object(model, "generate", side_effect=RuntimeError("boom")),
            self._load_fake(model),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(
                sys, "argv", ["inference.py", "--prompt", "def calculate_"]
            ),
        ):
            assert inference.main() == 1

    def test_quantize_path_is_exercised(self):
        model = _FakeModel()
        qinfo = QuantizationInfo(
            method="awq", applied=True, device="cpu", original_params=0
        )
        with (
            self._load_fake(model),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(
                inference, "quantize_model", return_value=(model, qinfo)
            ) as quantize_model,
            mock.patch.object(
                sys,
                "argv",
                ["inference.py", "--prompt", "def calculate_", "--quantize", "awq"],
            ),
        ):
            assert inference.main() == 0
        quantize_model.assert_called_once()

    def test_full_suite_writes_samples_file(self, tmp_path):
        samples = tmp_path / "samples.txt"
        with (
            mock.patch.object(inference, "SAMPLES_PATH", str(samples)),
            self._load_fake(_FakeModel()),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(sys, "argv", ["inference.py", "--samples", "1"]),
        ):
            assert inference.main() == 0
        content = samples.read_text(encoding="utf-8")
        assert "Inference samples" in content
        assert "checkpoint: best_model.pt" in content
