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
from inference import (  # noqa: E402
    format_prompt,
    get_encoder,
    load_model,
    looks_like_code,
    repetition_ratio,
    truncate_at_next_turn,
)
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


class TestFormatPrompt:
    def test_chat_mode_wraps_in_role_markers(self):
        assert format_prompt("What is the capital of Australia?", "chat") == (
            "User: What is the capital of Australia?\nAssistant:"
        )

    def test_chat_cue_has_no_trailing_space(self):
        """The corpus merges the space after 'Assistant:' into the answer's
        first token, so a trailing space is an out-of-training context."""
        assert format_prompt("Hello", "chat").endswith("\nAssistant:")
        assert not format_prompt("Hello", "chat").endswith(" ")

    def test_code_mode_sends_prompt_verbatim(self):
        assert format_prompt("def preprocess(text):", "code") == "def preprocess(text):"

    def test_auto_routes_code_prompts_verbatim(self):
        assert format_prompt("def preprocess(text):", "auto") == "def preprocess(text):"
        assert (
            format_prompt("SELECT * FROM users WHERE", "auto")
            == "SELECT * FROM users WHERE"
        )

    def test_auto_wraps_questions(self):
        formatted = format_prompt("I have been feeling really down lately", "auto")
        assert formatted == "User: I have been feeling really down lately\nAssistant:"

    def test_chat_mode_overrides_code_shape(self):
        assert format_prompt("def f():", "chat") == "User: def f():\nAssistant:"

    def test_looks_like_code_detects_unbalanced_braces(self):
        assert looks_like_code("return {") is True
        assert looks_like_code("Explain why the sky is blue.") is False


class TestTruncateAtNextTurn:
    def test_cuts_the_following_user_turn(self):
        text = "That sounds hard. What is going on?\nUser: I lost my job."
        assert truncate_at_next_turn(text) == "That sounds hard. What is going on?"

    def test_leaves_a_single_turn_untouched(self):
        assert (
            truncate_at_next_turn("I am sorry to hear that.")
            == "I am sorry to hear that."
        )

    def test_trims_the_leading_space_the_corpus_implies(self):
        assert (
            truncate_at_next_turn(" The capital is Paris.") == "The capital is Paris."
        )

    def test_keeps_an_answer_that_mentions_the_word_user(self):
        assert (
            truncate_at_next_turn("Ask the user for details.")
            == "Ask the user for details."
        )

    def test_cuts_a_second_assistant_turn(self):
        text = (
            "Here is one way to do it.\nAssistant: Alternatively, you could try this."
        )
        assert truncate_at_next_turn(text) == "Here is one way to do it."

    def test_cuts_at_whatever_marker_comes_first(self):
        text = "Answer.\nAssistant: more\nUser: next question"
        assert truncate_at_next_turn(text) == "Answer."


class TestGenerateStopStrings:
    """`stop_strings` ends generation at a decoded marker instead of at the cap."""

    @staticmethod
    def _fixed_logits(idx, idx_cond=None):
        # Every position predicts token 7, so the continuation decodes to "XXXX..."
        logits = torch.zeros(idx.size(0), idx.size(1), 64)
        logits[:, :, 7] = 100.0
        return logits, None

    def _model(self, monkeypatch):
        torch.manual_seed(0)
        model = GPT.from_config(_tiny_config())
        model.eval()
        monkeypatch.setattr(model, "forward", self._fixed_logits)
        return model

    def test_stop_strings_end_generation_early(self, monkeypatch):
        model = self._model(monkeypatch)
        x = torch.tensor([[1, 2, 3]])

        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=20,
                stop_strings=["XXXX"],
                decode_fn=lambda ids: "X" * len(ids),
            )

        assert out.shape[1] == x.shape[1] + 4

    def test_without_stop_strings_generation_runs_to_the_cap(self, monkeypatch):
        model = self._model(monkeypatch)
        x = torch.tensor([[1, 2, 3]])

        with torch.no_grad():
            out = model.generate(x, max_new_tokens=20)

        assert out.shape[1] == x.shape[1] + 20

    def test_stop_strings_without_decode_fn_is_rejected(self, monkeypatch):
        model = self._model(monkeypatch)
        with pytest.raises(ValueError, match="decode_fn"):
            model.generate(torch.tensor([[1]]), max_new_tokens=4, stop_strings=["X"])


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

    def test_repetition_penalty_default_is_passed_to_generate(self):
        """The CLI must forward the repetition penalty, or the model loops."""
        model = _FakeModel()
        with (
            self._load_fake(model),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(
                model, "generate", return_value=torch.tensor([[1, 2, 3, 4]])
            ) as gen,
            mock.patch.object(
                sys, "argv", ["inference.py", "--prompt", "def calculate_"]
            ),
        ):
            assert inference.main() == 0
        assert gen.call_args.kwargs["repetition_penalty"] == 1.2

    def test_repetition_penalty_flag_overrides_default(self):
        model = _FakeModel()
        with (
            self._load_fake(model),
            mock.patch.object(inference, "get_encoder", return_value=_FakeEncoder()),
            mock.patch.object(
                model, "generate", return_value=torch.tensor([[1, 2, 3, 4]])
            ) as gen,
            mock.patch.object(
                sys,
                "argv",
                [
                    "inference.py",
                    "--prompt",
                    "def calculate_",
                    "--repetition-penalty",
                    "1.0",
                ],
            ),
        ):
            assert inference.main() == 0
        assert gen.call_args.kwargs["repetition_penalty"] == 1.0

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
        assert "checkpoint: best_clean_model_v2.pt" in content
