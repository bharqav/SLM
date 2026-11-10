"""Tests for merge_adapter.py without downloading the 0.5B base model.

The HF/peft loaders are replaced with fakes, so the merge, save and
safe-serialization path is exercised without network access or a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-ft"))

import merge_adapter  # noqa: E402


class FakeModel:
    def __init__(self) -> None:
        self.saved: list[tuple[Path, dict]] = []

    def save_pretrained(self, path, safe_serialization=False) -> None:
        self.saved.append((Path(path), {"safe_serialization": safe_serialization}))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_text("fake", encoding="utf-8")


class FakePeftModel:
    def __init__(self, merged) -> None:
        self._merged = merged

    def merge_and_unload(self):
        return self._merged


class FakeTokenizer:
    def save_pretrained(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


def _patch_loaders(monkeypatch, merged):
    monkeypatch.setattr(
        merge_adapter.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: FakeModel(),
    )
    monkeypatch.setattr(
        merge_adapter.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        merge_adapter.PeftModel,
        "from_pretrained",
        lambda *a, **k: FakePeftModel(merged),
    )


class TestParseArgs:
    def test_defaults_anchor_to_script_dir(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["merge_adapter.py"])
        args = merge_adapter.parse_args()
        script_dir = PROJECT_ROOT / "slm-ft"
        assert args.base == "Qwen/Qwen2.5-0.5B-Instruct"
        assert Path(args.adapter) == script_dir / "adapter"
        assert Path(args.out) == script_dir / "merged_fp16"

    def test_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["merge_adapter.py", "--base", "base", "--adapter", "ad", "--out", "out"],
        )
        args = merge_adapter.parse_args()
        assert (args.base, args.adapter, args.out) == ("base", "ad", "out")


class TestMain:
    def test_merges_and_writes(self, tmp_path, monkeypatch, capsys):
        merged = FakeModel()
        _patch_loaders(monkeypatch, merged)
        out = tmp_path / "merged_fp16"
        monkeypatch.setattr(
            sys,
            "argv",
            ["merge_adapter.py", "--adapter", "adapter", "--out", str(out)],
        )
        merge_adapter.main()

        assert (out / "model.safetensors").exists()
        assert (out / "tokenizer.json").exists()
        assert merged.saved == [(out, {"safe_serialization": True})]
        assert "merged fp16 model written to" in capsys.readouterr().out
