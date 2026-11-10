"""Tests for slm-training/export_onnx.py.

Uses mocks so the suite runs without ONNX runtime or TensorRT installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "slm-training" / "best_clean_model_v2.pt"


@pytest.fixture()
def slm_checkpoint() -> Path:
    """Skip tests that need the trained checkpoint (gitignored, ~1 GB)."""
    if not CHECKPOINT.exists():
        pytest.skip(f"SLM checkpoint not found: {CHECKPOINT}")
    return CHECKPOINT


class TestExportOnnx:
    """Tests for the ONNX export function."""

    def test_export_creates_file(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import export_onnx

        out = tmp_path / "model.onnx"
        fake_model = MagicMock()

        with patch("export_onnx.torch.onnx.export") as mock_export:
            with patch("export_onnx._load_model", return_value=fake_model):
                export_onnx(
                    checkpoint=tmp_path / "best_model.pt",
                    output=out,
                    max_seq_len=128,
                )

        assert mock_export.called
        # output_path should have been created (parent dirs included)
        assert out.parent.exists()

    def test_export_uses_dynamic_axes(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import export_onnx

        out = tmp_path / "model.onnx"
        fake_model = MagicMock()

        with patch("export_onnx.torch.onnx.export") as mock_export:
            with patch("export_onnx._load_model", return_value=fake_model):
                export_onnx(
                    checkpoint=tmp_path / "best_model.pt",
                    output=out,
                    max_seq_len=64,
                )

        _, kwargs = mock_export.call_args
        axes = kwargs.get("dynamic_axes", {})
        assert "idx" in axes
        assert 0 in axes["idx"] or "batch_size" in axes["idx"].values()
        assert 1 in axes["idx"] or "seq_len" in axes["idx"].values()

    def test_export_calls_model_eval(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import export_onnx

        out = tmp_path / "model.onnx"

        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model

        def mock_load_model(checkpoint):
            fake_model.eval()
            return fake_model

        with patch("export_onnx._load_model", side_effect=mock_load_model):
            with patch("export_onnx.torch.onnx.export"):
                export_onnx(checkpoint=tmp_path / "best_model.pt", output=out)

        fake_model.eval.assert_called_once()

    def test_export_missing_checkpoint_raises(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import export_onnx

        missing = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            export_onnx(checkpoint=missing, output=tmp_path / "out.onnx")

    def test_export_output_path_created(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import export_onnx

        out = tmp_path / "deep" / "nested" / "model.onnx"
        fake_model = MagicMock()

        with patch("export_onnx.torch.onnx.export"):
            with patch("export_onnx._load_model", return_value=fake_model):
                export_onnx(checkpoint=tmp_path / "best_model.pt", output=out)

        assert out.parent.exists()


class TestBuildTensorRT:
    """Tests for the optional TensorRT engine build."""

    def test_returns_none_when_tensorrt_missing(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import build_tensorrt

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.engine"

        with patch.dict(sys.modules, {"tensorrt": None}):
            result = build_tensorrt(onnx_path, engine_path)

        assert result is None

    def test_returns_none_on_parse_failure(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import build_tensorrt

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.engine"

        mock_trt = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse_from_file.return_value = False
        mock_trt.Builder.return_value.create_network.return_value.create_onnx_parser.return_value = mock_parser

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            result = build_tensorrt(onnx_path, engine_path)

        assert result is None

    def test_returns_none_on_build_failure(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import build_tensorrt

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.engine"

        mock_trt = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse_from_file.return_value = True
        mock_trt.Builder.return_value.build_serialized_network.return_value = None

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            result = build_tensorrt(onnx_path, engine_path)

        assert result is None

    def test_returns_engine_path_on_success(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import build_tensorrt

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.engine"

        fake_plan = b"engine-bytes"
        mock_trt = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse_from_file.return_value = True
        mock_builder = mock_trt.Builder.return_value
        mock_builder.build_serialized_network.return_value = fake_plan

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            result = build_tensorrt(onnx_path, engine_path)

        assert result == engine_path
        assert engine_path.read_bytes() == fake_plan


class TestMainCLI:
    """Tests for the CLI entry point."""

    def test_default_checkpoint(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import main

        with patch("export_onnx.export_onnx") as mock_export:
            with patch("export_onnx.build_tensorrt"):
                rc = main([])

        assert rc == 0
        _, kwargs = mock_export.call_args
        assert (
            kwargs["checkpoint"]
            == PROJECT_ROOT / "slm-training" / "best_clean_model_v2.pt"
        )

    def test_trt_flag_enables_tensorrt(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import main

        with patch("export_onnx.export_onnx"):
            with patch("export_onnx.build_tensorrt") as mock_trt:
                rc = main(["--trt"])

        assert rc == 0
        mock_trt.assert_called_once()

    def test_missing_checkpoint_returns_error(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import main

        missing = tmp_path / "nonexistent.pt"
        rc = main(["--checkpoint", str(missing)])
        assert rc == 1

    def test_custom_output_path(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from export_onnx import main

        out = PROJECT_ROOT / "slm-training" / "custom.onnx"

        with patch("export_onnx.export_onnx") as mock_export:
            with patch("export_onnx.build_tensorrt"):
                rc = main(["--output", str(out)])

        assert rc == 0
        _, kwargs = mock_export.call_args
        assert kwargs["output"] == out
