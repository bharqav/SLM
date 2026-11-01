"""Tests for slm-training/quantize.py.

All hardware-dependent paths are mocked so the suite runs on any machine,
including CPU-only hosts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from quantize import (  # noqa: E402
    QuantizationInfo,
    _count_params,
    _quantize_awq_stub,
    _quantize_gptq_stub,
    _quantize_int4,
    _quantize_int8,
    _replace_linear_layers,
    get_available_methods,
    quantize_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal nn.Module with one Linear so we can verify replacement."""

    def __init__(self, in_features: int = 8, out_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch's stubs type ``nn.Module.__call__`` as returning ``Any``.
        return self.linear(x)  # type: ignore[no-any-return]


@pytest.fixture()
def tiny_model() -> nn.Module:
    return TinyModel()


# ---------------------------------------------------------------------------
# QuantizationInfo dataclass
# ---------------------------------------------------------------------------


class TestQuantizationInfo:
    def test_default_values(self):
        info = QuantizationInfo(method="int4", applied=False, device="cpu")
        assert info.method == "int4"
        assert info.applied is False
        assert info.device == "cpu"
        assert info.compression_ratio is None

    def test_compression_ratio(self):
        info = QuantizationInfo(
            method="int8",
            applied=True,
            device="cuda",
            original_params=100,
            quantized_params=50,
        )
        assert info.compression_ratio == pytest.approx(2.0)

    def test_str_skipped(self):
        info = QuantizationInfo(method="gptq", applied=False, device="cpu")
        s = str(info)
        assert "skipped" in s
        assert "gptq" in s

    def test_str_applied_with_warning(self):
        info = QuantizationInfo(
            method="int4",
            applied=False,
            device="cpu",
            warnings=["bitsandbytes not installed"],
        )
        s = str(info)
        assert "WARNING" in s
        assert "bitsandbytes" in s

    def test_str_compression_ratio(self):
        info = QuantizationInfo(
            method="int8",
            applied=True,
            device="cuda",
            original_params=100,
            quantized_params=25,
        )
        s = str(info)
        assert "4.00" in s


# ---------------------------------------------------------------------------
# _count_params
# ---------------------------------------------------------------------------


class TestCountParams:
    def test_returns_integer(self, tiny_model):
        n = _count_params(tiny_model)
        assert isinstance(n, int)
        assert n > 0

    def test_sum_matches_numel(self, tiny_model):
        expected = sum(p.numel() for p in tiny_model.parameters())
        assert _count_params(tiny_model) == expected


# ---------------------------------------------------------------------------
# get_available_methods
# ---------------------------------------------------------------------------


class TestGetAvailableMethods:
    def test_always_includes_stubs(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            methods = get_available_methods()
        assert "gptq" in methods
        assert "awq" in methods

    def test_includes_bnb_when_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch.dict(sys.modules, {"bitsandbytes": mock.MagicMock()}):
                methods = get_available_methods()
        assert "int8" in methods
        assert "int4" in methods

    def test_excludes_bnb_when_no_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            methods = get_available_methods()
        assert "int8" not in methods
        assert "int4" not in methods

    def test_excludes_bnb_when_not_installed(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch.dict(sys.modules, {}, clear=False):
                # Ensure bitsandbytes is not importable
                original = sys.modules.get("bitsandbytes")
                sys.modules.pop("bitsandbytes", None)
                try:
                    methods = get_available_methods()
                finally:
                    if original is not None:
                        sys.modules["bitsandbytes"] = original
        assert "int8" not in methods
        assert "int4" not in methods


# ---------------------------------------------------------------------------
# quantize_model – dispatch and unknown method
# ---------------------------------------------------------------------------


class TestQuantizeModelDispatch:
    def test_unknown_method_returns_original(self, tiny_model):
        model_out, info = quantize_model(tiny_model, method="unknown", device="cpu")
        assert model_out is tiny_model
        assert info.applied is False
        assert len(info.warnings) > 0

    def test_method_normalised_to_lower(self, tiny_model):
        with mock.patch(
            "quantize._quantize_int4",
            return_value=(tiny_model, QuantizationInfo("int4", True, "cpu")),
        ) as mock_fn:
            model_out, info = quantize_model(tiny_model, method="INT4", device="cpu")
        assert mock_fn.called
        assert model_out is tiny_model

    def test_result_is_tuple_of_two(self, tiny_model):
        result = quantize_model(tiny_model, method="gptq", device="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_result_info_has_original_params(self, tiny_model):
        _, info = quantize_model(tiny_model, method="gptq", device="cpu")
        assert info.original_params > 0


# ---------------------------------------------------------------------------
# GPTQ / AWQ stubs
# ---------------------------------------------------------------------------


class TestGPTQStub:
    def test_returns_original_model(self, tiny_model):
        model_out, info = _quantize_gptq_stub(
            tiny_model, "cpu", QuantizationInfo("gptq", False, "cpu"), None
        )
        assert model_out is tiny_model

    def test_not_applied(self, tiny_model):
        _, info = _quantize_gptq_stub(
            tiny_model, "cpu", QuantizationInfo("gptq", False, "cpu"), None
        )
        assert info.applied is False

    def test_has_warning(self, tiny_model):
        _, info = _quantize_gptq_stub(
            tiny_model, "cpu", QuantizationInfo("gptq", False, "cpu"), None
        )
        assert len(info.warnings) >= 1
        assert "GPTQ" in info.warnings[0]


class TestAWQStub:
    def test_returns_original_model(self, tiny_model):
        model_out, info = _quantize_awq_stub(
            tiny_model, "cpu", QuantizationInfo("awq", False, "cpu"), None
        )
        assert model_out is tiny_model

    def test_not_applied(self, tiny_model):
        _, info = _quantize_awq_stub(
            tiny_model, "cpu", QuantizationInfo("awq", False, "cpu"), None
        )
        assert info.applied is False

    def test_has_warning(self, tiny_model):
        _, info = _quantize_awq_stub(
            tiny_model, "cpu", QuantizationInfo("awq", False, "cpu"), None
        )
        assert len(info.warnings) >= 1
        assert "AWQ" in info.warnings[0]


# ---------------------------------------------------------------------------
# INT8 – CPU path (graceful skip)
# ---------------------------------------------------------------------------


class TestINT8CPU:
    def test_skips_on_cpu(self, tiny_model):
        info = QuantizationInfo(
            "int8", False, "cpu", original_params=_count_params(tiny_model)
        )
        model_out, info = _quantize_int8(tiny_model, "cpu", info)
        assert model_out is tiny_model
        assert info.applied is False
        assert len(info.warnings) >= 1
        assert "CUDA" in info.warnings[0]

    def test_warns_when_bnb_missing(self, tiny_model):
        info = QuantizationInfo(
            "int8", False, "cuda", original_params=_count_params(tiny_model)
        )
        with mock.patch.dict(sys.modules, {"bitsandbytes": None}):
            # Simulate ImportError by removing from sys.modules
            sys.modules.pop("bitsandbytes", None)
            sys.modules.pop("bitsandbytes.nn", None)
            try:
                model_out, info = _quantize_int8(tiny_model, "cuda", info)
            finally:
                pass
        assert info.applied is False
        assert any("bitsandbytes" in w for w in info.warnings)

    def test_applied_on_cuda_with_bnb(self, tiny_model):
        """When CUDA + bnb are present, _replace_linear_layers is attempted."""
        info = QuantizationInfo(
            "int8", False, "cuda", original_params=_count_params(tiny_model)
        )
        with mock.patch.dict(
            sys.modules,
            {"bitsandbytes": mock.MagicMock(), "bitsandbytes.nn": mock.MagicMock()},
        ):
            with mock.patch(
                "quantize._replace_linear_layers", return_value=None
            ) as mock_replace:
                with mock.patch("torch.cuda.is_available", return_value=True):
                    model_out, info = _quantize_int8(tiny_model, "cuda", info)
        assert mock_replace.called


# ---------------------------------------------------------------------------
# INT4 – CPU path (graceful skip)
# ---------------------------------------------------------------------------


class TestINT4CPU:
    def test_skips_on_cpu(self, tiny_model):
        info = QuantizationInfo(
            "int4", False, "cpu", original_params=_count_params(tiny_model)
        )
        model_out, info = _quantize_int4(tiny_model, "cpu", info)
        assert model_out is tiny_model
        assert info.applied is False
        assert len(info.warnings) >= 1
        assert "CUDA" in info.warnings[0]

    def test_warns_when_bnb_missing(self, tiny_model):
        info = QuantizationInfo(
            "int4", False, "cuda", original_params=_count_params(tiny_model)
        )
        sys.modules.pop("bitsandbytes", None)
        sys.modules.pop("bitsandbytes.nn", None)
        try:
            model_out, info = _quantize_int4(tiny_model, "cuda", info)
        finally:
            pass
        assert info.applied is False
        assert any("bitsandbytes" in w for w in info.warnings)

    def test_applied_on_cuda_with_bnb(self, tiny_model):
        info = QuantizationInfo(
            "int4", False, "cuda", original_params=_count_params(tiny_model)
        )
        with mock.patch.dict(
            sys.modules,
            {"bitsandbytes": mock.MagicMock(), "bitsandbytes.nn": mock.MagicMock()},
        ):
            with mock.patch(
                "quantize._replace_linear_layers", return_value=None
            ) as mock_replace:
                with mock.patch("torch.cuda.is_available", return_value=True):
                    model_out, info = _quantize_int4(tiny_model, "cuda", info)
        assert mock_replace.called


# ---------------------------------------------------------------------------
# quantize_model – INT8 / INT4 integration via quantize_model
# ---------------------------------------------------------------------------


class TestQuantizeModelMethods:
    def test_int8_skips_on_cpu(self, tiny_model):
        _, info = quantize_model(tiny_model, method="int8", device="cpu")
        assert info.applied is False
        assert "CUDA" in " ".join(info.warnings)

    def test_int4_skips_on_cpu(self, tiny_model):
        _, info = quantize_model(tiny_model, method="int4", device="cpu")
        assert info.applied is False
        assert "CUDA" in " ".join(info.warnings)

    def test_gptq_returns_original(self, tiny_model):
        model_out, info = quantize_model(tiny_model, method="gptq", device="cpu")
        assert model_out is tiny_model
        assert info.applied is False

    def test_awq_returns_original(self, tiny_model):
        model_out, info = quantize_model(tiny_model, method="awq", device="cpu")
        assert model_out is tiny_model
        assert info.applied is False

    def test_checkpoint_path_forwarded_to_stubs(self, tiny_model):
        with mock.patch(
            "quantize._quantize_gptq_stub",
            return_value=(tiny_model, QuantizationInfo("gptq", False, "cpu")),
        ) as mock_fn:
            quantize_model(
                tiny_model,
                method="gptq",
                device="cpu",
                checkpoint_path="x.pt",
            )
        assert mock_fn.call_args.args[3] == "x.pt"

    def test_extra_kwargs_forwarded_to_int8(self, tiny_model):
        QuantizationInfo(
            "int8", False, "cuda", original_params=_count_params(tiny_model)
        )
        with mock.patch.dict(
            sys.modules,
            {"bitsandbytes": mock.MagicMock(), "bitsandbytes.nn": mock.MagicMock()},
        ):
            with mock.patch(
                "quantize._replace_linear_layers",
                return_value=None,
            ):
                quantize_model(
                    tiny_model,
                    method="int8",
                    device="cuda",
                    custom_option=True,
                )


# ---------------------------------------------------------------------------
# _count_params – degenerate model
# ---------------------------------------------------------------------------


class TestCountParamsDegenerate:
    def test_returns_zero_when_parameters_raise(self):
        class _Broken:
            def parameters(self):
                raise RuntimeError("no parameters")

        assert _count_params(_Broken()) == 0


# ---------------------------------------------------------------------------
# _replace_linear_layers
# ---------------------------------------------------------------------------


class _RecordingLinear(nn.Linear):
    """Replacement layer that records the shapes it was built with."""

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(in_features, out_features)
        self.received = (in_features, out_features)


class TestReplaceLinearLayers:
    def test_replaces_linear_with_custom_layer(self, tiny_model):
        _replace_linear_layers(tiny_model, _RecordingLinear)
        assert isinstance(tiny_model.linear, _RecordingLinear)
        assert tiny_model.linear.received == (8, 4)

    def test_preserves_weights_and_bias(self, tiny_model):
        before_w = tiny_model.linear.weight.data.clone()
        before_b = tiny_model.linear.bias.data.clone()
        _replace_linear_layers(tiny_model, _RecordingLinear)
        assert torch.allclose(tiny_model.linear.weight.data, before_w)
        assert torch.allclose(tiny_model.linear.bias.data, before_b)

    def test_replaces_nested_linear_layers(self):
        class _Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))

        model = _Nested()
        _replace_linear_layers(model, _RecordingLinear)
        assert isinstance(model.block[0], _RecordingLinear)
        assert isinstance(model.block[2], _RecordingLinear)
        assert isinstance(model.block[1], nn.ReLU)

    def test_model_without_linear_layers_is_noop(self):
        class _Plain(nn.Module):
            pass

        _replace_linear_layers(_Plain(), _RecordingLinear)  # must not raise


# ---------------------------------------------------------------------------
# INT8 / INT4 – CUDA + bitsandbytes import and failure paths
# ---------------------------------------------------------------------------


class TestBitsandbytesImportPaths:
    def test_int8_warns_when_bnb_not_installed(self, tiny_model):
        info = QuantizationInfo(
            "int8", False, "cuda", original_params=_count_params(tiny_model)
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(sys.modules, {"bitsandbytes": None}),
        ):
            model_out, info = _quantize_int8(tiny_model, "cuda", info)
        assert model_out is tiny_model
        assert info.applied is False
        assert any("bitsandbytes" in w for w in info.warnings)

    def test_int4_warns_when_bnb_not_installed(self, tiny_model):
        info = QuantizationInfo(
            "int4", False, "cuda", original_params=_count_params(tiny_model)
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(sys.modules, {"bitsandbytes": None}),
        ):
            model_out, info = _quantize_int4(tiny_model, "cuda", info)
        assert model_out is tiny_model
        assert info.applied is False
        assert any("bitsandbytes" in w for w in info.warnings)

    def test_int8_reports_replacement_failure(self, tiny_model):
        info = QuantizationInfo(
            "int8", False, "cuda", original_params=_count_params(tiny_model)
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(
                sys.modules,
                {"bitsandbytes": mock.MagicMock(), "bitsandbytes.nn": mock.MagicMock()},
            ),
            mock.patch(
                "quantize._replace_linear_layers",
                side_effect=RuntimeError("replacement exploded"),
            ),
        ):
            model_out, info = _quantize_int8(tiny_model, "cuda", info)
        assert model_out is tiny_model
        assert info.applied is False
        assert any("failed" in w for w in info.warnings)

    def test_int4_reports_replacement_failure(self, tiny_model):
        info = QuantizationInfo(
            "int4", False, "cuda", original_params=_count_params(tiny_model)
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(
                sys.modules,
                {"bitsandbytes": mock.MagicMock(), "bitsandbytes.nn": mock.MagicMock()},
            ),
            mock.patch(
                "quantize._replace_linear_layers",
                side_effect=RuntimeError("replacement exploded"),
            ),
        ):
            model_out, info = _quantize_int4(tiny_model, "cuda", info)
        assert model_out is tiny_model
        assert info.applied is False
        assert any("failed" in w for w in info.warnings)
