"""Quantization support for the from-scratch SLM.

Supported methods
-----------------
* ``int8``   – Linear8bitL4 via bitsandbytes (requires CUDA + bnb >= 0.41).
* ``int4``   – Linear4bit via bitsandbytes NF4 quantization (requires CUDA + bnb).
* ``gptq``   – Stub: gracefully returns the original model and emits a warning.
* ``awq``    – Stub: gracefully returns the original model and emits a warning.

CPU-only environments (including Windows without CUDA) will fall back to the
original fp32 model with a warning for any quantized path; only the stub paths
(GPTQ/AWQ) are designed to be cross-platform.

Usage
-----
    from quantize import quantize_model, QuantizationInfo, get_available_methods

    info: QuantizationInfo = quantize_model(
        model,
        method="int4",
        device="cuda",
        checkpoint_path="slm-training/best_model.pt",
    )
    print(info)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QuantizationInfo:
    """Metadata returned after a quantization attempt."""

    method: str
    applied: bool
    device: str
    original_params: int = 0
    quantized_params: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float | None:
        if self.original_params and self.quantized_params:
            return self.original_params / self.quantized_params
        return None

    def __str__(self) -> str:
        status = "applied" if self.applied else "skipped"
        lines = [
            f"Quantization(method={self.method!r}, status={status}, device={self.device!r})",
        ]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        if self.compression_ratio:
            lines.append(f"  compression_ratio={self.compression_ratio:.2f}x")
        return "\n".join(lines)


def _count_params(model: Any) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def _replace_linear_layers(model: Any, new_layer_cls: type, **kwargs: Any) -> None:
    """Recursively replace nn.Linear with ``new_layer_cls`` in-place."""
    import torch.nn as nn

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                try:
                    new_module = new_layer_cls(
                        child.in_features, child.out_features, **kwargs
                    )
                    new_module.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        new_module.bias.data = child.bias.data.clone()
                    setattr(module, child_name, new_module)
                except Exception as exc:  # pragma: no cover – hardware-dependent
                    logger.debug("Could not replace %s: %s", name, exc)


def get_available_methods() -> list[str]:
    """Return quantization methods that are actually importable on this host.

    Bitsandbytes methods are only listed when CUDA is available; GPTQ and AWQ
    are always listed because their stubs work cross-platform.
    """
    import torch

    methods: list[str] = ["gptq", "awq"]
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401 – existence check

            methods.extend(["int8", "int4"])
        except ImportError:
            pass
    return methods


def quantize_model(
    model: Any,
    method: str = "int4",
    device: str = "cpu",
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> tuple[Any, QuantizationInfo]:
    """Quantize ``model`` using the requested ``method``.

    Parameters
    ----------
    model:
        An ``nn.Module`` to quantize in-place (a clone is *not* made; callers
        should ``import copy; copy.deepcopy(model)`` beforehand if they need
        to preserve the fp32 original).
    method:
        One of ``"int8"``, ``"int4"``, ``"gptq"``, ``"awq"``.
    device:
        Target device string (``"cpu"`` or ``"cuda"``).  Bitsandbytes-based
        methods silently skip when ``device == "cpu"`` or CUDA is absent.
    checkpoint_path:
        Optional path used to persist a quantized checkpoint for GPTQ/AWQ
        backends (currently unused, reserved for future use).

    Returns
    -------
    (quantized_model, info)
        ``quantized_model`` is the (possibly modified) model.  ``info`` carries
        metadata including whether quantization was actually applied.
    """
    method = method.lower().strip()
    info = QuantizationInfo(
        method=method,
        applied=False,
        device=device,
        original_params=_count_params(model),
    )

    if method == "int8":
        model, info = _quantize_int8(model, device, info, **kwargs)
    elif method == "int4":
        model, info = _quantize_int4(model, device, info, **kwargs)
    elif method == "gptq":
        model, info = _quantize_gptq_stub(
            model, device, info, checkpoint_path, **kwargs
        )
    elif method == "awq":
        model, info = _quantize_awq_stub(model, device, info, checkpoint_path, **kwargs)
    else:
        info.warnings.append(
            f"Unknown quantization method: {method!r}. Returning original model."
        )
        logger.warning("Unknown quantization method %r – returning fp32 model.", method)

    info.quantized_params = _count_params(model)
    return model, info


# ---------------------------------------------------------------------------
# bitsandbytes-backed methods
# ---------------------------------------------------------------------------


def _quantize_int8(
    model: Any,
    device: str,
    info: QuantizationInfo,
    **kwargs: Any,
) -> tuple[Any, QuantizationInfo]:
    import torch

    if device == "cpu" or not torch.cuda.is_available():
        info.warnings.append(
            "INT8 quantization requires CUDA + bitsandbytes; "
            "returning original fp32 model."
        )
        logger.warning("INT8 skipped: no CUDA available.")
        return model, info

    try:
        import bitsandbytes as bnb  # noqa: F401
        from bitsandbytes.nn import Linear8bitL4  # noqa: F401
    except ImportError:
        info.warnings.append(
            "bitsandbytes not installed; install with: pip install bitsandbytes"
        )
        logger.warning("INT8 skipped: bitsandbytes not installed.")
        return model, info

    try:
        _replace_linear_layers(
            model,
            Linear8bitL4,
            bias=kwargs.get("bias", "yes"),
            has_fp16_weights=kwargs.get("has_fp16_weights", False),
        )
        info.applied = True
        logger.info("INT8 quantization applied successfully.")
    except Exception as exc:
        info.warnings.append(
            f"INT8 quantization failed: {exc}. Returning original model."
        )
        logger.exception("INT8 quantization failed.")

    return model, info


def _quantize_int4(
    model: Any,
    device: str,
    info: QuantizationInfo,
    **kwargs: Any,
) -> tuple[Any, QuantizationInfo]:
    import torch

    if device == "cpu" or not torch.cuda.is_available():
        info.warnings.append(
            "INT4 quantization requires CUDA + bitsandbytes; "
            "returning original fp32 model."
        )
        logger.warning("INT4 skipped: no CUDA available.")
        return model, info

    try:
        import bitsandbytes as bnb  # noqa: F401
        from bitsandbytes.nn import Linear4bit  # noqa: F401
    except ImportError:
        info.warnings.append(
            "bitsandbytes not installed; install with: pip install bitsandbytes"
        )
        logger.warning("INT4 skipped: bitsandbytes not installed.")
        return model, info

    try:
        _replace_linear_layers(
            model,
            Linear4bit,
            bias=kwargs.get("bias", "yes"),
            compute_type=kwargs.get("compute_type", torch.bfloat16),
            quant_type=kwargs.get("quant_type", "nf4"),
        )
        info.applied = True
        logger.info("INT4 quantization applied successfully.")
    except Exception as exc:
        info.warnings.append(
            f"INT4 quantization failed: {exc}. Returning original model."
        )
        logger.exception("INT4 quantization failed.")

    return model, info


# ---------------------------------------------------------------------------
# Stub methods – graceful degradation when packages are missing
# ---------------------------------------------------------------------------


def _quantize_gptq_stub(
    model: Any,
    device: str,
    info: QuantizationInfo,
    checkpoint_path: str | None,
    **kwargs: Any,
) -> tuple[Any, QuantizationInfo]:
    info.warnings.append(
        "GPTQ quantization is not yet implemented for this model. "
        "Requires the 'auto-gptq' package and a calibration dataset. "
        "Returning original fp32 model."
    )
    logger.warning("GPTQ stub invoked – returning original model.")
    return model, info


def _quantize_awq_stub(
    model: Any,
    device: str,
    info: QuantizationInfo,
    checkpoint_path: str | None,
    **kwargs: Any,
) -> tuple[Any, QuantizationInfo]:
    info.warnings.append(
        "AWQ quantization is not yet implemented for this model. "
        "Requires the 'awq' (AutoAWQ) package and a calibration dataset. "
        "Returning original fp32 model."
    )
    logger.warning("AWQ stub invoked – returning original model.")
    return model, info
