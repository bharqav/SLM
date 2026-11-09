"""ONNX export and optional TensorRT conversion for the GPT model.

Usage:
    python -m slm-training.export_onnx \\
        --checkpoint slm-training/best_model.pt \\
        --output model.onnx \\
        [--trt]

The ``--trt`` flag attempts to build a TensorRT engine from the exported ONNX
model.  If the ``tensorrt`` package is not installed, the script prints a
warning and continues without building the engine (graceful degradation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_model(checkpoint: Path):
    """Load the GPT model from a checkpoint file.

    Returns the model in eval mode on CPU.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
    from model import GPT, GPTConfig  # noqa: E402

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = GPTConfig(**(state.get("model_config") or {}))
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def export_onnx(
    checkpoint: Path,
    output: Path,
    max_seq_len: int = 256,
) -> Path:
    """Export the GPT model to ONNX format.

    Args:
        checkpoint: Path to the PyTorch checkpoint file.
        output: Destination path for the ``.onnx`` file.
        max_seq_len: Maximum sequence length used for the dummy input.

    Returns:
        The resolved output path.
    """
    model = _load_model(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randint(
        0,
        model.config.vocab_size,
        (1, max_seq_len),
        dtype=torch.long,
    )

    dynamic_axes = {
        "idx": {0: "batch_size", 1: "seq_len"},
    }

    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["idx"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {output}")
    return output


def build_tensorrt(onnx_path: Path, engine_path: Path) -> Path | None:
    """Attempt to build a TensorRT engine from an ONNX model.

    If the ``tensorrt`` package is unavailable, prints a warning and returns
    ``None`` so callers can degrade gracefully.

    Args:
        onnx_path: Path to the source ONNX model.
        engine_path: Desired path for the serialized TensorRT engine.

    Returns:
        ``engine_path`` on success, ``None`` if TensorRT is unavailable or
        the build fails.
    """
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print(
            "tensorrt is not installed; skipping TensorRT engine build. "
            "Install it from the NVIDIA NGC repository to enable this step."
        )
        return None

    try:
        import tensorrt as trt

        logger = trt.Logger()
        builder = trt.Builder(logger)
        network = builder.create_network(
            flags=trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        parser = network.create_onnx_parser()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

        success = parser.parse_from_file(str(onnx_path))
        if not success:
            print("Failed to parse ONNX model for TensorRT conversion.")
            return None

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("TensorRT engine build failed.")
            return None

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(plan)
        print(f"TensorRT engine saved to {engine_path}")
        return engine_path
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"TensorRT build skipped due to error: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the GPT model to ONNX (and optionally TensorRT)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "slm-training" / "best_clean_model_v2.pt",
        help="Path to the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "slm-training" / "model.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length for the dummy export input.",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Also attempt to build a TensorRT engine.",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    export_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        max_seq_len=args.max_seq_len,
    )

    if args.trt:
        engine_path = args.output.with_suffix(".engine")
        build_tensorrt(args.output, engine_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
