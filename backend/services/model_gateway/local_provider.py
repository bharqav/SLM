"""Local PyTorch provider: serve the repo's trained SLM artifact directly.

Loads `<repo>/slm-training/best_model.pt` (or `final_model.pt`) and generates
greedy continuations in-process. Opt-in via `LOCAL_TORCH_MODEL_ENABLED`
off by default (pipeline.

Design constraints (governance):
- torch + tiktoken import lazily; absent → `is_available()` False → the
  gateway chain behaves exactly as before (external HF → stub) local provider
- Pinned heavy deps in backend the heavy deps are optional-all.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from backend.core import metrics as prom_metrics
from backend.core.config import settings
from backend.services.model_gateway.types import InferenceResult

logger = logging.getLogger("ai.gateway.local")


class LocalModelError(Exception):
    pass


def _artifact_path() -> Path | None:
    """Artifact resolution order: explicit setting → best_model.pt → final_model.pt."""
    override = getattr(settings, "LOCAL_TORCH_MODEL_PATH", None)
    if override:
        p = Path(override)
        return p if p.exists() else None
    training_dir = Path(__file__).resolve().parents[4] / "slm-training"
    best = training_dir / "best_model.pt"
    if best.exists():
        return best
    final = training_dir / "final_model.pt"
    if final.exists():
        return final
    return None


def is_available() -> bool:
    """Cheap readiness probe: enabled flag + torch importable + artifact present."""
    if not getattr(settings, "LOCAL_TORCH_MODEL_ENABLED", False):
        return False
    if not _artifact_path():
        return False
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


_model_cache: dict[str, tuple] = {}


def _load_generation_state():
    import tiktoken
    import torch

    from backend.services.model_gateway.local_model_def import MiniGPT, MiniGPTConfig

    path = _artifact_path()
    if path is None:
        raise LocalModelError("no local model artifact present")
    key = f"{path}:{path.stat().st_mtime}"
    if key in _model_cache:
        return _model_cache[key]

    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg_d = ck.get("model_config", {})
    cfg = MiniGPTConfig(
        vocab_size=int(cfg_d.get("vocab_size", 50257)),
        block_size=int(cfg_d.get("block_size", 256)),
        n_layer=int(cfg_d.get("n_layer", 10)),
        n_head=int(cfg_d.get("n_head", 10)),
        n_embd=int(cfg_d.get("n_embd", 640)),
        dropout=0.0,
        bias=bool(cfg_d.get("bias", True)),
    )
    model = MiniGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    state = (model, enc, int(cfg.block_size))
    _model_cache.clear()
    _model_cache[key] = state
    return state


def local_generate(prompt: str) -> InferenceResult:
    """Greedy generation; blocking — call from the gateway via asyncio.to_thread."""
    import torch

    t0 = time.perf_counter()
    model, enc, block_size = _load_generation_state()
    prompt_ids = enc.encode(prompt, allowed_special=set())
    ids = prompt_ids[-block_size:]
    generated: list[int] = []
    max_new_tokens = min(max(1, settings.MAX_COMPLETION_TOKENS), block_size)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(ids) >= block_size:
                break
            x = torch.tensor([ids], dtype=torch.long)
            logits, _ = model(x)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id == 50256:  # <|endoftext|>
                break
            generated.append(next_id)
            ids.append(next_id)
            text_so_far = enc.decode(generated)
            if text_so_far.count("\n\n") >= 1:
                break
    text = enc.decode(generated).strip()
    latency = time.perf_counter() - t0
    res = InferenceResult(
        text=text,
        provider="local_torch",
        model_name=str(_artifact_path().name),
        prompt_tokens=max(1, len(prompt_ids)),
        completion_tokens=len(generated),
        latency_seconds=latency,
        used_tier="local",
    )
    prom_metrics.inference_tokens_total.labels(role="completion", provider="local_torch").inc(len(generated))
    return res
