"""Provider for a locally-served fine-tuned model (slm-ft/serve_model.py).

POST {prompt} to http://127.0.0.1:8902/generate with optional bearer token
(SERVE_MODEL_TOKEN). Opt-in via SERVED_MODEL_URL (unset = disabled, chain
behaves exactly as before).
"""

from __future__ import annotations

import logging
import time

import httpx

from backend.core import metrics as prom_metrics
from backend.core.config import settings
from backend.services.model_gateway.types import InferenceResult

logger = logging.getLogger("ai.gateway.served")


async def served_generate(prompt: str, *, timeout_seconds: int | None = None) -> InferenceResult:
    timeout = timeout_seconds or settings.INFERENCE_TIMEOUT_SECONDS
    url = str(settings.SERVED_MODEL_URL).rstrip("/") + "/generate"
    headers = {}
    token = getattr(settings, "SERVE_MODEL_TOKEN", "") or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json={"prompt": prompt, "max_new_tokens": settings.MAX_COMPLETION_TOKENS}, headers=headers)
        r.raise_for_status()
        data = r.json()
    latency = time.perf_counter() - t0
    res = InferenceResult(
        text=str(data.get("text", "")),
        provider="served_local",
        model_name=str(data.get("model", "served")),
        prompt_tokens=max(1, len(prompt) // 4),
        completion_tokens=int(data.get("completion_tokens", 0)),
        latency_seconds=latency,
        used_tier="served",
    )
    prom_metrics.inference_tokens_total.labels(role="completion", provider="served_local").inc(
        res.completion_tokens
    )
    return res
