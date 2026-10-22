"""HTTP inference providers (HF-compatible text-generation by default)."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings
from backend.services.model_gateway.types import InferenceResult

logger = logging.getLogger(__name__)

DEGRADED_TEXT = (
    "Thanks for sharing — we're operating in a limited mode right now, "
    "so replies are brief. If you're in crisis, please contact local emergency services or a crisis line."
)
STUB_TEXT = (
    "That sounds really difficult. I'm here with you. "
    "Could you say a bit more about what would feel supportive right now?"
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_hf_text(data: object) -> str:
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"])
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"])
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        ch = data["choices"][0]
        if isinstance(ch, dict) and "text" in ch:
            return str(ch["text"])
    return str(data)


def _clean_generated_text(prompt: str, text: str) -> str:
    out = text.strip()
    # Some HF text-generation models echo the full prompt before completion.
    if out.startswith(prompt):
        out = out[len(prompt) :].strip()
    # Remove repeated whitespace and repeated consecutive lines.
    out = re.sub(r"[ \t]{2,}", " ", out)
    deduped: list[str] = []
    for line in out.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if deduped and deduped[-1].strip().lower() == ln.lower():
            continue
        deduped.append(ln)
    return "\n".join(deduped).strip()


@retry(
    stop=stop_after_attempt(max(1, settings.INFERENCE_MAX_ATTEMPTS)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
)
async def hf_generate(url: str, model_label: str, prompt: str) -> InferenceResult:
    if not settings.HF_API_TOKEN or not url:
        raise RuntimeError("HF not configured")
    t0 = time.perf_counter()
    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    prompt_est = _estimate_tokens(prompt)
    budget = max(16, min(settings.MAX_COMPLETION_TOKENS, 2048))
    dynamic_floor = max(16, settings.MIN_COMPLETION_TOKENS)
    # Slightly larger completions for complex turns while respecting global cap.
    dynamic_target = max(dynamic_floor, min(budget, max(64, prompt_est // 2)))
    params: dict[str, float | int | bool] = {
        "max_new_tokens": dynamic_target,
        "temperature": max(0.0, settings.INFERENCE_TEMPERATURE),
        "top_p": max(0.1, min(settings.INFERENCE_TOP_P, 1.0)),
        "repetition_penalty": max(1.0, settings.INFERENCE_REPETITION_PENALTY),
    }
    params["do_sample"] = bool(params["temperature"] > 0)
    payload = {"inputs": prompt, "parameters": params}
    timeout = float(settings.INFERENCE_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        text = _clean_generated_text(prompt, _parse_hf_text(r.json()))
    lat = time.perf_counter() - t0
    pt = _estimate_tokens(prompt)
    ct = _estimate_tokens(text)
    return InferenceResult(
        text=text,
        provider="huggingface",
        model_name=model_label,
        prompt_tokens=pt,
        completion_tokens=ct,
        latency_seconds=lat,
        used_tier="primary",
    )


async def degraded_generate() -> InferenceResult:
    return InferenceResult(
        text=DEGRADED_TEXT,
        provider=settings.INFERENCE_PROVIDER,
        model_name="degraded-safe",
        prompt_tokens=0,
        completion_tokens=_estimate_tokens(DEGRADED_TEXT),
        fallback_reason="degraded_mode",
        used_tier="degraded",
    )


async def stub_generate() -> InferenceResult:
    return InferenceResult(
        text=STUB_TEXT,
        provider=settings.INFERENCE_PROVIDER,
        model_name=settings.INFERENCE_MODEL_NAME,
        prompt_tokens=0,
        completion_tokens=_estimate_tokens(STUB_TEXT),
        used_tier="stub",
    )


async def stream_words(text: str) -> AsyncIterator[str]:
    for word in text.split():
        yield word + " "


async def hf_stream(url: str, model_label: str, prompt: str) -> AsyncIterator[str]:
    res = await hf_generate(url, model_label, prompt)
    async for w in stream_words(res.text):
        yield w
