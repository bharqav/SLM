"""
Model gateway: single place for provider calls, fallback chain, circuit breaker, token accounting.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import AsyncIterator
from uuid import UUID

from backend.core import metrics as prom_metrics
from backend.core.canary import user_in_canary_cohort
from backend.core.config import settings
from backend.core.request_context import request_id_ctx
from backend.services.model_gateway import circuit as cb
from backend.services.model_gateway.providers import degraded_generate, hf_generate, stub_generate, stream_words
from backend.services.model_gateway.types import GatewayTrace, InferenceResult

logger = logging.getLogger("ai.gateway")

gateway_trace_ctx: contextvars.ContextVar[GatewayTrace | None] = contextvars.ContextVar(
    "gateway_trace",
    default=None,
)


def get_last_gateway_trace() -> GatewayTrace | None:
    return gateway_trace_ctx.get()


def _build_trace(res: InferenceResult) -> GatewayTrace:
    return GatewayTrace(
        model_id=res.model_name,
        prompt_version=settings.PROMPT_VERSION,
        safety_ruleset_version=settings.SAFETY_RULESET_VERSION,
        gateway_policy_version=settings.GATEWAY_POLICY_VERSION,
        provider=res.provider,
        tier=res.used_tier,
        fallback_reason=res.fallback_reason,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        latency_seconds=res.latency_seconds,
        extra={"request_id": request_id_ctx.get()},
    )


def _record_tokens(res: InferenceResult) -> None:
    prom_metrics.inference_tokens_total.labels(role="prompt", provider=res.provider).inc(max(0, res.prompt_tokens))
    prom_metrics.inference_tokens_total.labels(role="completion", provider=res.provider).inc(max(0, res.completion_tokens))


def _is_simple_turn(prompt: str) -> bool:
    lines = prompt.count("\n") + 1
    return len(prompt) <= settings.ROUTER_SIMPLE_TURN_MAX_CHARS and lines <= settings.ROUTER_SIMPLE_TURN_MAX_LINES


def _is_complex_turn(prompt: str) -> bool:
    lines = prompt.count("\n") + 1
    return len(prompt) >= settings.ROUTER_COMPLEX_TURN_MIN_CHARS or lines >= settings.ROUTER_COMPLEX_TURN_MIN_LINES


async def generate_full(prompt: str, *, user_id: UUID | None = None) -> InferenceResult:
    """Primary -> fallback HF URL -> stub/degraded safe text."""
    gateway_trace_ctx.set(None)

    if settings.DEGRADED_MODE:
        res = await degraded_generate()
        res.fallback_reason = res.fallback_reason or "degraded_mode_flag"
        prom_metrics.gateway_fallback_total.labels(reason="degraded_mode").inc()
        _record_tokens(res)
        gateway_trace_ctx.set(_build_trace(res))
        logger.info("gateway_tier=%s model=%s", res.used_tier, res.model_name)
        return res

    primary_url: str | None = None
    primary_name = settings.INFERENCE_MODEL_NAME
    selected_tier = "primary"
    if user_id and settings.HF_CANARY_MODEL_URL and user_in_canary_cohort(user_id):
        primary_url = settings.HF_CANARY_MODEL_URL
        primary_name = settings.HF_CANARY_MODEL_NAME
        selected_tier = "canary"
    elif settings.HF_FAST_MODEL_URL and _is_simple_turn(prompt):
        primary_url = settings.HF_FAST_MODEL_URL
        primary_name = settings.HF_FAST_MODEL_NAME
        selected_tier = "fast"
    elif settings.HF_QUALITY_MODEL_URL and _is_complex_turn(prompt):
        primary_url = settings.HF_QUALITY_MODEL_URL
        primary_name = settings.HF_QUALITY_MODEL_NAME
        selected_tier = "quality"
    elif settings.HF_MODEL_URL:
        primary_url = settings.HF_MODEL_URL
        primary_name = settings.INFERENCE_MODEL_NAME
        selected_tier = "primary"

    if not primary_url or not settings.HF_API_TOKEN:
        res = await stub_generate()
        _record_tokens(res)
        gateway_trace_ctx.set(_build_trace(res))
        logger.info("gateway_tier=stub model=%s", res.model_name)
        return res

    res: InferenceResult | None = None
    reason: str | None = None

    if await cb.is_open("primary"):
        prom_metrics.gateway_fallback_total.labels(reason="circuit_open").inc()
        reason = "circuit_open"
    else:
        try:
            res = await hf_generate(primary_url, primary_name, prompt)
            res.used_tier = selected_tier
            _record_tokens(res)
            prom_metrics.gateway_latency_seconds.labels(stage="hf", tier=res.used_tier).observe(res.latency_seconds)
            res.fallback_reason = None
            gateway_trace_ctx.set(_build_trace(res))
            logger.info(
                "gateway_tier=%s model=%s latency_s=%.3f",
                res.used_tier,
                res.model_name,
                res.latency_seconds,
            )
            return res
        except Exception as e:
            logger.warning("primary_inference_failed: %s", e)
            prom_metrics.hf_inference_errors_total.inc()
            prom_metrics.circuit_breaker_open_total.labels(name="primary").inc()
            await cb.trip("primary")
            reason = "primary_failed"

    if settings.HF_FALLBACK_MODEL_URL and settings.HF_API_TOKEN:
        try:
            res = await hf_generate(
                settings.HF_FALLBACK_MODEL_URL,
                settings.HF_FALLBACK_MODEL_NAME,
                prompt,
            )
            res.used_tier = "fallback"
            res.fallback_reason = reason or "fallback_chain"
            prom_metrics.gateway_fallback_total.labels(reason=res.fallback_reason or "fallback").inc()
            _record_tokens(res)
            prom_metrics.gateway_latency_seconds.labels(stage="hf", tier="fallback").observe(res.latency_seconds)
            gateway_trace_ctx.set(_build_trace(res))
            logger.info("gateway_tier=fallback model=%s", res.model_name)
            return res
        except Exception as e:
            logger.warning("fallback_inference_failed: %s", e)

    # Opt-in locally-served fine-tune tier (before the stub): the empathetic
    # Qwen2.5-3B QLoRA model exposed by slm-ft/serve_model.py on localhost.
    if settings.SERVED_MODEL_URL:
        try:
            from backend.services.model_gateway import served_model
            t0_reason = reason or "fallback_chain"
            res = await served_model.served_generate(prompt)
            res.used_tier = "served"
            res.fallback_reason = t0_reason
            prom_metrics.gateway_fallback_total.labels(reason=t0_reason).inc()
            _record_tokens(res)
            prom_metrics.gateway_latency_seconds.labels(stage="served", tier="served").observe(res.latency_seconds)
            gateway_trace_ctx.set(_build_trace(res))
            logger.info("gateway_tier=served model=%s", res.model_name)
            return res
        except Exception as e:
            logger.warning("served_inference_failed: %s", e)

    # Opt-in local artifact tier (before the canned stub): real model output
    # instead of canned text when the local PyTorch artifact is available.
    import asyncio

    from backend.services.model_gateway import local_provider

    if local_provider.is_available():
        try:
            t0_reason = reason or "fallback_chain"
            res = await asyncio.to_thread(local_provider.local_generate, prompt)
            res.used_tier = "local"
            res.fallback_reason = t0_reason
            prom_metrics.gateway_fallback_total.labels(reason=t0_reason).inc()
            _record_tokens(res)
            prom_metrics.gateway_latency_seconds.labels(stage="local", tier="local").observe(res.latency_seconds)
            gateway_trace_ctx.set(_build_trace(res))
            logger.info("gateway_tier=local model=%s", res.model_name)
            return res
        except Exception as e:
            logger.warning("local_torch_inference_failed: %s", e)

    res = await stub_generate()
    res.fallback_reason = reason or "stub_fallback"
    prom_metrics.gateway_fallback_total.labels(reason=res.fallback_reason).inc()
    _record_tokens(res)
    gateway_trace_ctx.set(_build_trace(res))
    logger.info("gateway_tier=stub_after_failure model=%s", res.model_name)
    return res


async def generate_stream(prompt: str, *, user_id: UUID | None = None) -> AsyncIterator[str]:
    """Same routing as generate_full; stream word chunks after full decode (provider batch APIs)."""
    res = await generate_full(prompt, user_id=user_id)
    async for w in stream_words(res.text):
        yield w
