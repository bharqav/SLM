"""Prometheus metrics aligned with PRD observability (chat, HF, emotion cache)."""

from prometheus_client import Counter, Histogram

# Chat orchestration (PRD: response time / throughput)
chat_turns_total = Counter(
    "slm_chat_turns_total",
    "Completed synchronous chat turns (non-streaming path)",
)
chat_turn_latency_seconds = Histogram(
    "slm_chat_turn_latency_seconds",
    "End-to-end latency for run_turn (seconds)",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0),
)

# Hugging Face inference (PRD: HF API)
hf_inference_errors_total = Counter(
    "slm_hf_inference_errors_total",
    "HF inference failures after retries",
)

# Model gateway (managed inference + fallback chain)
gateway_fallback_total = Counter(
    "slm_gateway_fallback_total",
    "Inference served after leaving primary tier",
    labelnames=("reason",),
)
gateway_latency_seconds = Histogram(
    "slm_gateway_latency_seconds",
    "Gateway inference latency by stage",
    labelnames=("stage", "tier"),
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)
inference_tokens_total = Counter(
    "slm_inference_tokens_total",
    "Estimated tokens for billing / cost guardrails",
    labelnames=("role", "provider"),
)
circuit_breaker_open_total = Counter(
    "slm_gateway_circuit_breaker_open_total",
    "Times primary inference circuit tripped",
    labelnames=("name",),
)
safety_violation_flags_total = Counter(
    "slm_safety_violation_flags_total",
    "Output or input safety blocks (best-effort)",
    labelnames=("stage",),
)

# Emotion + Redis cache (PRD: caching <20ms)
emotion_cache_hits_total = Counter(
    "slm_emotion_cache_hits_total",
    "Emotion classification served from Redis cache",
)
