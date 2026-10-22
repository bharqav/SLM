"""Model gateway — inference routing, fallback, and observability hooks."""

from backend.services.model_gateway.gateway import (
    gateway_trace_ctx,
    generate_full,
    generate_stream,
    get_last_gateway_trace,
)

__all__ = [
    "gateway_trace_ctx",
    "generate_full",
    "generate_stream",
    "get_last_gateway_trace",
]
