from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceResult:
    """Single outcome from the model gateway (managed or stub)."""

    text: str
    provider: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    fallback_reason: str | None = None
    latency_seconds: float = 0.0
    used_tier: str = "primary"  # primary | quality | fast | canary | fallback | served | local | degraded | stub


@dataclass
class GatewayTrace:
    """Structured per-turn trace for logs, dashboards, and Message flags."""

    model_id: str
    prompt_version: str
    safety_ruleset_version: str
    gateway_policy_version: str
    provider: str
    tier: str
    fallback_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_moderation_extra(self) -> dict[str, Any]:
        return {
            "gateway": {
                "model_id": self.model_id,
                "prompt_version": self.prompt_version,
                "safety_ruleset_version": self.safety_ruleset_version,
                "gateway_policy_version": self.gateway_policy_version,
                "provider": self.provider,
                "tier": self.tier,
                "fallback_reason": self.fallback_reason,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "latency_seconds": round(self.latency_seconds, 4),
                **self.extra,
            }
        }
