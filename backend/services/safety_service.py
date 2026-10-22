"""Input/output safety checks (local heuristics + optional HF-backed classification).

Design (fail-safe posture):
- Local heuristic backstops (PII, prompt injection) ALWAYS run — no external dependency.
- A configured HF provider raises the fidelity; a provider *failure* never silently
  masquerades as "safe": it is logged at ERROR and, when SAFETY_FAIL_CLOSED is on
  (default), results in a blocking result instead of a pass.
"""

import logging
import re
from decimal import Decimal
from typing import Any

import httpx

from backend.core.config import settings
from backend.services.memory_pii import _CC, _EMAIL, _PHONE, _SSN

logger = logging.getLogger("ai.safety")

# --- Local heuristic backstops -------------------------------------------------

_INJECTION_PATTERNS = re.compile(
    r"(?is)\b(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|messages)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|reveal\s+(?:your\s+)?(?:system\s+prompt|instructions|prompt)"
    r"|print\s+(?:your\s+)?(?:system\s+prompt|instructions)"
    r"|you\s+are\s+now\s+(?:a\s+|an\s+)?(?:dan|developer\s+mode|unfiltered|jailbroken)"
    r"|developer\s+mode\s+(?:enabled|on)"
    r"|\bdan\s+mode\b"
    r"|act\s+as\s+(?:an?\s+)?(?:unfiltered|unrestricted|uncensored)"
    r"|forget\s+(?:everything|all)\s+(?:you|we)"
    r"|override\s+(?:your\s+)?(?:rules|guardrails|safety)"
    r")\b",
    re.I,
)

# Assist text must never smuggle instructions in role-markup that could be
# re-parsed as conversation turns on the next turn.
_ROLE_IMPERSONATION = re.compile(r"(?m)^\s*(?:User|Assistant|System)\s*:", re.I)

_PII_TYPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", _EMAIL),
    ("phone", _PHONE),
    ("ssn", _SSN),
    ("card", _CC),
)


def _detect_pii(text: str) -> list[str]:
    found: list[str] = []
    for name, rx in _PII_TYPES:
        if rx.search(text):
            found.append(name)
    return found


def _local_input_flags(content: str) -> dict[str, Any]:
    pii = _detect_pii(content)
    injection = bool(_INJECTION_PATTERNS.search(content))
    blocked: list[str] = []
    if injection:
        blocked.append("prompt_injection")
    return {
        "pipeline": "input",
        "jailbreak": injection,
        "pii_detected": bool(pii),
        "pii_types": pii,
        "severity": "high" if injection else ("medium" if pii else "low"),
        "blocked_labels": blocked,
        "heuristic": True,
    }


def _local_output_flags(text: str) -> dict[str, Any]:
    """Output backstop: role impersonation in generated text can hijack later turns."""
    impersonation = bool(_ROLE_IMPERSONATION.search(text))
    blocked: list[str] = []
    if impersonation:
        blocked.append("role_impersonation")
    return {
        "pipeline": "output",
        "harmful": False,
        "self_harm_language": False,
        "role_impersonation": impersonation,
        "severity": "high" if impersonation else "low",
        "blocked_labels": blocked,
        "heuristic": True,
    }


def _risk_thresholds() -> dict[str, float]:
    return {
        "self_harm": settings.SAFETY_SELF_HARM_THRESHOLD,
        "violence": settings.SAFETY_VIOLENCE_THRESHOLD,
        "abuse": settings.SAFETY_ABUSE_THRESHOLD,
        "medical_emergency": settings.SAFETY_MEDICAL_EMERGENCY_THRESHOLD,
        "prompt_injection": settings.SAFETY_PROMPT_INJECTION_THRESHOLD,
    }


def _parse_multilabel_scores(data: object) -> dict[str, float]:
    """Parse common HF output shapes into normalized multi-label scores."""
    scores = {k: 0.0 for k in _risk_thresholds()}
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        if data and isinstance(data[0], list):
            rows = [r for r in data[0] if isinstance(r, dict)]
        else:
            rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        rows = [data]

    aliases = {
        "self_harm": {"self_harm", "self-harm", "suicide", "selfharm"},
        "violence": {"violence", "violent"},
        "abuse": {"abuse", "harassment", "hate"},
        "medical_emergency": {"medical_emergency", "medical", "emergency"},
        "prompt_injection": {"prompt_injection", "jailbreak", "injection"},
    }
    for row in rows:
        raw_label = str(row.get("label", "")).strip().lower().replace("label_", "")
        raw_score = float(row.get("score", 0.0))
        for key, names in aliases.items():
            if raw_label in names:
                scores[key] = max(scores[key], raw_score)
                break
    return scores


def _blocked_labels(scores: dict[str, float]) -> list[str]:
    t = _risk_thresholds()
    return [k for k, v in scores.items() if v >= t[k]]


def _stub_input_check(content: str) -> tuple[Decimal, dict[str, Any]]:
    local = _local_input_flags(content)
    score = Decimal("0.9600") if not local["jailbreak"] else Decimal("0.0000")
    return score, local


def _stub_output_check(text: str) -> tuple[Decimal, dict[str, Any]]:
    local = _local_output_flags(text)
    score = Decimal("0.9700") if not local["role_impersonation"] else Decimal("0.0000")
    return score, local


def _provider_failed_result(
    pipeline: str, stage: str, local_flags: dict[str, Any] | None = None
) -> tuple[Decimal, dict[str, Any]]:
    """Result when a configured safety provider errors.

    Fail-closed (default): blocking score + unavailable label.
    Fail-open (explicit opt-out): stub-safe score but a loud 'degraded' marker so
    callers/logs can distinguish "checked and safe" from "provider down".
    """
    if settings.SAFETY_FAIL_CLOSED:
        merged = dict(local_flags or {})
        merged.update(
            {
                "pipeline": pipeline,
                "provider_unavailable": True,
                "blocked_labels": sorted(
                    set(merged.get("blocked_labels") or []) | {"safety_provider_unavailable"}
                ),
                "severity": "high",
            }
        )
        return Decimal("0.0000"), merged
    logger.error("safety provider %s failed; failing OPEN per SAFETY_FAIL_CLOSED=false", stage)
    local = local_flags or (_local_input_flags("") if pipeline == "input" else _local_output_flags(""))
    score = Decimal("0.0000") if local.get("blocked_labels") else Decimal("0.9600")
    return score, {**local, "degraded": True}


async def check_input(content: str) -> tuple[Decimal, dict[str, Any]]:
    local = _local_input_flags(content)
    if settings.SAFETY_INPUT_HF_URL and settings.HF_API_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    settings.SAFETY_INPUT_HF_URL,
                    headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
                    json={"inputs": content},
                )
                if r.is_success:
                    data = r.json()
                    ml = _parse_multilabel_scores(data)
                    blocked = _blocked_labels(ml)
                    flags: dict[str, Any] = {
                        "pipeline": "input",
                        "risk_scores": ml,
                        "blocked_labels": sorted(set(blocked) | set(local["blocked_labels"])),
                        "jailbreak": bool(ml.get("prompt_injection", 0.0) >= settings.SAFETY_PROMPT_INJECTION_THRESHOLD) or local["jailbreak"],
                        "pii_detected": local["pii_detected"],
                        "pii_types": local["pii_types"],
                        "severity": "high" if blocked else "low",
                    }
                    if any(ml.values()):
                        safe_score = max(0.0, 1.0 - max(ml.values()))
                        return Decimal(str(round(safe_score, 4))), flags
                    # Provider reachable, no risk signals: still apply local backstop score
                    return (Decimal("0.9600") if not local["jailbreak"] else Decimal("0.0000")), flags
                logger.error("safety input provider returned HTTP %s", r.status_code)
                return _provider_failed_result("input", "input", local)
        except Exception as e:
            logger.error("safety input provider call failed: %s", e)
            return _provider_failed_result("input", "input", local)
    return _stub_input_check(content)


async def check_output(text: str) -> tuple[Decimal, dict[str, Any]]:
    local = _local_output_flags(text)
    if settings.SAFETY_OUTPUT_HF_URL and settings.HF_API_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    settings.SAFETY_OUTPUT_HF_URL,
                    headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
                    json={"inputs": text},
                )
                if r.is_success:
                    data = r.json()
                    ml = _parse_multilabel_scores(data)
                    blocked = _blocked_labels(ml)
                    flags: dict[str, Any] = {
                        "pipeline": "output",
                        "risk_scores": ml,
                        "blocked_labels": sorted(set(blocked) | set(local["blocked_labels"])),
                        "harmful": bool(blocked),
                        "role_impersonation": local["role_impersonation"],
                        "severity": "high" if blocked else "low",
                    }
                    if any(ml.values()):
                        safe_score = max(0.0, 1.0 - max(ml.values()))
                        return Decimal(str(round(safe_score, 4))), flags
                    return (Decimal("0.9700") if not local["role_impersonation"] else Decimal("0.0000")), flags
                logger.error("safety output provider returned HTTP %s", r.status_code)
                return _provider_failed_result("output", "output", local)
        except Exception as e:
            logger.error("safety output provider call failed: %s", e)
            return _provider_failed_result("output", "output", local)
    return _stub_output_check(text)


def should_block_input(score: Decimal) -> bool:
    # Fail-closed control for simple scalar deployments (provider-less mode).
    if not settings.SAFETY_FAIL_CLOSED:
        return False
    return float(score) < settings.SAFETY_INPUT_THRESHOLD


def should_block_output(score: Decimal) -> bool:
    if not settings.SAFETY_FAIL_CLOSED:
        return False
    return float(score) < settings.SAFETY_OUTPUT_THRESHOLD


def should_block_from_flags(flags: dict[str, Any]) -> bool:
    blocked = flags.get("blocked_labels")
    if isinstance(blocked, list) and len(blocked) > 0:
        return True
    # Local backstop markers that must always block, regardless of provider mode.
    if flags.get("jailbreak") or flags.get("provider_unavailable"):
        return True
    return False
