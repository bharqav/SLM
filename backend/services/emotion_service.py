import hashlib
import json
import re
from dataclasses import dataclass

import httpx

from backend.core import metrics as prom_metrics
from backend.core.config import settings


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float


_ANXIOUS = re.compile(r"\b(anxious|anxiety|worried|panic|stress)\b", re.I)
_SAD = re.compile(r"\b(sad|depressed|down|lonely|hopeless)\b", re.I)
_ANGRY = re.compile(r"\b(angry|furious|mad|irritated)\b", re.I)
_ENGLISH_HINT = re.compile(r"[a-zA-Z]{3,}")


def classify(text: str) -> EmotionResult:
    t = text.strip()
    if not t:
        return EmotionResult("neutral", 0.5)
    if _ANXIOUS.search(t):
        return EmotionResult("anxiety", 0.87)
    if _SAD.search(t):
        return EmotionResult("sadness", 0.82)
    if _ANGRY.search(t):
        return EmotionResult("anger", 0.78)
    return EmotionResult("neutral", 0.74)


def _parse_hf_classification(data: object) -> EmotionResult | None:
    """Best-effort parse for HF text-classification style responses."""
    if isinstance(data, list) and data:
        row = data[0]
        if isinstance(row, dict):
            label = row.get("label") or row.get("class")
            score = row.get("score", 0.75)
            if label:
                return EmotionResult(str(label).replace("LABEL_", ""), float(score))
    if isinstance(data, dict):
        lbl = data.get("label")
        if lbl:
            return EmotionResult(str(lbl), float(data.get("score", 0.75)))
    return None


def _looks_english(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        return True
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in text if ch.isalpha())
    # Lightweight guardrail: if most alphabetic chars are non-ascii, likely non-English.
    if ascii_letters / max(1, letters) < 0.70:
        return False
    return bool(_ENGLISH_HINT.search(text))


def _calibrate(result: EmotionResult) -> EmotionResult:
    if result.confidence < settings.EMOTION_MIN_CONFIDENCE:
        return EmotionResult("neutral", result.confidence)
    return result


def _emotion_cache_key(text: str) -> str:
    h = hashlib.sha256(text.encode()).hexdigest()
    return f"emotion:v1:{h}"


async def classify_async(text: str) -> EmotionResult:
    """Keyword stub, HF Inference when configured, optional Redis cache (PRD)."""
    cache_key = _emotion_cache_key(text)
    if settings.EMOTION_CACHE_ENABLED:
        try:
            from backend.core.redis_client import get_redis

            r = await get_redis()
            cached = await r.get(cache_key)
            if cached:
                prom_metrics.emotion_cache_hits_total.inc()
                data = json.loads(cached)
                return EmotionResult(str(data["label"]), float(data["confidence"]))
        except Exception:
            pass

    result: EmotionResult | None = None
    if settings.EMOTION_ENGLISH_ONLY and not _looks_english(text):
        result = EmotionResult("neutral", 0.50)
    elif settings.EMOTION_HF_URL and settings.HF_API_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(
                    settings.EMOTION_HF_URL,
                    headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
                    json={"inputs": text},
                )
                if r.is_success:
                    parsed = _parse_hf_classification(r.json())
                    if parsed:
                        result = parsed
        except Exception:
            pass
    if result is None:
        result = classify(text)
    result = _calibrate(result)

    if settings.EMOTION_CACHE_ENABLED:
        try:
            from backend.core.redis_client import get_redis

            r = await get_redis()
            await r.setex(
                cache_key,
                settings.EMOTION_CACHE_TTL_SECONDS,
                json.dumps({"label": result.label, "confidence": result.confidence}),
            )
        except Exception:
            pass
    return result
