"""POST idempotency for chat turns (governance §9).

Contract:
- The key is bound to a SHA-256 of the request body — replaying the same key
  with *different* content is rejected by the caller (409), not silently
  answered with the first response.
- A reservation is created atomically (Redis SET NX), so two concurrent
  requests with the same key cannot both execute inference; the loser gets a
  conflict instead of a duplicate assistant message.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from backend.core.config import settings
from backend.core.redis_client import get_redis

_INFLIGHT = "__in_flight__"


def _key(user_id: UUID, idempotency_key: str) -> str:
    return f"idemp:chat:v1:{user_id}:{idempotency_key}"


async def begin_chat_request(
    user_id: UUID, idempotency_key: str, body_hash: str
) -> dict[str, Any] | None:
    """Atomically reserve the key.

    Returns:
      - cached response payload (dict) if the key was already completed,
      - {"_in_flight": True} if another request currently holds the key,
      - None if we reserved it (caller MUST call complete_chat_request after).
    """
    key = idempotency_key.strip()
    if not key:
        return None
    try:
        r = await get_redis()
        k = _key(user_id, key)
        # Reserve atomically; losers read the current state instead of racing.
        placed = await r.set(k, _INFLIGHT, ex=settings.IDEMPOTENCY_TTL_SECONDS, nx=True)
        if placed:
            return None
        raw = await r.get(k)
        if not raw:
            # Expired between set-nx and get (unlikely): just report in-flight.
            return {"_in_flight": True}
        value = raw.decode() if hasattr(raw, "decode") else raw
        if value == _INFLIGHT:
            return {"_in_flight": True}
        payload = json.loads(value)
        if payload.get("_body_hash") != body_hash:
            return {"_body_mismatch": True}
        payload.pop("_body_hash", None)
        return payload
    except Exception:
        # Fail open on infrastructure errors (availability), like the limiter.
        return None


_MISMATCH = {"_body_mismatch": True}


def is_body_mismatch(result: dict[str, Any] | None) -> bool:
    return bool(result and result.get("_body_mismatch"))


def is_in_flight(result: dict[str, Any] | None) -> bool:
    return bool(result and result.get("_in_flight"))


async def complete_chat_request(
    user_id: UUID,
    idempotency_key: str,
    body_hash: str,
    payload: dict[str, Any] | None,
) -> None:
    key = idempotency_key.strip()
    if not key:
        return
    try:
        r = await get_redis()
        k = _key(user_id, key)
        if payload is None:
            # Request failed before producing a response: release the hold so a
            # legitimate retry with the same key is not blocked for the full TTL.
            await r.delete(k)
            return
        stored = dict(payload)
        stored["_body_hash"] = body_hash
        await r.setex(
            k,
            settings.IDEMPOTENCY_TTL_SECONDS,
            json.dumps(stored, default=str),
        )
    except Exception:
        pass
