"""Redis-backed circuit breaker for primary inference endpoint."""

from __future__ import annotations

import time

from backend.core.config import settings


async def is_open(name: str) -> bool:
    try:
        from backend.core.redis_client import get_redis

        r = await get_redis()
        raw = await r.get(f"gateway:cb:{name}")
        if not raw:
            return False
        until = float(raw)
        return time.time() < until
    except Exception:
        return False


async def trip(name: str, cooldown_seconds: int | None = None) -> None:
    cd = cooldown_seconds if cooldown_seconds is not None else settings.GATEWAY_CB_COOLDOWN_SECONDS
    try:
        from backend.core.redis_client import get_redis

        r = await get_redis()
        until = time.time() + cd
        await r.setex(f"gateway:cb:{name}", cd, str(until))
    except Exception:
        pass


async def reset(name: str) -> None:
    try:
        from backend.core.redis_client import get_redis

        r = await get_redis()
        await r.delete(f"gateway:cb:{name}")
    except Exception:
        pass
