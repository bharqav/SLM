import logging
from uuid import UUID

from fastapi import HTTPException, Request, status

from backend.core.config import settings
from backend.core.redis_client import get_redis

logger = logging.getLogger(__name__)


def client_ip(request: Request) -> str:
    """Client IP honoring X-Forwarded-For only when TRUST_PROXY_X_FORWARDED_FOR is set."""
    if settings.TRUST_PROXY_X_FORWARDED_FOR:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    return request.client.host if request.client else "unknown"


async def _consume(key: str, limit: int, raise_on_over: bool) -> None:
    """INCR/EXPIRE fixed window. Returns without raising when `raise_on_over` is False."""
    try:
        r = await get_redis()
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, 60)
        if n > limit:
            if raise_on_over:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": "60"},
                )
            logger.warning("rate limit exceeded (non-blocking) key=%s n=%s", key, n)
    except HTTPException:
        raise
    except Exception:
        # Availability-first: a Redis outage must not 500 every rate-limited
        # endpoint on a crisis-support service; log loudly and allow through.
        logger.exception("rate limiter unavailable; failing OPEN key=%s", key)


async def enforce_rate_limit(request: Request, key_id: str, *, limit: int | None = None) -> None:
    await _consume(
        f"rl:{key_id}:{request.url.path}",
        limit or settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        raise_on_over=True,
    )


async def enforce_rate_limit_ip(request: Request, *, limit: int | None = None) -> None:
    await enforce_rate_limit(request, f"ip:{client_ip(request)}", limit=limit)


async def enforce_rate_limit_user(request: Request, user_id: UUID) -> None:
    await enforce_rate_limit(request, f"user:{user_id}")


async def enforce_rate_limit_path(key_id: str, path: str, *, limit: int | None = None) -> None:
    """Variant for non-HTTP contexts (WebSocket) without a Request object."""
    await _consume(f"rl:{key_id}:{path}", limit or settings.RATE_LIMIT_REQUESTS_PER_MINUTE, raise_on_over=False)
