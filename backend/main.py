from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import JSONResponse

from backend.api.v1.health import router as health_router
from backend.api.v1.router import api_router
from backend.core.config import settings
from backend.core.logging_setup import configure_logging
from backend.core.redis_client import close_redis
from backend.middleware.request_id import RequestIdMiddleware
from backend.middleware.tracing import TracingMiddleware

import logging

logger = logging.getLogger(__name__)

_PROD_ENVS = {"prod", "production", "live"}


def _is_production() -> bool:
    return settings.APP_ENV.strip().lower() in _PROD_ENVS


def _cors_origins() -> list[str]:
    return [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]


def _ensure_jwt_secret() -> None:
    if settings.JWT_SECRET:
        return
    import secrets

    settings.JWT_SECRET = secrets.token_urlsafe(48)
    logger.warning(
        "JWT_SECRET not configured: generated a per-process random secret. "
        "Tokens will be invalidated on restart — set JWT_SECRET for any multi-process or persistent deployment."
    )


def _validate_security_settings() -> None:
    if not _is_production():
        if not settings.JWT_SECRET:
            _ensure_jwt_secret()
        if "*" in _cors_origins():
            logger.warning("Wildcard CORS enabled — dev only.")
        return
    if not settings.JWT_SECRET or len(settings.JWT_SECRET) < 32:
        raise RuntimeError("In production, JWT_SECRET must be set and at least 32 chars.")
    if not settings.CORS_ORIGINS:
        raise RuntimeError("In production, CORS_ORIGINS must be explicitly configured.")
    origins = _cors_origins()
    if "*" in origins:
        raise RuntimeError("Wildcard CORS is not allowed in production.")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Basic, API-safe CSP; tighten further when serving frontend from same host.
        response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if _is_production():
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_security_settings()
    configure_logging(use_json=settings.LOG_JSON)
    if settings.SENTRY_DSN:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.1)
    yield
    await close_redis()


app = FastAPI(
    title="Emotional SLM API",
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "auth", "description": "JWT + refresh (OAuth roadmap under /auth/oauth)."},
        {"name": "chat", "description": "REST, SSE, WebSocket; idempotency via Idempotency-Key."},
        {"name": "legal", "description": "Product scope and disclaimers (governance)."},
        {"name": "meta", "description": "SLO targets and governance pointers."},
    ],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    detail = exc.detail
    msg = detail if isinstance(detail, str) else str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": msg,
                "request_id": rid,
                "details": {},
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "request_id": rid,
                "details": exc.errors(),
            }
        },
    )


class MetricsAuthMiddleware(BaseHTTPMiddleware):
    """Optionally gate the Prometheus scrape endpoint with a bearer token."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics" and settings.METRICS_BEARER_TOKEN:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {settings.METRICS_BEARER_TOKEN}":
                return JSONResponse(status_code=401, content={"error": {"code": "HTTP_401", "message": "Not authenticated"}})
        return await call_next(request)


Instrumentator(should_group_status_codes=True).instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MetricsAuthMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(TracingMiddleware)

app.include_router(health_router)
app.include_router(api_router, prefix="/api/v1")