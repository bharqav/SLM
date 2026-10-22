"""W3C traceparent propagation + X-Trace-Id for cross-service observability (governance §11)."""

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tp = request.headers.get("traceparent")
        if tp:
            request.state.traceparent = tp
            parts = tp.split("-")
            request.state.trace_id = parts[1] if len(parts) > 1 else None
        else:
            request.state.trace_id = secrets.token_hex(16)
            request.state.traceparent = None
        response = await call_next(request)
        if request.state.traceparent:
            response.headers["traceparent"] = request.state.traceparent
        else:
            response.headers["X-Trace-Id"] = request.state.trace_id or ""
        return response
