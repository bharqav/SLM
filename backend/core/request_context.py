"""Request correlation for structured logs (PRD: JSON logs + request id)."""

import contextvars

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
